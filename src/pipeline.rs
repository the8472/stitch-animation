use std;
use ffmpeg;
use ffmpeg::frame::Video;
use ffmpeg::codec::threading;
use ffmpeg::ffi::AVPictureType;
use ffmpeg::ffi::AVPictureType::*;
use ffmpeg::software::converter;
use ffmpeg::util::format::pixel::Pixel;
use std::collections::{VecDeque,HashMap};
use std::io::Write;
use ffmpeg::frame;
use std::fmt::*;
use std::path::*;
use motion::vectors::{MVInfo,ToMotionVectors, MVec};
use motion::search::{self, Estimate};
use euclid::rect;
use std::thread;



pub(crate) struct MVFrame {
    mv_info: MVInfo,
    frame: frame::Video,
    idx: u32,
    frame_type: AVPictureType,
    full_scans: HashMap<u32, Estimate>
}

impl MVFrame {
    pub fn new(mv_info: MVInfo, frame: Video, frame_type : AVPictureType, idx: u32) -> Self {
        //let idx = frame.display_number();
        MVFrame { mv_info, frame, frame_type, idx, full_scans: HashMap::new() }
    }

    fn res(&self) -> u32 {
        self.frame.height() * self.frame.width()
    }

    fn predicted_fraction(&self) -> f32 {
        self.mv_info.pred() as f32 / self.res() as f32
    }

    pub fn full_compare(&self, frame_idx: u32) -> Option<Estimate> {
        self.full_scans.get(&frame_idx).cloned()
    }

    fn add_full_compare(&mut self, frame_idx: u32, est: Estimate) {
        if self.full_scans.insert(frame_idx, est) == None {
            if est.error_fraction() < 5.0 && frame_idx == self.idx - 1 {
                let w = self.frame.width() as isize;
                let h = self.frame.height() as isize;
                let Estimate{x,y,..} = est;
                let inter = rect::<_, ::euclid::UnknownUnit>(0,0,w,h).intersection(&rect(x,y,w,h)).unwrap();
                let area = inter.size.width * inter.size.height;
                self.mv_info.add(MVec::new().from_vector(x,y).forward(area as usize))
            }
        }
    }
}

impl Debug for MVFrame {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{} {} {:?} | {:0.3} {:?}", self.idx, self.frame.packet().dts , self.frame_type, self.predicted_fraction(), self.mv_info)?;
        if !self.full_scans.is_empty() {
            write!(f, "\n  {:?}", self.full_scans)?;
        }
        Ok(())
    }
}

pub(crate) struct MVPrefilter {
    pre_queue: VecDeque<MVFrame>
}

impl MVPrefilter {
    pub fn new() -> Self {
        MVPrefilter {pre_queue: VecDeque::with_capacity(3)}
    }

    pub fn add_frame(&mut self, frame: MVFrame) {
        self.pre_queue.push_front(frame);

        if self.pre_queue.len() >= 3 {
            let pq = &mut self.pre_queue;

            let (p0, p1, p2) = unsafe {
                (&mut *(&mut pq[0] as *mut MVFrame),&mut *(&mut pq[1] as *mut MVFrame),&mut *(&mut pq[2] as *mut MVFrame))
            };

            p1.mv_info.populate(Some(&p2.frame),&p1.frame,Some(&p0.frame));

            // I-frames lack forward predictions, try to do a simple global motion search to
            // fill the blanks
            if p1.frame_type == AV_PICTURE_TYPE_I {
                let hint = p0.frame.most_common_vectors();

                let est = search::search(&p1.frame, &p2.frame, hint);
                //println!("I-frame {} err{} area{} x{} y{}", p1.idx, error, area, x, y);

                p1.add_full_compare(p2.idx, est);
                p2.add_full_compare(p1.idx, est.reverse());
            }

            // P and I frames lack back-predictions, transplant
            let pristine = p1.mv_info.clone();

            match p1.frame_type {
                AV_PICTURE_TYPE_P | AV_PICTURE_TYPE_I => {
                    p1.mv_info.transplant_from(Some(&p2.mv_info), None);
                }
                _ => {}
            }

            match p2.frame_type {
                AV_PICTURE_TYPE_P | AV_PICTURE_TYPE_I => {
                    p2.mv_info.transplant_from(None, Some(&pristine));
                }
                _ => {}
            }



        }
    }

    pub fn poll(&mut self) -> Option<MVFrame> {
        if self.pre_queue.len() >= 3 {
            self.pre_queue.pop_back()
        } else {
            None
        }
    }

    pub(crate) fn drain(mut self) -> impl Iterator<Item=MVFrame> {
        if self.pre_queue.len() > 0 {
            let p = &mut self.pre_queue[0];
            p.mv_info.populate(None, &p.frame, None);
        }

        self.pre_queue.into_iter().rev()
    }
}

use ::stitchers::linear::LinStitcher;

struct ImageOut {
    octx: ffmpeg::format::context::Output,
    encoder: ffmpeg::codec::encoder::video::Encoder,
    conv: Option<ffmpeg::software::scaling::context::Context>,
    start_frame: u32,
    last_frame_idx: u32,
    stitcher: LinStitcher,
    dir: PathBuf,
    next_frame: Option<MVFrame>,
    log: std::io::BufWriter<std::fs::File>,
}

impl ImageOut {
    fn write_linear_stitch(self) -> thread::JoinHandle<()> {
        let ImageOut{stitcher, start_frame, dir, ..} = self;

        thread::spawn(move || {
            let frame = stitcher.merge();

            let mut octx = ffmpeg::format::output(&dir.join(format!("{:06}_lin.png", start_frame))).unwrap();
            let codec = ffmpeg::encoder::find_by_name("png").unwrap();
            let mut encoder = octx.add_stream(codec).unwrap().codec().encoder().video().unwrap();
            //output.set_time_base((24, 1000));
            encoder.set_time_base((24, 1000));
            encoder.set_width(frame.width());
            encoder.set_height(frame.height());
            encoder.set_format(Pixel::RGBA);
            encoder.set_compression(Some(0));

            let mut packet = ffmpeg::codec::packet::packet::Packet::empty();
            let mut encoder = encoder.open_as(codec).unwrap();
            match encoder.encode(&frame, &mut packet) {
                Ok(true) => {}
                Ok(false) => {
                    encoder.flush(&mut packet);
                }
                Err(e) => eprintln!("{:?}", e)
            }

            octx.write_header().unwrap();
            packet.write(&mut octx).unwrap();
            octx.write_trailer().unwrap();
        })
    }

    fn next_frame(&mut self, frame: MVFrame) {
        assert!(self.next_frame.is_none());
        self.last_frame_idx = frame.idx;
        self.stitcher.add_frame(frame.frame.clone());
        self.next_frame = Some(frame);
    }

    fn encode(&mut self, stitch: bool, format: Format) {
        let mv_frame = match ::std::mem::replace(&mut self.next_frame, None) {
            Some(f) => f,
            None => return
        };

        let mut packet = ffmpeg::codec::packet::packet::Packet::empty();

        writeln!(self.log,"{:?}", mv_frame).unwrap();

        if let Format::NULL = format {
            return;
        }

        let frame_in = mv_frame.frame;
        let frame_out = if let Some(ref mut conv) = self.conv {
            let mut frame_out = ffmpeg::frame::Video::new(format.pixel_format(), frame_in.width(),frame_in.height());
            conv.run(&frame_in, &mut frame_out).unwrap();
            frame_out
        } else {
            frame_in
        };

        match self.encoder.encode(&frame_out, &mut packet) {
            Ok(true) => {
                write_packet(self, packet)
            }
            Ok(false) => {
            }
            Err(e) => {
                println!("{:?}", e);
                return;
            }
        }
    }

}


pub(crate) struct PanFinder {
    frame_nr: usize,
    frames: VecDeque<MVFrame>,
    out: Option<ImageOut>,
    output_path: PathBuf,
    stitch: bool,
    format: Format,
    pending_async: Option<thread::JoinHandle<()>>
}

fn write_packet(out: &mut ImageOut, mut packet: ffmpeg::packet::Packet) {
    //let mut packet = self.packets.pop_back().unwrap();
    //let mut out = self.out.as_mut().unwrap();
    packet.set_stream(0);
    //packet.set_pts(Some(mv_frame.idx as i64));
    //packet.set_dts(Some(mv_frame.idx as i64));

    out.octx.write_header().unwrap();
    packet.write(&mut out.octx).unwrap();

}

impl Drop for PanFinder {
    fn drop(&mut self) {
        self.finish_batch();
        self.await();
    }
}

#[derive(Debug)]
enum Run {
    Still,
    Scenechange,
    Run {motion_frames: usize,oldest_frame: usize},
    NeedsScan(u32)
}

impl PanFinder {
    pub fn new(p: PathBuf, stitch: bool, f: Format) -> Self {
        PanFinder {frame_nr: 0, frames: VecDeque::new(), stitch: stitch, out: None, output_path: p, format: f, pending_async: None}
    }

    pub fn add_frame(&mut self, frame: MVFrame) {

        self.frame_nr += 1;
        self.frames.push_front(frame);


        const MAX_QUEUE : usize = 20;

        if self.frames.len() > MAX_QUEUE {
            self.frames.pop_back();
            assert!(self.out.is_none(), "queue overflow only allowed when no batch present");
        }


        let mut oldest_queued_idx = self.frames[self.frames.len()-1].idx as usize;

        if let Some(out) = self.out.as_ref() {
            let last_in_encoder = out.last_frame_idx as usize;
            assert!(oldest_queued_idx - 1 == last_in_encoder, "queue discontinuity. queue: {:?}\nencoder:{}", self.frames, last_in_encoder);
            oldest_queued_idx = last_in_encoder;
        }


        match self.run_length2() {
            Run::Scenechange => {
                self.finish_batch();
            }
            Run::Run{motion_frames, oldest_frame} => {
                if oldest_frame != oldest_queued_idx {
                    self.finish_batch();
                }
            }
            _ => {}
        }

        if self.frames.len() == MAX_QUEUE {
            self.finish_batch();
        }


        self.try_open_batch();

        if let (Run::Run{..}, Some(out)) = (self.run_length(), self.out.as_mut()) {
            while self.frames.len() > 0 {
                out.encode(self.stitch, self.format);
                out.next_frame(self.frames.pop_back().unwrap());
            }
        }


    }


    fn await(&mut self) {
        if let Some(h) = std::mem::replace(&mut self.pending_async, None) {
            h.join().unwrap();
        }
    }

    fn finish_batch(&mut self) {
        let out = ::std::mem::replace(&mut self.out, None);

        if let Some(mut out) = out {

            // last frame if there is one
            out.encode(self.stitch, self.format);

            if self.format != Format::NULL {
                loop {
                    let mut packet = ffmpeg::codec::packet::packet::Packet::empty(); //std::mem::replace(&mut self.cur_packet, None).unwrap();
                    match out.encoder.flush(&mut packet) {
                        Ok(true) => write_packet(&mut out, packet),
                        _ => break
                    }
                }
                out.octx.write_trailer().unwrap();
            }

            if self.frames.len() > 0 {
                for frame in self.frames.iter().rev() {
                    writeln!(out.log, "pan end: {:?}", frame);
                }

            } else {
                writeln!(out.log, "pan end, no more frames");
            }

            if self.stitch {
                writeln!(out.log, "stitch\n{:?}", out.stitcher);

                self.await();
                self.pending_async = Some(out.write_linear_stitch());
            }


        }
    }

    fn run_length2(&mut self) -> Run {
        loop {
            let run = self.run_length();

            match run {
                Run::NeedsScan(i) => {
                    let scan_idx = match self.frames.iter().enumerate().find(|&(_, f)| f.idx == i) {
                        Some(tuple) => tuple.0,
                        None => {
                            panic!("should not happen. expected to find:{} available:{:?}", i, self.frames);
                        }
                    };

                    let (a,b) = unsafe {
                        let frames = &mut self.frames;
                        (&mut *(&mut frames[scan_idx] as *mut MVFrame),
                            if(frames.len() > scan_idx + 1) {
                                Some(&mut *(&mut frames[scan_idx+1] as *mut MVFrame))
                            }  else {
                                None
                            })
                    };

                    let b = match (b, self.out.as_mut().and_then(|out| out.next_frame.as_mut())) {
                        (Some(b),_) => b,
                        (None, Some(b)) => b,
                        _ => return Run::Still
                    };

                    let hint = a.frame.most_common_vectors().or_else(|| b.frame.most_common_vectors());
                    let estimate = search::search(&a.frame, &b.frame, hint);

                    a.add_full_compare(b.idx, estimate);
                    b.add_full_compare(a.idx, estimate.reverse());

                }
                _ => return run

            }
        }
    }

    fn run_length(&self) -> Run {

        if self.frames.len() < 1 {
            return Run::Still;
        }

        /*
         * goals
         * - find some run of linear motion
         * - extend scene around motion forwards and backwards through non-scene change still frames
         * - be robust against I- and low-predicted P-frames
         * - be robust against still frames/motion stutter
         * - handle changes of direction as long as frames during directional change are highly predicted
         *   e.g. if there is a gap in our motion knowledge look for continuity. otherwise look for motion + prediction between frames
         */


        let initial_still = (&self.frames[0].mv_info).forward_still_blocks();
        let mut prev_motion = (&self.frames[0].mv_info).forward_dominant_angle();

        let mut motion_frames = 0;
        let mut last = self.frames[0].idx as usize;

        // we need motion to start a search for similar vectors
        if prev_motion == (0,0) || initial_still >= prev_motion.1 {
            return Run::Still;
        }

        for frame in self.frames.iter().skip(1).chain(self.out.as_ref().and_then(|o| o.next_frame.as_ref()).into_iter()) {

            let frac = frame.predicted_fraction();
            let past = frame.mv_info.future();
            let future = frame.mv_info.past();
            let still = frame.mv_info.forward_still_blocks();
            let motion = frame.mv_info.forward_dominant_angle();

            // scene change
            if past * 8 < future {
                match frame.full_compare(frame.idx + 1) {
                    Some(est) => {
                        if est.error_fraction() > 5.0 {
                            return Run::Scenechange
                        }
                    }
                    None => return Run::NeedsScan(frame.idx + 1)

                }
            }

            // skip highly predicted still frames in the middle of pans
            // compare against next moving one
            // B-frames get bi-predicted and we transplant onto P/I frames
            // so "highly predicted" generally means more than 1.0
            if frac > 1.2 && still >= motion.1 * 2 {
                continue;
            }

            if (motion.0 - prev_motion.0).abs() > 23 {
                match frame.full_compare(frame.idx + 1) {
                    Some(est) => {
                        if est.error_fraction() > 5.0 {
                            break;
                        }
                    }
                    None => return Run::NeedsScan(frame.idx + 1)
                }
            }
            prev_motion = motion;
            last = frame.idx as usize;
            motion_frames += 1;
        }

        if motion_frames == 0 {
            return Run::Still;
        }

        return Run::Run{motion_frames, oldest_frame: last};
    }

    fn output_path(&self) -> PathBuf {
        let mut dir = PathBuf::from(".");
        dir.push(Path::new(self.output_path.file_stem().unwrap()));
        dir.set_extension("seq");
        dir
    }

    fn try_open_batch(&mut self) {
        if self.out.is_some() {
            return;
        }

        let run = match self.run_length() {
            Run::Run{motion_frames, oldest_frame} => (motion_frames, oldest_frame),
            _ => return
        };

        if run.0 < 6 {
            return;
        }

        let start_frame = run.1;

        let dir = self.output_path();
        ::std::fs::create_dir_all(&dir).unwrap();

        // /foo/bar/video.mkv -> video -> ./video.seq/XXXXXX+YYY.png

        let logname = dir.join(format!("{:06}.log", start_frame));
        let mut log = ::std::io::BufWriter::new(::std::fs::OpenOptions::new().create(true).write(true).open(logname).unwrap());

        let to_drop : usize = self.frames[0].idx as usize - start_frame + 1;

        for discard in self.frames.drain(to_drop..).rev() {
            writeln!(log, "pre-pan: {:?}", discard).unwrap();
            if let Some(mvs) = discard.frame.motion_vecs() {
                writeln!(log, "mv: {:?}", mvs);
            }
        }


        let image2format = format!("{:06}+%03d.{}", start_frame, self.format.extension());
        let p = dir.join(image2format);

        let mut octx = unsafe {
            let mut ps     = ::std::ptr::null_mut();
            let     path   = ::std::ffi::CString::new(p.as_os_str().to_str().unwrap()).unwrap();
            let     format = ::std::ffi::CString::new("image2").unwrap();

            match ffmpeg::ffi::avformat_alloc_output_context2(&mut ps, ::std::ptr::null_mut(), format.as_ptr(), path.as_ptr()) {
                0 => {
                    (*ps).flags |= ffmpeg::ffi::AVFMT_NOFILE;
                    Ok(ffmpeg::format::context::Output::wrap(ps))

                }

                e => Err(ffmpeg::Error::from(e))
            }
        }.unwrap();

        // TODO: simplify cargo-culted code
        let codec = ffmpeg::encoder::find_by_name(self.format.codec()).unwrap();
        let mut encoder = {
            let mut output = octx.add_stream(codec).unwrap();
            output.set_time_base((24, 1000));
            output.codec().set_threading(threading::Config{kind: threading::Type::Frame, count: 0, safe: true});
            output.codec().encoder()
        };

        encoder.set_time_base((24, 1000));
        encoder.set_threading(threading::Config{kind: threading::Type::Frame, count: 0, safe: true});


        let mut encoder = encoder.video().unwrap();
        let frame = &self.frames[0].frame;
        encoder.set_width(frame.width());
        encoder.set_height(frame.height());
        encoder.set_format(self.format.pixel_format());
        encoder.set_global_quality(2);
        encoder.set_qmin(1);
        encoder.set_qmax(3);
        let mut encoder = encoder.open_as(codec).unwrap();
        encoder.set_format(self.format.pixel_format());

        let conv = if self.format.pixel_format() != Pixel::YUV420P {
            Some(converter((frame.width(),frame.height()), frame.format(), self.format.pixel_format()).unwrap())
        } else {
            None
        };

        self.out = Some(ImageOut { next_frame: None, log, octx: octx, encoder: encoder, start_frame: start_frame as u32, last_frame_idx: 0, conv: conv, stitcher: LinStitcher::new(), dir: dir })
    }



}

arg_enum!{
    #[derive(Copy, Clone, PartialEq)]
    pub enum Format {
        PNG, JPG, NULL
    }
}

impl Format {
    fn extension(self) -> &'static str {
        match self {
            Format::NULL => "png",
            Format::PNG => "png",
            Format::JPG => "jpg"
        }
    }

    fn codec(self) -> &'static str {
        match self {
            Format::NULL => "png",
            Format::PNG => "png",
            Format::JPG => "mjpeg"
        }
    }

    fn pixel_format(self) -> Pixel {
        match self {
            Format::NULL => Pixel::RGBA,
            Format::PNG => Pixel::RGBA,
            Format::JPG => Pixel::YUVJ420P,
        }
    }
}
