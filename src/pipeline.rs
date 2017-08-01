use std;
use ffmpeg;
use ffmpeg::frame::Video;
use ffmpeg::codec::threading;
use ffmpeg::ffi::AVPictureType;
use ffmpeg::software::{converter,scaler};
use ffmpeg::util::format::pixel::Pixel;
use std::collections::{VecDeque,HashMap};
use std::io::Write;
use ffmpeg::frame;
use std::fmt::*;
use std::path::*;
use motion::vectors::{MVInfo,ToMotionVectors, MVec};
use motion::search::{self, Estimate};
use euclid::rect;
use std::fs::{File,OpenOptions};
use std::io::BufWriter;
use std::cmp::max;




pub(crate) struct MVFrame {
    mv_info: MVInfo,
    frame: frame::Video,
    idx: u32,
    frame_type: AVPictureType,
    motion_estimates: HashMap<u32, Estimate>,
    histogram: [u32; 256],
    sar: ffmpeg::Rational
}

impl MVFrame {
    pub fn new(mv_info: MVInfo, frame: Video, frame_type : AVPictureType, idx: u32, sar: ffmpeg::Rational) -> Self {
        //let idx = frame.display_number();
        MVFrame { mv_info, frame, frame_type, idx, motion_estimates: HashMap::new(), histogram: [0 ; 256], sar }
    }

    fn res(&self) -> u32 {
        self.frame.height() * self.frame.width()
    }

    fn predicted_fraction(&self) -> f32 {
        self.mv_info.pred() as f32 / self.res() as f32
    }

    pub fn full_compare(&self, frame_idx: u32) -> Option<Estimate> {
        self.motion_estimates.get(&frame_idx).cloned()
    }

    pub fn calculate_histogram(&mut self) {
        let pixels = self.frame.data(0);

        let bytes_per_pixel = match self.frame.format() {
            Pixel::YUV420P | Pixel::YUV444P => 1,
            Pixel::YUV420P10LE | Pixel::YUV444P10LE => 2,
            fm @ _ => unimplemented!("pixel format {:?} currently not supported", fm)
        };

        for i in (0 .. pixels.len()).step_by(65 * bytes_per_pixel) {
            self.histogram[pixels[i] as usize] = self.histogram[pixels[i] as usize].saturating_add(1);
        }
    }

    pub fn predecessor_me(&self) -> Option<Estimate> {
        if self.idx > 0 {
            self.motion_estimates.get(&(self.idx - 1)).cloned()
        } else {
            None
        }
    }


    pub fn mode(&self) -> u8 {
        self.histogram[..].into_iter().enumerate().max_by_key(|&(i,v)| v).unwrap().0 as u8
    }

    pub fn min(&self) -> u8 {
        self.histogram[..].into_iter().position(|v| *v > 0).unwrap_or(0) as u8
    }

    pub fn max(&self) -> u8 {
        255 - self.histogram[..].into_iter().rev().position(|v| *v > 0).unwrap_or(0) as u8
    }

    pub fn hist_pop(&self) -> u32 {
        self.histogram[..].into_iter().map(|v| *v as u32).sum()
    }

    pub fn quantile(&self, q: f32) -> u8 {
        let threshold = (self.hist_pop() as f32 * q) as u32;
        let mut sum = 0;
        for i in 0..self.histogram.len() {
            sum += self.histogram[i];
            if sum >= threshold {
                return i as u8;
            }
        }
        return 0;
    }

    pub fn avg(&self) -> u32 {
        self.histogram[..].into_iter().enumerate().map(|(i,v)| i as u32 * *v as u32).sum::<u32>() / max(self.hist_pop(),1)
    }

    fn add_full_compare(&mut self, frame_idx: u32, est: Estimate) {
        if self.motion_estimates.insert(frame_idx, est) == None {
            if est.error_fraction() < 5.0 && frame_idx < self.idx {
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
        write!(f, "\n hist: avg{} mode{} min{} 10th{} 25th{} 50th{} 75th{} 90th{} max{}",
               self.avg(), self.mode(), self.min(),  self.quantile(0.1), self.quantile(0.25), self.quantile(0.5), self.quantile(0.75), self.quantile(0.9), self.max())?;
        if !self.motion_estimates.is_empty() {
            write!(f, "\n  {:?}", self.motion_estimates)?;
        }
        Ok(())
    }
}

pub(crate) struct MVPrefilter {
    unprocessed: Vec<MVFrame>,
    processed: VecDeque<MVFrame>,
    subsample: u8
}

impl MVPrefilter {
    pub fn new(subsampling: u8) -> Self {
        MVPrefilter {unprocessed: vec![], processed: VecDeque::new(), subsample: subsampling}
    }

    pub fn add_frames(&mut self, mut frames: &mut Vec<MVFrame>) {
        use rayon::prelude::*;

        frames.par_iter_mut().for_each(|f| {
            f.calculate_histogram();
        });

        self.unprocessed.extend(frames.drain(..));

        let estimates : Vec<_> =  self.unprocessed.par_windows(2).map(|window| {
            let ref predecessor = window[0];
            let ref current = window[1];
            let hint = current.frame.most_common_vectors();

            (current.idx, predecessor.idx, search::search(&current.frame, &predecessor.frame, hint, self.subsample))
        }).collect();

        for (ci, pi, est) in estimates {
            {
                let current = self.unprocessed.iter_mut().find(|f| f.idx == ci).unwrap();
                current.add_full_compare(pi, est);
            }

            {
                let pred = self.unprocessed.iter_mut().find(|f| f.idx == pi).unwrap();
                pred.add_full_compare(ci, est.reverse());
            }
        }


        let len = self.unprocessed.len()-1;
        for f in self.unprocessed.drain(0..len) {
            self.processed.push_front(f);
        }
    }

    pub fn poll(&mut self) -> Option<MVFrame> {
        self.processed.pop_back()
    }

    pub(crate) fn drain(mut self) -> impl Iterator<Item=MVFrame> {
        self.processed.into_iter().rev()
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
    log: Option<BufWriter<File>>,
}

impl ImageOut {
    fn to_stitcher(mut self) -> LinStitcher {
        let mut st = self.stitcher;
        if let Some(log) = self.log.as_mut() {
            writeln!(log, "stitch\n{:?}", st);
        }

        st.set_start_frame(self.start_frame);
        st
    }

    fn next_frame(&mut self, frame: MVFrame, stitch: bool) {
        assert!(self.next_frame.is_none());
        self.last_frame_idx = frame.idx;

        let est = if frame.idx > 0 {
            frame.full_compare(frame.idx - 1)
        } else {
            None
        };
        if stitch {
            self.stitcher.add_frame(frame.frame.clone(), est, frame.sar);
        }
        self.next_frame = Some(frame);
    }

    fn encode(&mut self, format: Format) {
        let mv_frame = match ::std::mem::replace(&mut self.next_frame, None) {
            Some(f) => f,
            None => return
        };

        let mut packet = ffmpeg::codec::packet::packet::Packet::empty();

        if let Some(log) = self.log.as_mut() {
            writeln!(log,"{:?}", mv_frame).unwrap();
        }


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
    log: Option<BufWriter<File>>,
    output_path: PathBuf,
    pub image_batches: Vec<LinStitcher>,
    config: ::Config
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

#[derive(Debug)]
enum Run {
    Still,
    SceneChange,
    Run {motion_frames: usize,oldest_frame: usize, end: RunEnd},
}

#[derive(Debug)]
enum RunEnd {
    DirectionChange,
    OutOfFrames,
    MismatchAfterStills,
    SceneChange,
    LowEntropyFrame
}

#[derive(Debug)]
enum PanEnd {
    Scenechange,
    QueueSaturation,
    RunDiscontinuity(Run),
    EndOfStream
}

impl PanFinder {
    pub fn new(output_path: PathBuf, config: ::Config) -> Self {
        ::std::fs::create_dir_all(&output_path).unwrap();

        let log_path = output_path.join("frames.log");
        let log = if config.log {
            Some(BufWriter::new(OpenOptions::new().write(true).create(true).open(log_path).unwrap()))
        } else {
            None
        };

        PanFinder {frame_nr: 0, frames: VecDeque::new(), out: None, output_path, image_batches: vec![], log, config}
    }

    pub fn add_frame(&mut self, frame: MVFrame) {

        if frame.idx > 0 {
            if let (Some(est), Some(log)) = (frame.full_compare(frame.idx - 1), self.log.as_mut()) {
                writeln!(log, "{:?}", frame);
            }
        }

        self.frame_nr += 1;
        self.frames.push_front(frame);


        const MAX_QUEUE : usize = 24;

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


        let run = self.run_length();
        match run {
            Run::SceneChange => {
                self.finish_batch(PanEnd::Scenechange);
            }
            Run::Run{oldest_frame, ..} => {
                if oldest_frame != oldest_queued_idx {
                    self.finish_batch(PanEnd::RunDiscontinuity(run));
                }
            }
            _ => {}
        }

        if self.frames.len() == MAX_QUEUE {
            self.finish_batch(PanEnd::QueueSaturation);
        }


        if let run @ Run::Run{..} = self.run_length() {
            self.try_open_batch(run);

            if let Some(ref mut out) = self.out {
                while self.frames.len() > 0 {
                    out.encode(self.config.single_frame_format);
                    out.next_frame(self.frames.pop_back().unwrap(),self.config.stitch);
                }
            }
        }
    }


    pub fn close(mut self) -> Vec<LinStitcher> {
        self.finish_batch(PanEnd::EndOfStream);
        self.image_batches
    }

    fn finish_batch(&mut self, reason: PanEnd) {
        let out = ::std::mem::replace(&mut self.out, None);

        if let Some(mut out) = out {

            let format = self.config.single_frame_format;
            // last frame if there is one
            out.encode(format);

            if format  != Format::NULL {
                loop {
                    let mut packet = ffmpeg::codec::packet::packet::Packet::empty(); //std::mem::replace(&mut self.cur_packet, None).unwrap();
                    match out.encoder.flush(&mut packet) {
                        Ok(true) => write_packet(&mut out, packet),
                        _ => break
                    }
                }
                out.octx.write_trailer().unwrap();
            }

            if let Some(log) = out.log.as_mut() {
                writeln!(log, "{:?}", reason);

                if self.frames.len() > 0 {
                    for frame in self.frames.iter().rev() {
                        writeln!(log, "pan end: {:?}", frame);
                    }
                } else {
                    writeln!(log, "pan end, no more frames");
                }
            }

            assert!(out.last_frame_idx >= out.start_frame, "created an out without frame {} {}", out.last_frame_idx, out.start_frame);

            self.image_batches.push(out.to_stitcher());
        }
    }

    fn compare_frames(newer: &mut MVFrame, older: &mut MVFrame) -> Estimate {
        if let Some(est) =  newer.full_compare(older.idx) {
            return est;
        }

        let hint = newer.frame.most_common_vectors().or_else(|| newer.frame.most_common_vectors());
        let estimate = search::search(&newer.frame, &older.frame, hint, 0);

        newer.add_full_compare(older.idx, estimate);
        older.add_full_compare(newer.idx, estimate.reverse());
        estimate
    }

    fn run_length(&mut self) -> Run {

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


        let mut motion_frames = 0;
        let mut last = self.frames[0].idx as usize;

        let mut frame_refs : Vec<&mut MVFrame> = vec![];
        let (a,b) = self.frames.as_mut_slices();
        frame_refs.extend(a);
        frame_refs.extend(b);
        if let Some(out) = self.out.as_mut() {
            if let Some(last) = out.next_frame.as_mut() {
                frame_refs.push(last);
            }
        }

        if frame_refs.len() < 2 {
            return Run::Still;
        }

        let mut end_reason = RunEnd::OutOfFrames;
        let mut mvec = MVec::new();

        {
            let ref current = frame_refs[0];
            let ref pred = frame_refs[1];

            match current.full_compare(current.idx - 1) {
                Some(est) => {
                    if let Some(pred_est) = pred.predecessor_me() {
                        if (pred_est.error_fraction() - est.error_fraction()).abs() > 6.5 {
                            return Run::SceneChange;
                        }
                    }
                    mvec = mvec.from_vector(est.x, est.y);
                    if est.quantile(0.75) >= 10 {
                        if let Some(pred_est) = pred.predecessor_me() {
                            let vec = MVec::new().from_vector(pred_est.x, pred_est.y);
                            if !vec.is_similar(&mvec) {
                                return Run::SceneChange;
                            }
                        } else {
                            return Run::SceneChange;
                        }
                    }
                    if est.x == 0 && est.y == 0 {
                        return Run::Still;
                    }
                }
                None => return Run::Still
            }
        }

        for frame_idx in 1..frame_refs.len() /* self.frames.iter().skip(1).chain(self.out.as_ref().and_then(|o| o.next_frame.as_ref()).into_iter()) */ {
            let (successors,predecessors) = frame_refs.split_at_mut(frame_idx);
            let (current,predecessors) = predecessors.split_at_mut(1);
            let ref mut current = current[0];

            if current.quantile(0.9) - current.quantile(0.1) <= 35 {
                end_reason = RunEnd::LowEntropyFrame;
                break;
            }

            let successor_estimate = {
                let newer_idx = successors.len()-1;
                let ref mut newer = successors[newer_idx];
                PanFinder::compare_frames(newer, current)
            };

            let vec = MVec::new().from_vector(successor_estimate.x, successor_estimate.y);
            if successor_estimate.quantile(0.75) >= 10 && !vec.is_similar(&mvec) {
                end_reason = RunEnd::SceneChange;
                break
            }

            let current_idx = current.idx as usize;

            last = current_idx;
            if successor_estimate.x != 0 || successor_estimate.y != 0 {
                mvec = vec;
                motion_frames += 1;
            }

            // probably redundant given the same check above the loop
            if let Some(predecessor_estimate) = current.predecessor_me() {
                if (predecessor_estimate.error_fraction() - successor_estimate.error_fraction()).abs() > 6.5 {
                    end_reason = RunEnd::SceneChange;
                    break;
                }
            }
        }


        if motion_frames == 0 {
            return Run::Still;
        }

        return Run::Run{motion_frames, oldest_frame: last, end: end_reason};



        /*
        let initial_still = (&self.frames[0].mv_info).still_blocks(::motion::vectors::Direction::Forward);
        let mut prev_motion = (&self.frames[0].mv_info).dominant_angle(::motion::vectors::Direction::Forward);

        let mut motion_frames = 0;
        let mut last = self.frames[0].idx as usize;

        // we need motion to start a search for similar vectors
        if prev_motion == (0,0) || initial_still >= prev_motion.1 {
            return Run::Still;
        }

        let mut frame_refs : Vec<&mut MVFrame> = vec![];
        let (a,b) = self.frames.as_mut_slices();
        frame_refs.extend(a);
        frame_refs.extend(b);
        if let Some(out) = self.out.as_mut() {
            if let Some(last) = out.next_frame.as_mut() {
                frame_refs.push(last);
            }
        }

        let mut end_reason = RunEnd::OutOfFrames;

        for frame_idx in 1..frame_refs.len() /* self.frames.iter().skip(1).chain(self.out.as_ref().and_then(|o| o.next_frame.as_ref()).into_iter()) */ {
            let (newer,older) = frame_refs.split_at_mut(frame_idx);
            let (frame,older) = older.split_at_mut(1);
            let ref mut frame = frame[0];

            let frac = frame.predicted_fraction();
            let past = frame.mv_info.past();
            let future = frame.mv_info.future();
            let still = frame.mv_info.still_blocks(::motion::vectors::Direction::Backward);
            let motion = frame.mv_info.dominant_angle(::motion::vectors::Direction::Backward);


            // skip highly predicted still frames in the middle of pans
            // compare against next moving one
            // B-frames get bi-predicted and we transplant onto P/I frames
            // so "highly predicted" generally means more than 1.0
            if frac > 1.2 && still >= motion.1 * 2 {
                continue;
            }

            // potential scene change
            if past * 8 < future {
                let idx = newer.len()-1;
                let ref mut newer = newer[idx];
                let est = PanFinder::compare_frames(newer, frame);
                let vec = MVec::new().from_vector(est.x, est.y);
                if est.error_fraction() > 5.0 && (vec.angle() - prev_motion.0).abs() > 23 {
                    return Run::Scenechange
                }
            }


            if (motion.0 - prev_motion.0).abs() > 23 {
                let idx = newer.len()-1;
                let ref mut newer = newer[idx];
                let est = PanFinder::compare_frames(newer, frame);
                let vec = MVec::new().from_vector(est.x, est.y);
                if est.error_fraction() > 5.0 && (vec.angle() - prev_motion.0).abs() > 23 {
                    end_reason = RunEnd::DirectionChange;
                    break;
                }
            }

            let current = frame.idx as usize;

            // skipped frames, verify that it's still the same scene
            if current + 1 != last {

                let last_idx = match newer.iter().position(|f| f.idx == last as u32) {
                    Some(i) => i,
                    None => {
                        panic!("expected to find {} avail:{:?}; cur:{}", last, newer.iter().map(|n| n.idx).collect::<Vec<_>>(), current)
                    }
                };
                let ref mut newer = newer[last_idx];
                let est = PanFinder::compare_frames(newer, frame);
                if est.error_fraction() > 5.0 {
                    end_reason = RunEnd::MismatchAfterStills;
                    break
                }
            }

            prev_motion = motion;
            last = current;
            motion_frames += 1;
        }

        if motion_frames == 0 {
            return Run::Still;
        }

        return Run::Run{motion_frames, oldest_frame: last, end: end_reason};
        */
    }

    pub fn output_path_from_input(p: &Path) -> PathBuf {
        let mut dir = PathBuf::from(".");
        dir.push(Path::new(p.file_stem().unwrap()));
        dir.set_extension("seq");
        dir
    }

    fn try_open_batch(&mut self, run: Run) {
        if self.out.is_some() {
            return;
        }

        let run_info = match run {
            Run::Run{motion_frames, oldest_frame, ..} => (motion_frames, oldest_frame),
            _ => return
        };

        if run_info.0 < 6 {
            return;
        }

        let start_frame = run_info.1;

        let dir = self.output_path.to_owned();
        ::std::fs::create_dir_all(&dir).unwrap();

        // /foo/bar/video.mkv -> video -> ./video.seq/XXXXXX+YYY.png

        let logname = dir.join(format!("{:06}.log", start_frame));
        let mut log = if self.config.log {
            Some(::std::io::BufWriter::new(::std::fs::OpenOptions::new().create(true).write(true).truncate(true).open(logname).unwrap()))
        } else {
            None
        };

        let to_drop : usize = self.frames[0].idx as usize - start_frame + 1;


        for discard in self.frames.drain(to_drop..).rev() {
            if let Some(log) = log.as_mut() { writeln!(log, "pre-pan: {:?}", discard).unwrap();}
            /*
            if let Some(mvs) = discard.frame.motion_vecs() {
                writeln!(log, "mv: {:?}", mvs);
            }*/
        }

        if let Some(log) = log.as_mut() {writeln!(log, "{:?}", run);}

        let format = self.config.single_frame_format;

        let image2format = format!("{:06}+%03d.{}", start_frame, format.extension());
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
        let codec = ffmpeg::encoder::find_by_name(format.codec()).unwrap();
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
        encoder.set_format(format.pixel_format());
        encoder.set_global_quality(2);
        encoder.set_qmin(1);
        encoder.set_qmax(3);
        let mut encoder = encoder.open_as(codec).unwrap();
        encoder.set_format(format.pixel_format());

        let conv = if format.pixel_format() != Pixel::YUV420P {
            Some(converter((frame.width(),frame.height()), frame.format(), format.pixel_format()).unwrap())
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
