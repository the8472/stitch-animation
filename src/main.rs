#![feature(repr_align)]
#![feature(attr_literals)]

extern crate ffmpeg;

use std::env;
use std::slice;
use ffmpeg::frame::side_data::Type;
use ffmpeg::ffi::AVMotionVector;
use ffmpeg::codec::threading;
use std::collections::HashMap;
use std::io::Write;
use ffmpeg::frame;
use std::fmt::*;
use std::path::*;

fn mvs(input: &[u8]) -> &[ffmpeg::ffi::AVMotionVector] {
    let ptr = input.as_ptr() as *const ffmpeg::ffi::AVMotionVector ;
    let len = input.len() / std::mem::size_of::<ffmpeg::ffi::AVMotionVector>();
    unsafe { slice::from_raw_parts(ptr, len) }
}


struct MVec {
    angle: i16,
    len: i16,
    forward: usize,
    backward: usize,
    intra: usize
}

impl MVec {
    fn cnt(&self) -> usize {
        self.forward + self.backward + self.intra
    }
}

impl Debug for MVec {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "({} {} | ->{} {}<- ~{})", self.angle, self.len, self.forward, self.backward, self.intra)
    }
}

struct MVInfo {
    swarms: Vec<MVec>,
}

impl MVInfo {
    fn past(&self) -> usize {
        self.swarms.iter().map(|e| e.forward).sum::<usize>()
    }

    fn future(&self) -> usize {
        self.swarms.iter().map(|e| e.backward).sum::<usize>()
    }

    fn intra(&self) -> usize {
        self.swarms.iter().map(|e| e.intra).sum::<usize>()
    }

    fn potential_still_frame(&self) -> bool {
        self.swarms[0].len == 0
    }

    fn still_blocks(&self) -> usize {
        self.swarms.iter().map(|e| e.len as usize).sum::<usize>()
    }

    fn dominant_angle(&self) -> (i16, usize) {
        let mut bins = HashMap::new();

        for mv in &self.swarms {
            if mv.len == 0 {
                continue;
            }
            let entry = bins.entry(mv.angle).or_insert(0);
            (*entry) += mv.cnt();
        }

        match bins.iter().max_by_key(|&(k,v)| *v) {
            Some((angle, count)) => (*angle, *count),
            None => (0,0)
        }
    }


}

impl Debug for MVInfo {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "->{} {}<- ~{} | {:?}", self.past(), self.future(), self.intra(), self.swarms)
    }
}

fn detect(vec: &[AVMotionVector]) -> MVInfo {
    let mut bins = HashMap::new();

    for v in vec {
        let y = (v.src_y - v.dst_y) as f32;
        let x = (v.src_x - v.dst_x) as f32;


        /*
        // intra-frame prediction is not motion
        if v.source == 0 {
            continue;
        }*/

        let mut angle = (y.atan2(x).to_degrees() / 22.5).round() * 22.5;
        angle += 180.0; // norm to 0-360
        if v.source > 0 { // invert if motion from future
            angle += 180.0;
        }
        angle %= 360.0;

        let len = x.hypot(y);

        /*
        if len.abs() < 0.01 {
            angle = 0.0;
        }*/


        let len = len.log2().round().exp2();
        let bin = bins.entry((angle as i16, len as i16)).or_insert((0,0,0));

        match v.source.signum() {
            -1 => (*bin).0 += 1,
            0 => (*bin).2 += 1,
            1 => (*bin).1 += 1,
            _ => unreachable!()
        }

        /*
        if len != 0.0 {
            print!(" | {} {}", len, angle);
        }*/
    }

    let mut sorted = bins.iter().map(|(k, v)| MVec{angle: k.0, len: k.1, forward: (*v).0, backward: (*v).1, intra: (*v).2}).collect::<Vec<_>>();
    sorted.sort_by_key(|e| e.cnt());
    sorted.reverse();


    MVInfo{swarms: sorted}
}

struct MVFrame {
    mv_info: MVInfo,
    frame: frame::Video,
    idx: usize,
    frame_type: ffmpeg::ffi::AVPictureType
}

impl Debug for MVFrame {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "{} {:?} | {:?}", self.idx, self.frame_type, self.mv_info)
    }
}

struct PngOut {
    octx: ffmpeg::format::context::Output,
    encoder: ffmpeg::codec::encoder::video::Encoder
}

use std::collections::VecDeque;

struct Writer {
    frame_nr: usize,
    frames: std::collections::VecDeque<MVFrame>,
    out: Option<PngOut>,
    output_path: PathBuf,
    log: Option<std::io::BufWriter<std::fs::File>>
}

fn write_packet(out: &mut PngOut, mut packet: ffmpeg::packet::Packet) {
    //let mut packet = self.packets.pop_back().unwrap();
    //let mut out = self.out.as_mut().unwrap();
    packet.set_stream(0);
    //packet.set_pts(Some(mv_frame.idx as i64));
    //packet.set_dts(Some(mv_frame.idx as i64));

    out.octx.write_header().unwrap();
    packet.write(&mut out.octx).unwrap();

}

impl Writer {
    fn new(p: PathBuf) -> Self {
        Writer{frame_nr: 0, frames: std::collections::VecDeque::new(), out: None, output_path: p, log: None}
    }

    fn add_frame(&mut self, mut frame: MVFrame) {
        frame.idx = self.frame_nr;
        self.frames.push_front(frame);
        self.frame_nr += 1;
        if self.frames.len() > 20 {
            self.frames.pop_back();
        }

        if self.out.is_some() && self.pan_end() {
            while self.frames.len() > 1 {
                self.encode();
            }
            self.close()
        }

        self.open_batch();

        if self.out.is_some() {
            while self.frames.len() > 1 {
                self.encode();
            }
        }

    }

    fn pan_end(&self) -> bool {
        if self.frames.len() < 2 {
            return false;
        }

        let ref cur = self.frames[0];
        let ref prev = self.frames[1];

        use ffmpeg::ffi::AVPictureType::*;

        // if previous frame used a good chunk of backwards predictions from p-frame then the current frame is likely a continuation of a pan
        if cur.frame_type == AV_PICTURE_TYPE_P && prev.frame_type == AV_PICTURE_TYPE_B && prev.mv_info.swarms[0].backward * 3 > prev.mv_info.swarms[0].forward  {
            return false
        }

        let cur_angle = cur.mv_info.dominant_angle();
        let prev_angle = cur.mv_info.dominant_angle();

        // end pans if still blocks dominate
        // TODO: figure out an efficient way to continue through still frames
        if cur_angle.1 * 2 <= cur.mv_info.still_blocks()  {
            return true;
        }

        return (cur_angle.0 - prev_angle.0).abs() > 23;
    }

    fn close(&mut self) {
        if let Some(ref mut out) = self.out {

            //if self.cur_packet.is_some() {
                loop {
                    let mut packet = ffmpeg::codec::packet::packet::Packet::empty(); //std::mem::replace(&mut self.cur_packet, None).unwrap();
                    match out.encoder.flush(&mut packet) {
                        Ok(true) => write_packet(out, packet),
                        _ => break
                    }
                }
                out.octx.write_trailer().unwrap();
            //}

            assert!(self.frames.len() >= 1);
            writeln!(self.log.as_mut().unwrap(), "pan end: {:?}", self.frames[0]).unwrap();
        }
        self.out = None;
        self.log = None;

    }

    fn run_length(&self) -> usize {
        if self.frames.len() < 1 {
            return 0;
        }
        let mut prev_dir = &self.frames[0].mv_info.swarms[0];
        let mut last = 0;

        for fr_idx in 1..self.frames.len() {
            let frame = &self.frames[fr_idx];
            let cur_dir = &frame.mv_info.swarms[0];
            if cur_dir.len == 0 {
                last = fr_idx;
                break;
            }
            if (cur_dir.angle - prev_dir.angle).abs() > 23 {
                break;
            }
            prev_dir = cur_dir;
            last = fr_idx;
        }

        return last + 1;
    }

    fn strip(&mut self) {
        let len = self.run_length();
        self.frames.truncate(len);
    }

    fn encode(&mut self) {
        if let Some(ref mut out) = self.out {

            let mut packet = ffmpeg::codec::packet::packet::Packet::empty();


            let mv_frame = self.frames.pop_back().unwrap();

            writeln!(self.log.as_mut().unwrap(),"{:?}", mv_frame).unwrap();

            let frame_in = mv_frame.frame;
            let mut frame_out = ffmpeg::frame::Video::new(ffmpeg::util::format::pixel::Pixel::RGBA, frame_in.width(),frame_in.height());
            let mut conv = ffmpeg::software::converter((frame_in.width(),frame_in.height()), frame_in.format(), ffmpeg::util::format::pixel::Pixel::RGBA).unwrap();
            conv.run(&frame_in, &mut frame_out).unwrap();

            match out.encoder.encode(&frame_out, &mut packet) {
                Ok(true) => {
                    write_packet(out, packet)
                }
                Ok(false) => {
                    //self.cur_packet = Some(packet)
                }
                Err(e) => {
                    println!("{:?}", e);
                    return;
                }
            }
        }
    }



    fn open_batch(&mut self) {
        if self.out.is_some() {
            return;
        }
        if self.run_length() < 6 {
            return;
        }
        self.strip();


        let start_frame = self.frames[self.frames.len()-1].idx;



        let stem = self.output_path.file_stem().unwrap();
        let mut dir = PathBuf::from(".");
        dir.push(stem);
        dir.set_extension("seq");
        std::fs::create_dir_all(&dir).unwrap();

        // /foo/bar/video.mkv -> video -> ./video.seq/outXXXX+001.png

        let logname = dir.join(format!("out{:06}.log", start_frame));
        self.log = Some(std::io::BufWriter::new(std::fs::OpenOptions::new().create(true).write(true).open(logname).unwrap()));

        let image2format = format!("out{:06}+%03d.png", start_frame);
        let p = dir.join(image2format);

        let mut octx = unsafe {
            let mut ps     = std::ptr::null_mut();
            let     path   = std::ffi::CString::new(p.as_os_str().to_str().unwrap()).unwrap();
            let     format = std::ffi::CString::new("image2").unwrap();

            match ffmpeg::ffi::avformat_alloc_output_context2(&mut ps, std::ptr::null_mut(), format.as_ptr(), path.as_ptr()) {
                0 => {
                    (*ps).flags |= ffmpeg::ffi::AVFMT_NOFILE;
                    Ok(ffmpeg::format::context::Output::wrap(ps))

                }

                e => Err(ffmpeg::Error::from(e))
            }
        }.unwrap();

        // TODO: simplify cargo-culted code
        let codec = ffmpeg::encoder::find_by_name("png").unwrap();
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
        encoder.set_format(ffmpeg::util::format::pixel::Pixel::RGBA);
        let mut encoder = encoder.open_as(codec).unwrap();
        encoder.set_format(ffmpeg::util::format::pixel::Pixel::RGBA);

        self.out = Some(PngOut { octx: octx, encoder: encoder })
    }



}


fn main() {
    ffmpeg::init().unwrap();

    /*
    unsafe {
        ffmpeg::ffi::av_log_set_level(ffmpeg::ffi::AV_LOG_TRACE);
    }*/

    let arg = env::args().nth(1).expect("missing file");
    let input = std::path::Path::new(&arg);


    match ffmpeg::format::input(&input) {
        Ok(mut ctx) => {

            let mut vdecoder;
            let vid_idx;
            let timebase;

            {
                let vstream = ctx.streams().best(ffmpeg::media::Type::Video).expect("video stream");
                vid_idx = vstream.index();
                timebase = vstream.time_base();
                let mut decoder = vstream.codec().decoder();
                unsafe {
                    let ctx = decoder.as_mut_ptr();
                    (*ctx).flags2 |= ffmpeg::ffi::AV_CODEC_FLAG2_EXPORT_MVS as i32;
                    (*ctx).refcounted_frames = 1;
                }
                decoder.set_threading(threading::Config{kind: threading::Type::Frame, count: 0, safe: true});

                vdecoder = decoder.video().unwrap();
            };

            use std::thread;
            use std::sync::mpsc::sync_channel;

            let (tx, rx) = sync_channel(40);

            let p = input.to_owned();

            let thread = thread::spawn(move|| {
                let mut writer = Writer::new(p);

                while let Ok(Some(mv_frame)) = rx.recv() {
                    writer.add_frame(mv_frame);
                }
            });


            for (stream, packet) in ctx.packets() {
                if stream.index() != vid_idx {
                    continue;
                }

                let mut frame = ffmpeg::frame::Video::new(vdecoder.format(), vdecoder.width(), vdecoder.height());

                match vdecoder.decode(&packet, &mut frame) {
                    Ok(true) => {
                        let frame_type;

                        unsafe {
                            let ptr = frame.as_ptr();
                            frame_type = (*ptr).pict_type;
                        }

                        let mut dir = MVInfo{swarms: vec![MVec{angle: 0, len: 0, forward: 0, backward: 0, intra: 0}]};

                        if let Some(raw_vecs) =  frame.side_data(Type::MotionVectors) {
                            let vecs = mvs(raw_vecs.data());
                            dir = detect(vecs);
                            //write!(out, "->{} {}<- | {:?}", past, future, sorted);
                        }

                        let mv_frame = MVFrame{ mv_info: dir, frame: frame, frame_type: frame_type, idx: 0};

                        //writer.add_frame(MVFrame{ mv_info: dir, frame: frame, frame_type: frame_type, idx: 0});
                        tx.send(Some(mv_frame)).unwrap();

                    },
                    _ => {
                    }

                }
            }

            tx.send(None).unwrap();

            thread.join().unwrap();

        }
        Err(e) => {
            eprintln!("{}", e);
        }
    }
}
