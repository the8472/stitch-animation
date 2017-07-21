#![feature(repr_align)]
#![feature(attr_literals)]
#![feature(conservative_impl_trait)]
#![feature(iterator_step_by)]
#![feature(const_fn)]
#![feature(cfg_target_feature)]

extern crate ffmpeg;
#[macro_use]
extern crate clap;
extern crate simd;
extern crate rayon;
extern crate euclid;
extern crate itertools;
extern crate float_ord;
extern crate atomic;
extern crate stdsimd;

mod stitchers;
mod motion;
mod pipeline;

use ffmpeg::codec::threading;
use std::path::*;
use clap::{Arg, App};
use std::io::BufRead;
use motion::vectors::MVInfo;
use pipeline::{PanFinder, Format, MVPrefilter, MVFrame};




fn process_video(input: &Path, stitch: bool, format: Format, start: u32, max_frames: u32) {
    match ffmpeg::format::input(&input) {
        Ok(mut ctx) => {
            let mut vdecoder;
            let vid_idx;

            /*

            if start > 0 {
                unsafe {
                    use ffmpeg::ffi;
                    let ctx = ctx.as_mut_ptr();
                    let res = ffi::avformat_seek_file(ctx, -1, 0, start as i64, start as i64, ffi::AVSEEK_FLAG_FRAME);
                    if res != 0 {
                        panic!("seek failed, error code:{}", res);
                    }
                }

            }
            */

            {
                let vstream = ctx.streams().best(ffmpeg::media::Type::Video).expect(
                    "video stream",
                );
                vid_idx = vstream.index();
                let mut decoder = vstream.codec().decoder();
                unsafe {
                    let ctx = decoder.as_mut_ptr();
                    (*ctx).flags2 |= ffmpeg::ffi::AV_CODEC_FLAG2_EXPORT_MVS as i32;
                    (*ctx).refcounted_frames = 1;
                }
                decoder.set_threading(threading::Config {
                    kind: threading::Type::Frame,
                    count: 0,
                    safe: true,
                });

                vdecoder = decoder.video().unwrap();
            };

            use std::thread;
            use std::sync::mpsc::sync_channel;

            let (to_prefilter, prefilter_in) = sync_channel(25);
            let (to_writer, writer_in) = sync_channel(25);

            thread::spawn(move || {
                let mut filter = MVPrefilter::new();

                while let Ok(Some(mv_frame)) = prefilter_in.recv() {
                    filter.add_frame(mv_frame);
                    if let Some(frame) = filter.poll() {
                        to_writer.send(Some(frame)).unwrap();
                    }
                }

                for remainder in filter.drain() {
                    to_writer.send(Some(remainder)).unwrap();
                }
                to_writer.send(None).unwrap();
            });



            let p = input.to_owned();

            let thread = thread::spawn(move || {
                let mut writer = PanFinder::new(p, stitch, format);

                while let Ok(Some(mv_frame)) = writer_in.recv() {
                    writer.add_frame(mv_frame);
                }
            });

            let mut frame_counter = 0;

            for (stream, packet) in ctx.packets() {
                if stream.index() != vid_idx {
                    continue;
                }

                // TODO: find previous keyframe, seek back, skip forwards while decoding
                if frame_counter < start {
                    frame_counter += 1;
                    continue;
                }


                let mut frame = ffmpeg::frame::Video::new(
                    vdecoder.format(),
                    vdecoder.width(),
                    vdecoder.height(),
                );

                match vdecoder.decode(&packet, &mut frame) {
                    Ok(true) => {
                        let frame_type;
                        //let frame_idx = frame.display_number();

                        unsafe {
                            let ptr = frame.as_ptr();
                            frame_type = (*ptr).pict_type;
                        }

                        let mv_frame =
                            MVFrame::new(MVInfo::new(), frame, frame_type, frame_counter);
                        to_prefilter.send(Some(mv_frame)).unwrap();

                        if frame_counter > start.saturating_add(max_frames) {
                            break;
                        }
                    }
                    _ => {}
                }

                frame_counter += 1;
            }

            to_prefilter.send(None).unwrap();

            thread.join().unwrap();
        }
        Err(e) => {
            eprintln!("{}", e);
        }
    }
}

fn main() {
    let matches = App::new("animation linear panning detection, scene extraction and stitching for videos")
        .version(crate_version!())
        .arg(Arg::with_name("nostitch").long("nostitch").required(false).takes_value(false)
            .help("do not create composite images"))
        .arg(Arg::with_name("pics").short("p").long("pictures").required(false).takes_value(true)
            .possible_values(&["png","jpg"]).help("save individual frames"))
        .arg(Arg::with_name("inputs").index(1).multiple(true).required(true)
            .help("videos files to process. specify '-' to read a newline-separated list from stdin.\nExample: find /media/videos -type f -name '*.mkv' | stitch-animation -"))
        .arg(Arg::with_name("seek_to").short("s").takes_value(true).help("seek to frame number [currently inaccurate, specify a lower number than the desired actual frame]"))
        .arg(Arg::with_name("N").short("n").takes_value(true).help("process at most N frames, after seeking"))
        .get_matches();


    ffmpeg::init().unwrap();

    let stitch = !matches.is_present("nostitch");

    if !cfg!(debug_assertions) {
        unsafe {
            ffmpeg::ffi::av_log_set_level(ffmpeg::ffi::AV_LOG_ERROR);
        }
    }

    let format = value_t!(matches, "pics", Format).unwrap_or(Format::NULL);
    let seek = value_t!(matches, "seek_to", u32).unwrap_or(0);
    let max_frames = value_t!(matches, "N", u32).unwrap_or(std::u32::MAX);

    for p in matches.values_of_os("inputs").unwrap().map(Path::new) {
        if p == Path::new("-") {
            let stdin = std::io::stdin();
            let reader = stdin.lock();
            for line in reader.lines() {
                process_video(Path::new(&line.unwrap()), stitch, format, seek, max_frames);
            }
            continue;
        }
        process_video(p, stitch, format, seek, max_frames);
    }

    let counts = motion::search::COUNTS.load(atomic::Ordering::Relaxed);
    println!("loops:{} points:{}", counts.0, counts.1);
}
