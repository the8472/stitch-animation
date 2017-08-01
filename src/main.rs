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
extern crate oxipng;
extern crate std_semaphore;

mod stitchers;
mod motion;
mod pipeline;

use ffmpeg::codec::threading;
use std::path::*;
use clap::{Arg, App};
use std::io::BufRead;
use motion::vectors::MVInfo;
use pipeline::{PanFinder, Format, MVPrefilter, MVFrame};
use rayon::prelude::*;




fn process_video(input: &Path, config: Config) {
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
            let (to_pan_finder, finder_rx) = sync_channel(25);
            let (to_image_writer, writer_rx) = sync_channel(3);

            thread::spawn(move || {
                let mut filter = MVPrefilter::new(config.subsample);

                let mut batch = vec![];

                loop {
                    batch.push(match prefilter_in.recv() {
                        Ok(mv_frame) => mv_frame,
                        _ => break
                    });

                    batch.extend(prefilter_in.try_iter().take(rayon::current_num_threads()-1));

                    //filter.add_frame(mv_frame);
                    filter.add_frames(&mut batch);


                    while let Some(frame) = filter.poll() {
                        to_pan_finder.send(frame).unwrap();
                    }
                }

                for remainder in filter.drain() {
                    to_pan_finder.send(remainder).unwrap();
                }
            });


            let p = PanFinder::output_path_from_input(&input);
            let p2 = p.clone();

            thread::spawn(move || {
                let mut writer = PanFinder::new(p, config);

                while let Ok(mv_frame) = finder_rx.recv() {
                    writer.add_frame(mv_frame);
                    for out in writer.image_batches.drain(..).filter(|out| {
                        out.expansion_ratio() < config.min_expand
                    }) {
                        to_image_writer.send(out).unwrap();
                    }
                }

                for out in writer.close().into_iter().filter(|out| {
                    out.expansion_ratio() < config.min_expand
                }) {
                    to_image_writer.send(out).unwrap();
                }
            });

            let thread = thread::spawn(move || {

                loop {
                    let mut batch = vec![];
                    batch.push(match writer_rx.recv() {
                        Ok(stitcher) => stitcher,
                        _ => break
                    });

                    batch.extend(writer_rx.try_iter());

                    batch.into_par_iter().for_each(|stitcher| {
                        stitcher.write_linear_stitch(config.optimize, &p2);
                    });
                }
            });

            let mut frame_counter = 0;


            let ctxptr = unsafe {
                ctx.as_mut_ptr()
            };

            for (stream, packet) in ctx.packets() {
                if stream.index() != vid_idx {
                    continue;
                }

                // TODO: find previous keyframe, seek back, skip forwards while decoding
                if frame_counter < config.seek {
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

                        //let frame_idx = frame.display_number();

                        let (sar,frame_type) = unsafe {
                            let frame_ptr = frame.as_mut_ptr();
                            use ffmpeg::ffi::*;

                            (av_guess_sample_aspect_ratio(ctxptr,
                                                          stream.as_ptr() as *mut AVStream,
                                                          frame_ptr),
                             (*frame_ptr).pict_type
                            )
                        };


                        let mv_frame = MVFrame::new(MVInfo::new(), frame, frame_type, frame_counter, sar.into());
                        to_prefilter.send(mv_frame).unwrap();

                        if frame_counter > config.seek.saturating_add(config.max_frames) {
                            break;
                        }

                        frame_counter += 1;
                    }
                    Ok(false) => {
                        // TODO: handle decoder flushing at end of stream
                    }
                    Err(e) => {
                        eprintln!("{:?}", e);
                    }
                }
            }

            drop(to_prefilter);

            thread.join().unwrap();
        }
        Err(e) => {
            eprintln!("{}", e);
        }
    }
}

#[derive(Copy,Clone)]
struct Config {
    optimize: bool,
    single_frame_format: Format,
    seek: u32,
    max_frames :u32,
    subsample: u8,
    min_expand: f32,
    log: bool,
    stitch: bool
}

fn main() {
    let matches = App::new("animation linear panning detection, scene extraction and stitching for videos")
        .version(crate_version!())
        .arg(Arg::with_name("nostitch").long("nostitch").required(false).takes_value(false)
            .help("do not create composite images"))
        .arg(Arg::with_name("pic_format").short("p").long("pictures").required(false).takes_value(true)
            .possible_values(&["png","jpg"]).help("write individual frames of detected pans to disk"))
        .arg(Arg::with_name("opt").long("opt").required(false).takes_value(false)
            .help("optimize composite PNGs for size [slower]"))
        .arg(Arg::with_name("inputs").index(1).multiple(true).required(true)
            .help("videos files to process. specify '-' to read a newline-separated list from stdin.\nExample: find /media/videos -type f -name '*.mkv' | stitch-animation -"))
        .arg(Arg::with_name("seek_to").short("s").takes_value(true).help("seek to frame number [currently inaccurate, specify a lower number than the desired actual frame]"))
        .arg(Arg::with_name("N").short("n").takes_value(true).help("process at most N frames, after seeking"))
        .arg(Arg::with_name("log").long("log").takes_value(false).help("log debug info"))
        .arg(Arg::with_name("min").long("min").takes_value(true)
            .default_value("20")
            .help("composites must be at least min% larger than the video frame size [higher = faster, may miss small pans]"))
        .arg(Arg::with_name("S").long("sub").takes_value(true)
            .possible_values(&["1","2","4", "8"])
            .help("subsample motion search by a factor of S. default 2 @ >= 720p, 4 @ >= 1080p, 1 otherwise. [higher = faster, less accurate]"))
        .get_matches();


    ffmpeg::init().unwrap();



    if !cfg!(debug_assertions) {
        unsafe {
            ffmpeg::ffi::av_log_set_level(ffmpeg::ffi::AV_LOG_ERROR);
        }
    }

    let config = Config {
        stitch: !matches.is_present("nostitch"),
        log: matches.is_present("log"),
        single_frame_format: value_t!(matches, "pic_format", Format).unwrap_or(Format::NULL),
        seek: value_t!(matches, "seek_to", u32).unwrap_or(0),
        subsample: value_t!(matches, "S", u8).unwrap_or(0),
        min_expand: (value_t!(matches, "min", u16).unwrap() as f32 / 100.0) + 1.0,
        max_frames: value_t!(matches, "N", u32).unwrap_or(std::u32::MAX),
        optimize: matches.is_present("opt"),
    };

    for p in matches.values_of_os("inputs").unwrap().map(Path::new) {
        if p == Path::new("-") {
            let stdin = std::io::stdin();
            let reader = stdin.lock();
            for line in reader.lines() {
                process_video(Path::new(&line.unwrap()), config);
            }
            continue;
        }
        process_video(p, config);
    }

    let counts = motion::search::COUNTS.load(atomic::Ordering::Relaxed);
    println!("loops:{} points:{}", counts.0, counts.1);
}
