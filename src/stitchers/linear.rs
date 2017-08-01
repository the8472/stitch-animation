use euclid::{rect,Rect, TypedSize2D, UnknownUnit};
use ffmpeg::frame::Video;
use ffmpeg::util::format::pixel::Pixel;
use std::fmt;
use motion::search::{self, Estimate};
use oxipng;
use std::collections::HashSet;
use ffmpeg;


// ideas/todo
// - diamond search with image pyramid
// - half-pel motion bilinear blending
// - chroma me or me in RGB?
// - scale, rotate (or restricted affine transform); use image and imageproc crates?
// - smart blending (gradient blending, N-layer outlier removal)
// - optimize blending part
//   - cull overlapping frames when blending / paint less
// - detect and crop letterbox
// [x] either penalize intersections with low dynamic range, e.g. black fringes or boost smaller motions
// - clamping of per-pixel error contribution (static logos!) so that they don't dominate more gradual changes
// - deal with lighting changes? sobel filter on source frames before SAD?
// - reduce motion search cost by only diffing every Nth frame and filling the gaps when we detect a scene
// - simplify run detection logic by operating on windows over buffers of frames


struct AlignedFrame {
    avframe: Video,
    offset_x: isize,
    offset_y: isize,
    estimate: Estimate,
    sar: ffmpeg::Rational
}


impl AlignedFrame {


    fn compute_estimate(&mut self, other: &AlignedFrame, hint: Option<(isize, isize)> ) {

        let estimate = search::search(&self.avframe, &other.avframe, hint, 0);
        self.estimate = estimate;
    }

    fn set_estimate(&mut self, estimate: Estimate) {
        self.estimate = estimate;
    }

    fn offset_from_estimate(&mut self, other: &AlignedFrame) {
        self.offset_x = other.offset_x + self.estimate.x;
        self.offset_y = other.offset_y + self.estimate.y;
    }
}


impl fmt::Debug for AlignedFrame {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", (self.offset_x, self.offset_y, self.estimate))
    }
}

pub struct LinStitcher {
    start_frame: u32,
    frames: Vec<AlignedFrame>
}

impl LinStitcher {
    pub fn new() -> LinStitcher {
        LinStitcher{start_frame: 0, frames: vec![]}
    }

    pub fn set_start_frame(&mut self, frame_idx: u32) {
        self.start_frame = frame_idx;
    }

    pub fn add_frame(&mut self, frame: Video, motion: Option<Estimate>, sar: ffmpeg::Rational) {
        let area = frame.height() * frame.width();
        let mut new_frame = AlignedFrame{avframe: frame, offset_x: 0, offset_y: 0, estimate: Estimate::still(area), sar};

        if let Some(m) = motion {
            new_frame.set_estimate(m);
        }

        if let Some(frame) = self.frames.iter().rev()
            .filter(|f| f.estimate.x != 0 || f.estimate.y != 0)
            .chain(self.frames.iter().take(1))
            .next() {
            if motion.is_none() {
                let hint = self.frames.iter().rev().map(|f| (f.estimate.x,f.estimate.y)).next();
                new_frame.compute_estimate(&frame, hint);
            }
            new_frame.offset_from_estimate(&frame);
        }


        self.frames.push(new_frame);
    }

    pub fn expansion_ratio(&self) -> f32 {
        let frame = &self.frames[0].avframe;
        let frame_size = frame.width() * frame.height();
        let merged_size = self.dims();
        (merged_size.size.width * merged_size.size.height) as f32 / frame_size as f32
    }

    fn dims(&self) -> ::euclid::Rect<isize> {
        let mut canvas_dims : Rect<_> = rect(0,0,0,0);

        for fr in &self.frames {
            let r = rect(fr.offset_x, fr.offset_y, fr.avframe.width() as isize, fr.avframe.height() as isize);
            canvas_dims = canvas_dims.union(&r);
        }

        canvas_dims
    }

    pub fn merge(mut self) -> Video {
        let canvas_dims = self.dims();

        let mut canvas = Video::new(Pixel::RGBA, canvas_dims.size.width as u32, canvas_dims.size.height as u32);
        let canvas_stride = canvas.stride(0) / 4;

        let sar = self.frames[0].sar;
        let (w,h,f) = {
            let frame = &self.frames[0].avframe;
            (frame.width(),frame.height(),frame.format())
        };

        let slice_target_height = canvas_dims.size.height as usize / ::rayon::current_num_threads();
        let slice_chunk_size = slice_target_height * canvas_stride;

        // TODO 16bit PNG support
        use ffmpeg::software::scaling::{flag, Context};
        use ffmpeg::util::color::range::Range;
        let mut flags = flag::ACCURATE_RND;
        flags.insert(flag::ERROR_DIFFUSION);
        flags.insert(flag::BICUBIC);
        flags.insert(flag::FULL_CHR_H_INP);
        flags.insert(flag::FULL_CHR_H_INT);

        let mut conv = Context::get(f, w, h, Pixel::RGBA, w, h, flags).unwrap();
        let mut intermediate = Video::new(Pixel::RGBA, w, h);
        intermediate.set_color_range(Range::JPEG);

        {
            let data_out : &mut[(u8,u8,u8,u8)] = canvas.plane_mut(0);

            for (_, fr) in self.frames.drain(..).enumerate().filter(|&(i,ref f)| i == 0 || f.estimate.x != 0 || f.estimate.y != 0) {
                conv.run(&fr.avframe, &mut intermediate).unwrap();

                let data_in : &[(u8,u8,u8,u8)] = intermediate.plane(0);

                let x = fr.estimate.x;
                let y = fr.estimate.x;
                let h = intermediate.height();
                let w = intermediate.width();
                let input_stride = intermediate.stride(0) as u32 / 4;

                const SEAM_WIDTH : u32 = 8;

                let idx_out = (fr.offset_x - canvas_dims.min_x()) + (fr.offset_y - canvas_dims.min_y()) * canvas_stride as isize;

                for y in 0 .. h {
                    let idx_out = idx_out + y as isize * canvas_stride as isize;
                    let idx_in = y * input_stride;

                    use std::cmp::min;

                    let vertical_edge_dist = min(y, h - y - 1);

                    for x in 0 .. w {
                        let idx_out = (idx_out + x as isize) as usize;
                        let idx_in = (idx_in + x) as usize;

                        let edge_dist = min(min(x, w - x - 1), vertical_edge_dist);
                        if edge_dist < SEAM_WIDTH  {
                            let old = data_out[idx_out];
                            // old is not opaque. just paint over it.
                            if old.3 < 255 {
                                data_out[idx_out] = data_in[idx_in];
                            } else {
                                // alpha-blend
                                let alpha = edge_dist * 255 / SEAM_WIDTH;
                                data_out[idx_out].0 = ((data_in[idx_in].0 as u32 * alpha + old.0 as u32 * (255 - alpha)) / 255) as u8;
                                data_out[idx_out].1 = ((data_in[idx_in].1 as u32 * alpha + old.1 as u32 * (255 - alpha)) / 255) as u8;
                                data_out[idx_out].2 = ((data_in[idx_in].2 as u32 * alpha + old.2 as u32 * (255 - alpha)) / 255) as u8;
                                data_out[idx_out].3 = 255;
                            }

                        } else {

                            data_out[idx_out] = data_in[idx_in];
                        }


                    }
                }
            }
        }

        if sar.numerator() > 0 && (sar.denominator() != 1 || sar.numerator() != 1) {
            let (w,h) = (canvas.width(),canvas.height());
            let (dw, dh) = if sar.numerator() > sar.denominator() {
                (w * sar.numerator() as u32 / sar.denominator() as u32,h)
            } else {
                (w, h * sar.denominator() as u32 / sar.numerator() as u32)
            };
            let mut conv = Context::get(Pixel::RGBA, w, h, Pixel::RGBA, dw , dh, flags).unwrap();
            let mut aspect_corrected = Video::empty(); // ::new(Pixel::RGBA, display_w , display_h );
            aspect_corrected.set_color_range(Range::JPEG);
            conv.run(&canvas, &mut aspect_corrected).unwrap();
            ::std::mem::replace(&mut canvas, aspect_corrected);
        }

        return canvas
    }

    pub fn write_linear_stitch(self, optimize: bool, dir: &::std::path::Path) {
        let start = self.start_frame;
        let path = dir.join(format!("{:06}_lin.png", start));

        {
            let frame = self.merge();

            let mut octx = ffmpeg::format::output(&path).unwrap();
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
                    encoder.flush(&mut packet).unwrap();
                }
                Err(e) => eprintln!("{:?}", e)
            }

            octx.write_header().unwrap();
            packet.write(&mut octx).unwrap();
            octx.write_trailer().unwrap();
        }

        if optimize {
            let mut options = oxipng::Options::default();
            let mut zm = HashSet::new();
            zm.insert(9);
            options.memory = zm;
            let mut zf = HashSet::new();
            zf.insert(5);
            options.backup = false;
            options.filter = zf;
            options.verbosity = None;
            options.out_file = path.clone();

            oxipng::optimize(&path, &options).unwrap();
        }
    }

}



impl fmt::Debug for LinStitcher {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        for frame in &self.frames {
            writeln!(f, "{:?}", frame)?;
        }
        Ok(())
    }
}