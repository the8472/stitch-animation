use euclid::{rect,Rect, TypedSize2D, UnknownUnit};
use ffmpeg::frame::Video;
use ffmpeg::util::format::pixel::Pixel;
use std::fmt;
use motion::search::{self, Estimate};

// TODO:
// [ ] real diamond search with image pyramid
// [ ] half-pel motion bilinear blending
// [ ] scale, rotate (or restricted affine transform); use image and imageproc crates?
// [ ] smart blending (gradient blending, N-layer outlier removal)
// [ ] optimize blending part
// [ ] detect and crop letterbox
// [ ] clamping of per-pixel error contribution (static logos!) so that they don't dominate more gradual changes
// [ ] deal with lighting changes? search chroma? error sum opt instead of area?

struct AlignedFrame {
    avframe: Video,
    offset_x: isize,
    offset_y: isize,
    estimate: Estimate
}


impl AlignedFrame {


    fn align_to(&mut self, other: &AlignedFrame, hint: Option<(isize,isize)> ) {

        let estimate = search::search(&self.avframe, &other.avframe, hint);

        self.offset_x = other.offset_x + estimate.x;
        self.offset_y = other.offset_y + estimate.y;
        self.estimate = estimate;
    }
}


impl fmt::Debug for AlignedFrame {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", (self.offset_x, self.offset_y, self.estimate))
    }
}

pub struct LinStitcher {
    frames: Vec<AlignedFrame>
}

impl LinStitcher {
    pub fn new() -> LinStitcher {
        LinStitcher{frames: vec![]}
    }

    pub fn add_frame(&mut self, frame: Video) {
        let area = frame.height() * frame.width();
        let mut new_frame = AlignedFrame{avframe: frame, offset_x: 0, offset_y: 0, estimate: Estimate::still(area)};

        // if several frames have the same offsets we take the oldest one
        // this should provide better estimates in the presence of sub-pixel motion
        if let Some(frame) = self.frames.iter().rev()
            .filter(|f| f.estimate.x != 0 || f.estimate.y != 0)
            .chain(self.frames.iter().take(1))
            .next() {
            use motion::vectors::ToMotionVectors;
            let hint = self.frames.iter().rev().map(|f| (f.estimate.x,f.estimate.y)).next();
            new_frame.align_to(&frame, hint);
        }

        self.frames.push(new_frame);
    }

    pub fn merge(&self) -> Video {
        let mut canvas_dims : Rect<_> = rect(0,0,0,0);

        for fr in &self.frames {
            let r = rect(fr.offset_x, fr.offset_y, fr.avframe.width() as isize, fr.avframe.height() as isize);
            canvas_dims = canvas_dims.union(&r);
        }

        let mut canvas = Video::new(Pixel::RGBA, canvas_dims.size.width as u32, canvas_dims.size.height as u32);
        let canvas_stride = canvas.stride(0) / 4;

        use ffmpeg::software::converter;

        let frame = &self.frames[0].avframe;

        let slice_target_height = canvas_dims.size.height as usize / ::rayon::current_num_threads();
        let slice_chunk_size = slice_target_height * canvas_stride;

        let mut conv = converter((frame.width(),frame.height()), frame.format(), Pixel::RGBA).unwrap();
        let mut intermediate = Video::new(Pixel::RGBA, frame.width(), frame.height());


        {
            let data_out : &mut[(u8,u8,u8,u8)] = canvas.plane_mut(0);

            for (_, fr) in self.frames.iter().enumerate().filter(|&(i,f)| i == 0 || f.estimate.x != 0 || f.estimate.y != 0) {
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

        canvas
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