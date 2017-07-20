use euclid::{rect,vec2, Rect, TypedSize2D, UnknownUnit};
use ffmpeg::frame::Video;
use ffmpeg::util::format::pixel::Pixel;
use std::collections::HashSet;
use std::fmt;
use motion::search::{self, Estimate};
use itertools::Itertools;

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


    fn align_to(&mut self, other: &AlignedFrame) {

        let estimate = search::search(&self.avframe, &other.avframe, vec![]);

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
            let mut hints = frame.avframe.most_common_vectors();
            hints.extend(new_frame.avframe.most_common_vectors());
            hints.extend(self.frames.iter().filter_map(|f| if f.estimate.x != 0 || f.estimate.y != 0 {Some((f.estimate.x,f.estimate.y))} else {None} ).take(10));
            new_frame.align_to(&frame);
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

        let mut conv = converter((frame.width(),frame.height()), frame.format(), Pixel::RGBA).unwrap();
        let mut intermediate = Video::new(Pixel::RGBA, frame.width(), frame.height());

        {
            let data_out : &mut[(u8,u8,u8,u8)] = canvas.plane_mut(0);

            for fr in self.frames.iter().coalesce(|a,b| {
                if a.offset_x == b.offset_x && a.offset_y == b.offset_y { Ok(a) } else { Err((a,b)) }
            }) {
                conv.run(&fr.avframe, &mut intermediate).unwrap();

                let data_in : &[(u8,u8,u8,u8)] = intermediate.plane(0);

                let x = fr.estimate.x;
                let y = fr.estimate.x;
                let h = intermediate.height();
                let w = intermediate.width();
                let input_stride = intermediate.stride(0) as u32 / 4;

                let r = rect::<_, UnknownUnit>(fr.offset_x,fr.offset_y,w as isize,h as isize);

                for y in 0 .. h {
                    let row_out = (y as isize + fr.offset_y - canvas_dims.min_y()) * canvas_stride as isize;
                    let row_in = y * input_stride;

                    for x in 0 .. w {
                        let x_out = (row_out + x as isize + (fr.offset_x - canvas_dims.min_x())) as usize;
                        let x_in = (row_in + x) as usize;

                        use std::cmp::min;

                        const SEAM_WIDTH : u32 = 8;

                        let edge_dist = min(min(x, w - x - 1), min(y, h - y - 1));
                        if edge_dist < SEAM_WIDTH  {
                            let old = data_out[x_out];
                            // old is not opaque. just paint over it.
                            if old.3 < 255 {
                                data_out[x_out] = data_in[x_in];
                            } else {
                                // alpha-blend
                                let alpha = edge_dist * 255 / SEAM_WIDTH;
                                data_out[x_out].0 = ((data_in[x_in].0 as u32 * alpha + old.0 as u32 * (255 - alpha)) / 255) as u8;
                                data_out[x_out].1 = ((data_in[x_in].1 as u32 * alpha + old.1 as u32 * (255 - alpha)) / 255) as u8;
                                data_out[x_out].2 = ((data_in[x_in].2 as u32 * alpha + old.2 as u32 * (255 - alpha)) / 255) as u8;
                                data_out[x_out].3 = 255;
                            }

                        } else {
                            data_out[x_out] = data_in[x_in];
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