use euclid::{rect,vec2, Rect, UnknownUnit};
use ffmpeg::frame::Video;
use ffmpeg::util::format::pixel::Pixel;
use std::collections::HashSet;
use float_ord::FloatOrd;



#[derive(Copy, Clone, PartialEq)]
pub struct Estimate {
    pub x: isize,
    pub y: isize,
    pub area: u32,
    pub error_sum: u64,
    pub error_area: u64,
}

impl Estimate {

    pub fn still(area: u32) -> Self {
        Estimate {x: 0, y: 0, area, error_sum: 0, error_area: 0}
    }

    pub fn reverse(mut self) -> Self {
        self.x = -self.x;
        self.y = -self.y;
        self
    }


    pub fn area_fraction(&self) -> f32 {
        self.error_area as f32 / self.area as f32
    }

    pub fn error_fraction(&self) -> f32 {
        self.error_sum as f32 / self.area as f32
    }
}

use std::fmt;

impl fmt::Debug for Estimate {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "est: (x:{} y:{} a: {} sum:{} aerr:{} fr:{} afr:{})", self.x, self.y, self.area, self.error_sum, self.error_area, self.error_fraction(), self.area_fraction())
    }
}



pub fn search(a: &Video, b: &Video, hints: Vec<(isize, isize)>) -> Estimate {
    use rayon::prelude::*;

    let w = a.width() as isize;
    let h = a.height() as isize;
    let mut best_match = error_sum(&a, &b, 0,0);
    let ref mut visited = HashSet::new();

    visited.insert((0,0));


    // TODO: if we need additional search phases
    // - do vertical/horizontal first
    // - only switch to diagonal or hexagon if those can't refine any further
    // - greedy multi-threaded search using atomic?

    // exponential cross search search
    loop {
        let Estimate{x,y,..} = best_match;
        let tuples : Vec<_> = (0..11).map(|i| 1 << i).flat_map(|i: isize| {
            vec![
                (x,y+i),(x,y-i), (x+i, y), (x-i,y),
                //(x+i,y+i),(x+i,y-i), (x-i, y+i), (x-i,y-i),
            ].into_iter()
        }).chain(hints.iter().cloned()).filter(|t| {
            match rect::<_,UnknownUnit>(0,0,w,h).intersection(&rect(t.0, t.1, w, h)) {
                None => {return false}
                Some(intersection) => {
                    if intersection.size.width * intersection.size.height < w*h/4 {
                        return false
                    }
                }
            }

            if visited.contains(t) {
                false
            } else {
                visited.insert(*t);
                true
            }
        }).collect();

        let best = tuples.par_iter().map(|&(x,y)| {
            error_sum(&b, &a, x,y)
        }).min_by_key({|est| FloatOrd(est.error_fraction())}).unwrap_or(best_match);

        //print!("{:?} ", best);

        if best.error_fraction() < best_match.error_fraction() {
            best_match = best;
        } else {
            break
        }

    }

    //println!("");

    best_match
}

pub fn error_sum(a: &Video, b: &Video, offset_x: isize, offset_y: isize) -> Estimate {
    let luma_a = a.data(0);
    let luma_b = b.data(0);
    let frame_w = a.width() as isize;
    let frame_h = a.height() as isize;

    let dims_a : Rect<_> = rect(0,0,frame_w,frame_h);
    let dims_b = dims_a.translate(&vec2(offset_x,offset_y));
    let intersection = dims_a.intersection(&dims_b).unwrap();
    // some encodes contain black lines around the edges.
    // focus on the inner parts instead
    let intersection = intersection.inflate(-16,-16);

    // TODO: conditional compilation
    use simd::u8x16;
    use simd::x86::sse2::u64x2;
    use simd::x86::sse2::Sse2U8x16;
    //use simd::x86::ssse3::Ssse3U8x16;
    use std::cmp::{min, max};

    let mut accumulator : u64 = 0;
    let mut area_sum : u64 = 0;

    match a.format() {
        Pixel::YUV420P => {
            //for row in (intersection.min_y()..intersection.max_y()) {
            for row in (intersection.min_y()..intersection.max_y() - 7).step_by(8) {
                let row_a = (row - dims_a.min_y()) * frame_w;
                let row_b = (row - dims_b.min_y()) * frame_w;

                for col in (intersection.min_x()..intersection.max_x() - 15).step_by(16) {
                    let stride_a = (row_a + col - dims_a.min_x()) as usize;
                    let stride_b = (row_b + col - dims_b.min_x()) as usize;

                    let a = u8x16::load(luma_a, stride_a + 0 * frame_w as usize);
                    let b = u8x16::load(luma_b, stride_b + 0 * frame_w as usize);
                    let sad0 = a.sad(b);
                    let a = u8x16::load(luma_a, stride_a + 1 * frame_w as usize);
                    let b = u8x16::load(luma_b, stride_b + 1 * frame_w as usize);
                    let sad1 = a.sad(b);
                    let a = u8x16::load(luma_a, stride_a + 2 * frame_w as usize);
                    let b = u8x16::load(luma_b, stride_b + 2 * frame_w as usize);
                    let sad2 = a.sad(b);
                    let a = u8x16::load(luma_a, stride_a + 3 * frame_w as usize);
                    let b = u8x16::load(luma_b, stride_b + 3 * frame_w as usize);
                    let sad3 = a.sad(b);
                    let a = u8x16::load(luma_a, stride_a + 4 * frame_w as usize);
                    let b = u8x16::load(luma_b, stride_b + 4 * frame_w as usize);
                    let sad4 = a.sad(b);
                    let a = u8x16::load(luma_a, stride_a + 5 * frame_w as usize);
                    let b = u8x16::load(luma_b, stride_b + 5 * frame_w as usize);
                    let sad5 = a.sad(b);
                    let a = u8x16::load(luma_a, stride_a + 6 * frame_w as usize);
                    let b = u8x16::load(luma_b, stride_b + 6 * frame_w as usize);
                    let sad6 = a.sad(b);
                    let a = u8x16::load(luma_a, stride_a + 7 * frame_w as usize);
                    let b = u8x16::load(luma_b, stride_b + 7 * frame_w as usize);
                    let sad7 = a.sad(b);

                    let sad = sad1 + sad2 + sad3 + sad4 + sad5 + sad6 + sad7;

                    let a = sad.extract(0);
                    let b = sad.extract(1);
                    accumulator += a + b;
                    area_sum += min(8*8, a) + min(8*8, b);
                }
            }
        }
        Pixel::YUV420P10LE => {
            // little-endian 2bytes per value
            // 1 byte integer + 1byte fractional value,
            // just mask away the fraction for the SAD calculation
            const UPPER_MASK : u8x16 = u8x16::new(0xFF,0,0xFF,0,0xFF,0,0xFF,0,0xFF,0,0xFF,0,0xFF,0,0xFF, 0);

            for row in intersection.min_y() .. intersection.max_y() {
                // TODO: eliminate multiplication
                let row_a = (row - dims_a.min_y()) * frame_w * 2;
                let row_b = (row - dims_b.min_y()) * frame_w * 2;

                for col in (intersection.min_x() .. intersection.max_x() - 15).step_by(16) {
                    let stride_a = row_a + col * 2 - dims_a.min_x() * 2;
                    let stride_b = row_b + col * 2 - dims_b.min_x() * 2;

                    let a1 = u8x16::load(luma_a, stride_a as usize) & UPPER_MASK;
                    let a2 = u8x16::load(luma_a, (stride_a + 16) as usize) & UPPER_MASK;
                    let b1 = u8x16::load(luma_b, stride_b as usize) & UPPER_MASK;
                    let b2 = u8x16::load(luma_b, (stride_b + 16) as usize) & UPPER_MASK;

                    let (sad1,sad2) = (a1.sad(b1), a2.sad(b2));
                    let (s1,s2) = (sad1.extract(0), sad1.extract(1));
                    let (s3,s4) = (sad2.extract(0), sad2.extract(1));

                    accumulator +=  s1 + s2 + s3 + s4;
                    area_sum += min(4, s1) + min(4, s2) + min(4, s3) + min(4, s4);


                }
            }
        },
        _ => unimplemented!("pixel formats other than 8 and 10bit 4:2:0")
    };

    let pixels = (intersection.size.width & !0x0f) * intersection.size.height &!0x07;

    Estimate {
        error_sum: accumulator as u64,
        error_area: area_sum as u64,
        x: offset_x,
        y: offset_y,
        area: pixels as u32
    }

}

mod test {

}