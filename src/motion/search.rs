use euclid::{rect,vec2, Rect, UnknownUnit};
use ffmpeg::frame::Video;
use ffmpeg::util::format::pixel::Pixel;
use std::collections::HashSet;
use float_ord::FloatOrd;
use simd::u8x16;
// TODO: generic implementation
use simd::x86::sse2::Sse2U8x16;
use simd::x86::ssse3::Ssse3U8x16;
use stdsimd::simd as simd2;
use stdsimd::vendor as x86;
use std::cmp::{min, max};


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

use atomic::{Atomic,Ordering};

pub static COUNTS: Atomic<(usize,usize)> = Atomic::new((0,0));


#[derive(PartialEq)]
enum Mode {
    ConstrainedCross,
    UnconstrainedCross
}

use self::Mode::*;

pub fn search(a: &Video, b: &Video, hint: Option<(isize, isize)>) -> Estimate {
    use rayon::prelude::*;

    let w = a.width() as isize;
    let h = a.height() as isize;

    let (x,y) = if let Some(hint) = hint {
        if hint.0.abs() >= w / 2 || hint.1.abs() >= h / 2 {
            (0,0)
        } else {
            hint
        }
    } else {
        (0,0)
    };

    use std::u64;

    let mut best_match = Estimate{x:x,y:y,area:(w*h) as u32,error_sum:u64::MAX,error_area:u64::MAX}; //error_sum(&a, &b, x,y);
    let ref mut visited = HashSet::with_capacity(180);

    //visited.insert((0,0));


    // TODO: if we need additional search phases
    // - do vertical/horizontal first
    // - only switch to diagonal or hexagon if those can't refine any further
    // - greedy multi-threaded search using atomic?

    // exponential cross search search
    let mut i = 0;

    let mut mode = UnconstrainedCross;

    loop {
        i+=1;
        let Estimate{x,y,..} = best_match;

        let range = match mode {
            UnconstrainedCross => 11,
            ConstrainedCross => 4
        };

        let tuples : Vec<_> = (0..range).map(|i| 1 << i).chain(Some(0).into_iter()).flat_map(|i: isize| {
            let directions = vec![
                (x,y+i),(x,y-i), (x+i, y), (x-i,y),
            ];

            /*
            if mode == ConstrainedCross {
                directions.extend([(x+i,y+i),(x+i,y-i), (x-i, y+i), (x-i,y-i)].into_iter());
            }*/

            directions.into_iter()
        }).chain(Some((0,0)).into_iter()).filter(|t| {
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

        let found = tuples.par_iter().map(|&(x,y)| {
            error_sum(&b, &a, x,y)
        }).min_by_key({|est| FloatOrd(est.error_fraction())}).unwrap_or(best_match);

        //print!("{:?} ", found);

        if found.error_fraction() < best_match.error_fraction() {
            let taxicab = max((found.x - best_match.x).abs(), (found.y - best_match.y).abs());

            if taxicab < 1<<4 {
                mode = ConstrainedCross;
            }

            best_match = found;
        } else {
            match mode {
                UnconstrainedCross => break,
                ConstrainedCross => {
                    // escape local minima
                    mode = UnconstrainedCross
                }
            }
        }

    }

    //println!("");

    //println!("loops: {} points:{} ", i, visited.len());
    loop {
        let current = COUNTS.load(Ordering::Relaxed);
        let mut new = current;
        new.0 += i;
        new.1 += visited.len();
        if let Ok(_) = COUNTS.compare_exchange_weak(current, new, Ordering::Relaxed, Ordering::Relaxed) {
            break;
        }

    }

    best_match
}


// YUV420P. 16x8 pixels
macro_rules! sad8 {
    ($a:expr, $b:expr, $offset_a:expr, $offset_b:expr, $stride:expr, $i:expr ) => {
        {
            let a = u8x16::load($a, $offset_a + $i * $stride);
            let b = u8x16::load($b, $offset_b + $i * $stride);
            a.sad(b)
        }
    };
}

// 32x8 pixels
macro_rules! sad8avx {
    ($a:expr, $b:expr, $offset_a:expr, $offset_b:expr, $stride:expr, $i:expr) => {
        {
            // TODO: investigate gathering ops to fetch 2x16bytes instead of 1x32
            let a = simd2::u8x32::load($a, $offset_a + $i * $stride);
            let b = simd2::u8x32::load($b, $offset_b + $i * $stride);
            x86::_mm256_sad_epu8(a,b)
        }
    };
}


// YUV420P10LE. 16x8 pixels
// little-endian 2bytes per value
// 1 byte integer + 1byte fractional value,
// just mask away the fractional part for the SAD calculation
const UPPER_MASK : u8x16 = u8x16::new(0xFF,0,0xFF,0,0xFF,0,0xFF,0,0xFF,0,0xFF, 0,0xFF, 0,0xFF, 0);
const MOVE_UP : u8x16    = u8x16::new(0x80,0,0x80,2,0x80,4,0x80,6,0x80,8,0x80,10,0x80,12,0x80,14);

macro_rules! sad16 {
    ($a:expr, $b:expr, $offset_a:expr, $offset_b:expr, $stride:expr, $i:expr ) => {
        {
            let a1 = u8x16::load($a, $offset_a + $i * $stride) & UPPER_MASK;
            let a2 = u8x16::load($a, $offset_a + $i * $stride + 16).shuffle_bytes(MOVE_UP);
            let a = a1 | a2;
            let b1 = u8x16::load($b, $offset_b + $i * $stride) & UPPER_MASK;
            let b2 = u8x16::load($b, $offset_b + $i * $stride + 16).shuffle_bytes(MOVE_UP);
            let b = b1 | b2;
            a.sad(b)
        }
    };
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


    let mut accumulator : u64 = 0;
    let mut area_sum : u64 = 0;

    match a.format() {
        Pixel::YUV420P => {
            let stride = frame_w as usize;

            let idx_a = - dims_a.min_x() - dims_a.min_y() * frame_w;
            let idx_b = - dims_b.min_x() - dims_b.min_y() * frame_w;

            if cfg![target_feature = "avx2"] {

                for row in (intersection.min_y()..intersection.max_y() - 7).step_by(8) {
                    let idx_a = idx_a + row  * frame_w;
                    let idx_b = idx_b + row  * frame_w;

                    for col in (intersection.min_x()..intersection.max_x() - 31).step_by(32) {
                        let idx_a = (idx_a + col ) as usize;
                        let idx_b = (idx_b + col ) as usize;

                        let sad0 = sad8avx!(luma_a,luma_b,idx_a, idx_b, stride, 0);
                        let sad1 = sad8avx!(luma_a,luma_b,idx_a, idx_b, stride, 1);
                        let sad2 = sad8avx!(luma_a,luma_b,idx_a, idx_b, stride, 2);
                        let sad3 = sad8avx!(luma_a,luma_b,idx_a, idx_b, stride, 3);
                        let sad4 = sad8avx!(luma_a,luma_b,idx_a, idx_b, stride, 4);
                        let sad5 = sad8avx!(luma_a,luma_b,idx_a, idx_b, stride, 5);
                        let sad6 = sad8avx!(luma_a,luma_b,idx_a, idx_b, stride, 6);
                        let sad7 = sad8avx!(luma_a,luma_b,idx_a, idx_b, stride, 7);

                        let sad = sad0 + sad1 + sad2 + sad3 + sad4 + sad5 + sad6 + sad7;

                        let a = sad.extract(0);
                        let b = sad.extract(1);
                        let c = sad.extract(2);
                        let d = sad.extract(3);
                        accumulator += a + b + c + d;
                        area_sum += min(8*8, a) + min(8*8, b) + min(8*8, c) + min(8*8, d);
                    }
                }
            } else {
                for row in (intersection.min_y()..intersection.max_y() - 7).step_by(8) {
                    let idx_a = idx_a + row  * frame_w;
                    let idx_b = idx_b + row  * frame_w;

                    for col in (intersection.min_x()..intersection.max_x() - 15).step_by(16) {
                        let idx_a = (idx_a + col ) as usize;
                        let idx_b = (idx_b + col ) as usize;

                        let sad0 = sad8!(luma_a,luma_b,idx_a, idx_b, stride, 0);
                        let sad1 = sad8!(luma_a,luma_b,idx_a, idx_b, stride, 1);
                        let sad2 = sad8!(luma_a,luma_b,idx_a, idx_b, stride, 2);
                        let sad3 = sad8!(luma_a,luma_b,idx_a, idx_b, stride, 3);
                        let sad4 = sad8!(luma_a,luma_b,idx_a, idx_b, stride, 4);
                        let sad5 = sad8!(luma_a,luma_b,idx_a, idx_b, stride, 5);
                        let sad6 = sad8!(luma_a,luma_b,idx_a, idx_b, stride, 6);
                        let sad7 = sad8!(luma_a,luma_b,idx_a, idx_b, stride, 7);

                        let sad = sad0 + sad1 + sad2 + sad3 + sad4 + sad5 + sad6 + sad7;

                        let a = sad.extract(0);
                        let b = sad.extract(1);
                        accumulator += a + b;
                        area_sum += min(8*8, a) + min(8*8, b);
                    }
                }
            }

        }
        Pixel::YUV420P10LE => {
            let frame_stride = (frame_w * 2) as usize;

            for row in (intersection.min_y() .. intersection.max_y()-7).step_by(8) {
                let row_a = (row - dims_a.min_y()) * frame_stride as isize;
                let row_b = (row - dims_b.min_y()) * frame_stride as isize;

                for col in (intersection.min_x() .. intersection.max_x() - 15).step_by(16) {
                    let off_a = (row_a + col * 2 - dims_a.min_x() * 2) as usize;
                    let off_b = (row_b + col * 2 - dims_b.min_x() * 2) as usize;


                    let sad0 = sad16!(luma_a,luma_b,off_a, off_b, frame_stride, 0);
                    let sad1 = sad16!(luma_a,luma_b,off_a, off_b, frame_stride, 1);
                    let sad2 = sad16!(luma_a,luma_b,off_a, off_b, frame_stride, 2);
                    let sad3 = sad16!(luma_a,luma_b,off_a, off_b, frame_stride, 3);
                    let sad4 = sad16!(luma_a,luma_b,off_a, off_b, frame_stride, 4);
                    let sad5 = sad16!(luma_a,luma_b,off_a, off_b, frame_stride, 5);
                    let sad6 = sad16!(luma_a,luma_b,off_a, off_b, frame_stride, 6);
                    let sad7 = sad16!(luma_a,luma_b,off_a, off_b, frame_stride, 7);

                    let sad = sad0 + sad1 + sad2 + sad3 + sad4 + sad5 + sad6 + sad7;
                    let a = sad.extract(0);
                    let b = sad.extract(1);
                    accumulator += a + b;
                    area_sum += min(8*8, a) + min(8*8, b);
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