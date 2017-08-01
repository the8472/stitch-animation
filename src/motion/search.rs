use euclid::{rect,vec2, Rect, UnknownUnit};
use ffmpeg::frame::Video;
use ffmpeg::util::format::pixel::Pixel;
use std::collections::HashSet;
use float_ord::FloatOrd;
use simd::u8x16;
#[cfg(target_feature = "sse2")]
use simd::x86::sse2::Sse2U8x16;
#[cfg(target_feature = "ssse3")]
use simd::x86::ssse3::Ssse3U8x16;
use std::cmp::{min, max};

#[derive(Copy)]
pub struct Estimate {
    pub x: isize,
    pub y: isize,
    pub area: u32,
    pub error_sum: u64,
    pub error_area: u64,
    pub histogram: [u16 ; 256]
}

impl PartialEq for Estimate {

    fn eq(&self, other: &Estimate) -> bool {
        self.x == other.x && self.y == other.y && self.error_sum == other.error_sum && self.error_area == other.error_area && self.histogram[..] == other.histogram[..]
    }
}

impl Clone for Estimate {
    fn clone(&self) -> Self {
        *self
    }
}

impl Estimate {

    pub fn still(area: u32) -> Self {
        Estimate {x: 0, y: 0, area, error_sum: 0, error_area: 0, histogram: [0 ; 256]}
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

    pub fn mode(&self) -> u8 {
        self.histogram[..].into_iter().enumerate().max_by_key(|&(_,v)| v).unwrap().0 as u8
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
        let threshold = (self.hist_pop() as f32 * q) as u16;
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

}

use std::fmt;

impl fmt::Debug for Estimate {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        let num_blocks = self.hist_pop();
        /*
        let normalized = self.histogram[..].iter().map(|v| *v as f32 / num_blocks as f32).map(|v| {
            format!("{:02.3}", v)
        }).join(", ");
        */

        write!(f, "est: (x:{} y:{} a: {} sum:{} aerr:{} fr:{} afr:{} | avg{} mode{} min{} 10th{} 25th{} 50th{} 75th{} 90th{} max{})",
               self.x, self.y, self.area, self.error_sum, self.error_area, self.error_fraction(), self.area_fraction(),
               self.avg(), self.mode(), self.min(),  self.quantile(0.1), self.quantile(0.25), self.quantile(0.5), self.quantile(0.75), self.quantile(0.9), self.max())
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

pub fn search(current: &Video, predecessor: &Video, hint: Option<(isize, isize)>, subsample: u8) -> Estimate {
    use rayon::prelude::*;

    let w = current.width() as isize;
    let h = current.height() as isize;

    let subsample = match subsample {
        1 | 2 | 4 | 8 => subsample.trailing_zeros(),
        _ if h >= 1080 => 2,
        _ if h >= 720 => 1,
        _ => 0
    };

    let (x,y) = if let Some(hint) = hint {
        if hint.0.abs() >= w / 2 || hint.1.abs() >= h / 2 {
            (0,0)
        } else {
            hint
        }
    } else {
        (0,0)
    };

    use std::{u64,u16};

    let mut histo = [0 ; 256];
    histo[255] = 255;

    let mut best_match = Estimate{x:x,y:y,area:(w*h) as u32,error_sum:u64::MAX,error_area:u64::MAX, histogram:histo}; //error_sum(&a, &b, x,y);
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

        let tuples : Vec<_> = (0..range).map(|i| 1 << i).flat_map(|i: isize| {
            let mut directions = vec![
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
                    if intersection.size.width * intersection.size.height < w*h/3 {
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
            error_sum(&predecessor, &current, x,y, subsample as u8)
        }).min_by_key({|est| FloatOrd(est.error_fraction())}).unwrap_or(best_match);

        //print!("{:?} ", found);

        if found.error_fraction() <= best_match.error_fraction() && (found.error_fraction() - best_match.error_fraction()).abs() > 0.00001  {
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


macro_rules! pxloop {
    ($rect: expr, $bs:expr, $da: expr, $db: expr, $a:ident, $b:ident, $bpp: expr, $stride: expr, $body:block) => {
        {
            let idx_a = - $da.min_x() * $bpp - $da.min_y() * $stride as isize;
            let idx_b = - $db.min_x() * $bpp - $db.min_y() * $stride as isize;

            let blocksize = $bs;

            for row in ($rect.min_y()..$rect.max_y() - (blocksize.1 - 1)).step_by(blocksize.1 as usize) {
                let idx_a = idx_a + row  * $stride as isize;
                let idx_b = idx_b + row  * $stride as isize;

                for col in ($rect.min_x()..$rect.max_x() - (blocksize.0 - 1)).step_by(blocksize.0 as usize) {
                    let $a = (idx_a + col * $bpp) as usize;
                    let $b = (idx_b + col * $bpp) as usize;

                    $body
                }
            }
        }
    }
}


#[cfg(target_feature = "sse2")]
macro_rules! sad8 {
    ($a:expr, $b:expr, $offset_a:expr, $offset_b:expr, $stride:expr, $i:expr ) => {
        {
            let a = u8x16::load($a, $offset_a + $i * $stride);
            let b = u8x16::load($b, $offset_b + $i * $stride);
            a.sad(b)
        }
    };
}

#[cfg(target_feature = "avx2")]
macro_rules! sad8avx {
    ($a:expr, $b:expr, $offset_a:expr, $offset_b:expr, $stride:expr, $i:expr) => {
        {
            use simd::x86::avx::u8x32;
            use simd::x86::avx2::*;
            let a = u8x32::load($a, $offset_a + $i * $stride);
            let b = u8x32::load($b, $offset_b + $i * $stride);
            a.sad(b)
        }
    };
}


// YUV420P10LE. 16x8 pixels
// little-endian 2bytes per value
// 1 byte integer + 1byte fractional value,
// just mask away the fractional part for the SAD calculation
const UPPER_MASK : u8x16 = u8x16::new(0xFF,0,0xFF,0,0xFF,0,0xFF,0,0xFF,0,0xFF, 0,0xFF, 0,0xFF, 0);
const MOVE_UP : u8x16    = u8x16::new(0x80,0,0x80,2,0x80,4,0x80,6,0x80,8,0x80,10,0x80,12,0x80,14);

#[cfg(target_feature = "ssse3")]
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

#[cfg(target_feature = "avx2")]
use simd::x86::avx::u8x32;

#[cfg(target_feature = "avx2")]
// shuffle integer parts to lanes 0 and 2, fractional parts to lanes 1 and 3
const MOVE_LANE : u8x32    = u8x32::new(0,2,4,6,8,10,12,14,1,3,5,7,9,11,13,15,
                                        0,2,4,6,8,10,12,14,1,3,5,7,9,11,13,15,);

#[cfg(target_feature = "avx2")]
macro_rules! sad16avx {
    ($a:expr, $b:expr, $offset_a:expr, $offset_b:expr, $stride:expr, $i:expr) => {
        {
            use simd::x86::avx::u8x32;
            use simd::x86::avx2::*;
            use ::motion::search::{MOVE_LANE};
            let a = u8x32::load($a, $offset_a + $i * $stride).shuffle_bytes(MOVE_LANE);
            let b = u8x32::load($b, $offset_b + $i * $stride).shuffle_bytes(MOVE_LANE);
            a.sad(b)
        }
    };
}

#[derive(Clone, Copy)]
struct Params {
    bs: (isize,isize),
    strides: [usize; 8],
    xoffset: [usize; 8]
}

const AVX2_YUV8 : [Params; 4] = [
    Params{bs:(32,8), strides:[0,1,2,3,4,5,6,7],xoffset:[0,0,0,0,0,0,0,0]},
    Params{bs:(64,8), strides:[0,1,2,3,4,5,6,7],xoffset:[0,16,0,16,0,16,0,16]},
    Params{bs:(64,16),strides:[0,1,2,6,8,10,12,14],xoffset:[0,16,0,16,0,16,0,16]},
    Params{bs:(64,32),strides:[0,1,2,6,8,18,24,28],xoffset:[0,16,0,16,0,16,0,16]}
];

const AVX2_YUV10 : [Params; 4] = [
    Params{bs:(16,8), strides:[0,1,2,3,4,5,6,7],xoffset:[0,0,0,0,0,0,0,0]},
    Params{bs:(16,16), strides:[0,1,2,6,8,10,12,14],xoffset:[0,0,0,0,0,0,0,0]},
    Params{bs:(32,16),strides:[0,1,2,6,8,10,12,14],xoffset:[0,16,0,16,0,16,0,16]},
    Params{bs:(32,32),strides:[0,1,2,6,8,18,24,28],xoffset:[0,16,0,16,0,16,0,16]}
];


pub fn error_sum(a: &Video, b: &Video, offset_x: isize, offset_y: isize, subsample: u8) -> Estimate {
    let stride = a.stride(0);
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


    let mut accumulator : u64 = 0;
    let mut area_sum : u64 = 0;

    let mut histogram = [0 ; 256];

    let mut wmask = 0x0f;
    let mut hmask = 0x07;

    // TODO: generic fallbacks

    #[allow(unreachable_patterns)]
    match a.format() {
        #[cfg(target_feature = "avx2")]
        Pixel::YUV420P | Pixel::YUV444P => {

            let Params{bs: blocksize, xoffset:xoff, strides: yoff} = AVX2_YUV8[subsample as usize];
            wmask = blocksize.0 - 1;
            hmask = blocksize.1 - 1;

            use simd::x86::avx::u64x4;
            const HIST_MASK : u64x4 = u64x4::splat(0xff);

            pxloop!(intersection, blocksize, dims_a, dims_b, idx_a, idx_b, 1, stride, {
                let sad0 = sad8avx!(luma_a,luma_b,idx_a, idx_b, stride, yoff[0]);
                let sad1 = sad8avx!(luma_a,luma_b,idx_a + xoff[1], idx_b + xoff[1], stride, yoff[1]);
                let sad2 = sad8avx!(luma_a,luma_b,idx_a, idx_b, stride, yoff[2]);
                let sad3 = sad8avx!(luma_a,luma_b,idx_a + xoff[3], idx_b + xoff[3], stride, yoff[3]);
                let sad4 = sad8avx!(luma_a,luma_b,idx_a, idx_b, stride, yoff[4]);
                let sad5 = sad8avx!(luma_a,luma_b,idx_a + xoff[5], idx_b + xoff[5], stride, yoff[5]);
                let sad6 = sad8avx!(luma_a,luma_b,idx_a, idx_b, stride, yoff[6]);
                let sad7 = sad8avx!(luma_a,luma_b,idx_a + xoff[7], idx_b + xoff[7], stride, yoff[7]);

                let sad = sad0 + sad1 + sad2 + sad3 + sad4 + sad5 + sad6 + sad7;

                let (a,b,c,d) = (sad.extract(0),sad.extract(1),sad.extract(2),sad.extract(3));
                accumulator += a + b + c + d;

                // 32 pixels per row * 8 rows / 4 lanes
                // scale down error values
                let hist_idx : u64x4 = (sad >> 6) & HIST_MASK;
                histogram[hist_idx.extract(0) as usize] += 1;
                histogram[hist_idx.extract(1) as usize] += 1;
                histogram[hist_idx.extract(2) as usize] += 1;
                histogram[hist_idx.extract(3) as usize] += 1;

                area_sum += min(8*8, a) + min(8*8, b) + min(8*8, c) + min(8*8, d);
            });
        }
        #[cfg(all(target_feature = "sse2", not(target_feature = "avx2")))]
        Pixel::YUV420P | Pixel::YUV444P => {
            let idx_a = - dims_a.min_x() - dims_a.min_y() * stride as isize;
            let idx_b = - dims_b.min_x() - dims_b.min_y() * stride as isize;

            for row in (intersection.min_y()..intersection.max_y() - 7).step_by(8) {
                let idx_a = idx_a + row  * stride as isize;
                let idx_b = idx_b + row  * stride as isize;

                for col in (intersection.min_x()..intersection.max_x() - 15).step_by(16) {
                    let idx_a = (idx_a + col) as usize;
                    let idx_b = (idx_b + col) as usize;

                    let sad0 = sad8!(luma_a,luma_b,idx_a, idx_b, stride, 0);
                    let sad1 = sad8!(luma_a,luma_b,idx_a, idx_b, stride, 1);
                    let sad2 = sad8!(luma_a,luma_b,idx_a, idx_b, stride, 2);
                    let sad3 = sad8!(luma_a,luma_b,idx_a, idx_b, stride, 3);
                    let sad4 = sad8!(luma_a,luma_b,idx_a, idx_b, stride, 4);
                    let sad5 = sad8!(luma_a,luma_b,idx_a, idx_b, stride, 5);
                    let sad6 = sad8!(luma_a,luma_b,idx_a, idx_b, stride, 6);
                    let sad7 = sad8!(luma_a,luma_b,idx_a, idx_b, stride, 7);

                    let sad = sad0 + sad1 + sad2 + sad3 + sad4 + sad5 + sad6 + sad7;

                    // 16px per row * 8 rows / 2 lanes
                    let a = sad.extract(0);
                    let b = sad.extract(1);
                    accumulator += a + b;
                    area_sum += min(8*8, a) + min(8*8, b);
                    // max error is 256 * 8 * 8
                    use simd::x86::sse2::u64x2;
                    let hist_idx : u64x2 = (sad >> 6) & u64x2::splat(0xff);
                    histogram[hist_idx.extract(0) as usize] += 1;
                    histogram[hist_idx.extract(1) as usize] += 1;
                }
            }
        }
        #[cfg(target_feature = "avx2")]
        Pixel::YUV420P10LE | Pixel::YUV444P10LE => {

            let Params{bs: blocksize, xoffset:xoff, strides: yoff} = AVX2_YUV10[subsample as usize];
            wmask = blocksize.0 - 1;
            hmask = blocksize.1 - 1;

            use simd::x86::avx::u64x4;
            const HIST_MASK : u64x4 = u64x4::splat(0xff);

            pxloop!(intersection, blocksize, dims_a, dims_b, idx_a, idx_b, 2, stride, {
                    let sad0 = sad16avx!(luma_a,luma_b,idx_a, idx_b, stride, yoff[0]);
                    let sad1 = sad16avx!(luma_a,luma_b,idx_a + xoff[1], idx_b + xoff[1], stride, yoff[1]);
                    let sad2 = sad16avx!(luma_a,luma_b,idx_a, idx_b, stride, yoff[2]);
                    let sad3 = sad16avx!(luma_a,luma_b,idx_a + xoff[3], idx_b + xoff[3], stride, yoff[3]);
                    let sad4 = sad16avx!(luma_a,luma_b,idx_a, idx_b, stride, yoff[4]);
                    let sad5 = sad16avx!(luma_a,luma_b,idx_a + xoff[5], idx_b + xoff[5], stride, yoff[5]);
                    let sad6 = sad16avx!(luma_a,luma_b,idx_a, idx_b, stride, yoff[6]);
                    let sad7 = sad16avx!(luma_a,luma_b,idx_a + xoff[7], idx_b + xoff[7], stride, yoff[7]);

                    let sad = sad0 + sad1 + sad2 + sad3 + sad4 + sad5 + sad6 + sad7;

                    // fractional values in rows 1 and 3
                    let (a,b,c,d) = (sad.extract(0),sad.extract(1),sad.extract(2),sad.extract(3));
                    accumulator += a + c + (b + d) >> 2;

                    // 16 pixels per row * 8 rows / 4 lanes
                    let hist_idx : u64x4 = (sad >> 5) & HIST_MASK;
                    histogram[hist_idx.extract(0) as usize] += 1;
                    histogram[hist_idx.extract(1) as usize >> 2] += 1;
                    histogram[hist_idx.extract(2) as usize] += 1;
                    histogram[hist_idx.extract(3) as usize >> 2] += 1;

                    area_sum += min(32, a) + min(32, b) + min(32, c) + min(32, d);
            });
        },
        #[cfg(all(target_feature = "ssse3", not(target_feature = "avx2")))]
        Pixel::YUV420P10LE | Pixel::YUV444P10LE => {
            let idx_a = - dims_a.min_x() * 2 - dims_a.min_y() * stride as isize;
            let idx_b = - dims_b.min_x() * 2 - dims_b.min_y() * stride as isize;

            for row in (intersection.min_y() .. intersection.max_y()-7).step_by(8) {
                let idx_a = idx_a + row  * stride as isize;
                let idx_b = idx_b + row  * stride as isize;

                for col in (intersection.min_x() .. intersection.max_x() - 15).step_by(16) {
                    let idx_a = (idx_a + col * 2) as usize;
                    let idx_b = (idx_b + col * 2) as usize;

                    let sad0 = sad16!(luma_a,luma_b,idx_a, idx_b, stride, 0);
                    let sad1 = sad16!(luma_a,luma_b,idx_a, idx_b, stride, 1);
                    let sad2 = sad16!(luma_a,luma_b,idx_a, idx_b, stride, 2);
                    let sad3 = sad16!(luma_a,luma_b,idx_a, idx_b, stride, 3);
                    let sad4 = sad16!(luma_a,luma_b,idx_a, idx_b, stride, 4);
                    let sad5 = sad16!(luma_a,luma_b,idx_a, idx_b, stride, 5);
                    let sad6 = sad16!(luma_a,luma_b,idx_a, idx_b, stride, 6);
                    let sad7 = sad16!(luma_a,luma_b,idx_a, idx_b, stride, 7);

                    let sad = sad0 + sad1 + sad2 + sad3 + sad4 + sad5 + sad6 + sad7;
                    // load 16 x 2 bytes -> 16 pixels; 8 rows; 2 simd result lanes. 16 * 8 /2
                    let a = sad.extract(0);
                    let b = sad.extract(1);
                    accumulator += a + b;
                    area_sum += min(8*8, a) + min(8*8, b);
                    // max error is 256 * 8 * 8
                    histogram[((a / (8*8)) & 0xff) as usize] += 1;
                    histogram[((b / (8*8)) & 0xff) as usize] += 1;
                }
            }
        }
        Pixel::YUV420P10LE | Pixel::YUV420P => unimplemented!("did you forget to compile with ssse3 or avx2?"),
        fmt @ _ => unimplemented!("for pixel format {:?} ",fmt)
    };

    let pixels = ((intersection.size.width & !wmask) * (intersection.size.height &!hmask)) >> subsample ;

    Estimate {
        error_sum: accumulator as u64,
        error_area: area_sum as u64,
        x: offset_x,
        y: offset_y,
        area: pixels as u32,
        histogram
    }

}

#[cfg(test)]
mod test {
    #[test]
    #[cfg(target_feature = "avx2")]
    fn sad16avx() {

        let mut a = [0 ; 1024];
        let mut b = [0 ; 1024];

        assert!(sad16avx!(&a[..],&b[..],0,0,0,0,0).extract(0) == 0);

        a[512] = 1;

        assert!(sad16avx!(&a[..],&b[..],511,0,10,0,1).extract(0) == 0);
        assert!(sad16avx!(&a[..],&b[..],0,0,512,1,0).extract(0) == 1);
        assert!(sad16avx!(&a[..],&b[..],0,0,511,0,1).extract(0) == 1);
        assert!(sad16avx!(&a[..],&b[..],512,0,17,0,1).extract(0) == 1);
        assert!(sad16avx!(&a[..],&b[..],512,0,17,0,0).extract(0) == 1);



    }

}