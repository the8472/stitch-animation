
use simd;
use std::fmt::Result;
use std::slice;
use ffmpeg::frame::side_data::Type;
use ffmpeg::frame::Video;
use ffmpeg::ffi::AVMotionVector;
use ffmpeg::util::format::pixel::Pixel;
use std::collections::HashMap;
use std::fmt::*;
use std::mem::size_of;
use itertools::Itertools;

pub(crate) trait ToMotionVectors {
    fn motion_vecs(&self) -> Option<&[AVMotionVector]>;

    fn most_common_vectors(&self) -> Vec<(isize, isize)> {
        match self.motion_vecs() {
            Some(vecs) => {
                let mut bins = HashMap::new();
                for vec in vecs {
                    let mut xy = (vec.motion_x as isize / vec.motion_scale as isize, vec.motion_y as isize / vec.motion_scale as isize);
                    if vec.source > 0 {
                        xy.0 = -xy.0;
                        xy.1 = -xy.1;
                    }
                    *bins.entry(xy).or_insert(0) += 1;
                }

                let sorted = bins.into_iter().sorted_by(|a,b| a.cmp(b));
                sorted.into_iter().rev().take(10).map(|(k,_)| k).collect()
            }
            None => vec![]
        }
    }
}

impl ToMotionVectors for Video {
    fn motion_vecs(&self) -> Option<&[AVMotionVector]> {
        let side_data = self.side_data(Type::MotionVectors);
        match side_data {
            Some(raw) => {
                let raw = raw.data();
                let ptr = raw.as_ptr() as *const AVMotionVector ;
                let len = raw.len() / size_of::<AVMotionVector>();
                Some(unsafe { slice::from_raw_parts(ptr, len) })
            }
            None => None
        }
    }
}

#[derive(Copy,Clone)]
pub(crate) struct MVec {
    angle: i16,
    len: i16,
    forward: usize,
    backward: usize,
    intra: usize
}

impl MVec {
    pub fn new() -> Self {
        MVec { angle: 0, len: 0, forward: 0, backward: 0, intra: 0 }
    }

    pub fn forward(mut self, forward: usize) -> Self {
        self.forward = forward;
        self
    }

    pub fn from_vector(mut self, x: isize, y: isize) -> Self {
        let (x,y) = (x as f32, y as f32);
        let mut angle = (y.atan2(x).to_degrees() / 22.5).round() * 22.5;
        angle += 180.0; // norm to 0-360
        angle %= 360.0;

        let len = x.hypot(y);

        self.angle = angle as i16;
        self.len = len.log2().ceil().exp2() as i16;

        self
    }

    pub fn cnt(&self) -> usize {
        self.forward + self.backward + self.intra
    }
}

impl Debug for MVec {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "({} {} | ->{} {}<- ~{})", self.angle, self.len, self.forward, self.backward, self.intra)
    }
}

#[derive(Clone)]
pub(crate) struct MVInfo {
    swarms: Vec<MVec>
}

impl MVInfo {
    pub fn new() -> Self {
        MVInfo { swarms: vec![MVec::new()]}
    }

    pub fn populate(&mut self, prev: Option<&Video>, cur: &Video, nxt: Option<&Video>) {

        if let Some(mvs) = cur.motion_vecs(){
            assert!(cur.width() % 16 == 0, "only supporting horizontal resolutions that are a multiple of 16");

            let bpp = match cur.format() {
                Pixel::YUV420P => 1,
                Pixel::YUV420P10LE => 2,
                _ => unimplemented!("pixel format not implemented")
            };

            let frame_w = cur.width() as usize;
            let frame_h = cur.height() as usize;
            assert!(cur.stride(0) == frame_w * bpp);

            let mut bins = HashMap::new();

            'vloop: for v in mvs {
                // TODO: letterbox detection


                // cull zero-motion MVs that don't actually match the previous/next frame they say they come from
                // this is not accurate due to ffmpeg motion vectors only telling us the temporal direction
                // not the precise frame index. but it removes some stuff that's actually intra-prediction
                // ... probably.
                if v.motion_x == 0 && v.motion_y == 0 && v.w == 16 && v.h == 16 {
                    let other = if v.source < 0 {
                        prev
                    } else {
                        nxt
                    };

                    if let Some(other) = other {

                        assert!(cur.data(0).len() >= (cur.width() * cur.height()) as usize);

                        let luma_self = cur.data(0);
                        let luma_other = other.data(0);


                        let offset_x = v.dst_x as usize - (v.w as usize) / 2;
                        let offset_y = v.dst_y as usize - (v.h as usize) / 2;

                        use std::cmp::{min,max};


                        let start = offset_y * frame_w + offset_x;
                        let end = min(start + frame_w * 16, frame_h);

                        let mut error_sum : u16 = 0;
                        let bsize = 16 * (min(frame_h,offset_y + 16) - offset_y) ;
                        let max_error = (bsize * 3) as u16;

                        // TODO: conditional compilation
                        use simd::u8x16;
                        use simd::x86::sse2::Sse2U8x16;

                        match cur.format() {
                            Pixel::YUV420P => {
                                for stride in (start .. end).step_by(frame_w) {

                                    let a = simd::u8x16::load(luma_self, stride);
                                    let b = simd::u8x16::load(luma_other, stride);

                                    let diffs = a.sad(b);

                                    error_sum += (diffs.extract(0) + diffs.extract(1)) as u16;

                                    if error_sum > max_error {
                                        continue 'vloop;
                                    }
                                }
                            },
                            Pixel::YUV420P10LE => {
                                // little-endian 2bytes per value
                                // 1byte fractional value, 1 byte integer
                                // just mask away the fraction for the SAD calculation
                                //let upper_mask = simd::u8x16::new(0,0xFF,0,0xFF,0,0xFF,0,0xFF,0,0xFF,0,0xFF,0,0xFF,0,0xFF);
                                const UPPER_MASK : u8x16 = u8x16::new(0xFF,0,0xFF,0,0xFF,0,0xFF,0,0xFF,0,0xFF,0,0xFF,0,0xFF, 0);

                                for stride in (start * 2 .. end * 2).step_by(frame_w * 2) {

                                    let a1 = u8x16::load(luma_self, stride) & UPPER_MASK;
                                    let a2 = u8x16::load(luma_self, stride + 16) & UPPER_MASK;
                                    let b1 = u8x16::load(luma_other, stride) & UPPER_MASK;
                                    let b2 = u8x16::load(luma_other, stride + 16) & UPPER_MASK;

                                    let diffs1 = a1.sad(b1);
                                    let diffs2 = a2.sad(b2);

                                    error_sum += diffs1.extract(0) as u16;
                                    error_sum += diffs1.extract(1) as u16;
                                    error_sum += diffs2.extract(0) as u16;
                                    error_sum += diffs2.extract(1) as u16;

                                    if error_sum > max_error {
                                        continue 'vloop;
                                    }
                                }
                            },
                            _ => unimplemented!()
                        }


                    }

                }

                let y = v.motion_y as f32 / v.motion_scale as f32;
                let x = v.motion_x as f32 / v.motion_scale as f32;



                let mut angle = (y.atan2(x).to_degrees() / 22.5).round() * 22.5;
                angle += 180.0; // norm to 0-360
                if v.source > 0 { // invert if motion from future
                    angle += 180.0;
                }
                angle %= 360.0;

                let len = x.hypot(y);



                let len = len.log2().ceil().exp2();
                let bin = bins.entry((angle as i16, len as i16)).or_insert((0,0,0));

                let area = v.w as usize * v.h as usize;

                match v.source.signum() {
                    -1 => (*bin).0 += area,
                    0 => (*bin).2 += area,
                    1 => (*bin).1 += area,
                    _ => unreachable!()
                }



            }

            let mut sorted = bins.into_iter().map(|(k, v)| MVec{angle: k.0, len: k.1, forward: v.0, backward: v.1, intra: v.2}).collect::<Vec<_>>();
            sorted.sort_by_key(|e| e.cnt());
            sorted.reverse();

            self.swarms = sorted;
        }
    }

    pub fn add(&mut self, v: MVec) {
        let mut bins = self.bins();

        {
            let bin = bins.entry((v.angle, v.len)).or_insert(MVec {angle: v.angle, len: v.len, backward: 0, forward: 0, intra: 0});
            (*bin).forward += v.forward;
            (*bin).backward += v.backward;
            (*bin).intra += v.intra;
        }

        let mut sorted = bins.into_iter().map(|(_, v)| v).collect::<Vec<_>>();
        sorted.sort_by_key(|e| e.cnt());
        sorted.reverse();
        self.swarms = sorted;
    }

    fn bins(&self) -> HashMap<(i16,i16),MVec> {
        let mut bins = HashMap::with_capacity(self.swarms.len());

        for v in self.swarms.iter() {
            bins.insert((v.angle,v.len), *v);
        }

        bins
    }

    pub fn transplant_from(&mut self, prev: Option<&MVInfo>, nxt: Option<&MVInfo>) {

        let mut bins = self.bins();

        if let Some(prev) = prev {
            for v in prev.swarms.iter().filter(|v| v.backward > 0) {
                let bin = bins.entry((v.angle, v.len)).or_insert(MVec {angle: v.angle, len: v.len, backward: 0, forward: 0, intra: 0});
                (*bin).forward += v.backward;
            }
        }

        if let Some(nxt) = nxt {
            for v in nxt.swarms.iter().filter(|v| v.forward> 0) {
                let bin = bins.entry((v.angle, v.len)).or_insert(MVec {angle: v.angle, len: v.len, backward: 0, forward: 0, intra: 0});
                (*bin).backward += v.forward;
            }
        }

        let mut sorted = bins.into_iter().map(|(_, v)| v).collect::<Vec<_>>();
        sorted.sort_by_key(|e| e.cnt());
        sorted.reverse();
        self.swarms = sorted;

    }

    pub fn past(&self) -> usize {
        self.swarms.iter().map(|e| e.forward).sum::<usize>()
    }

    pub fn future(&self) -> usize {
        self.swarms.iter().map(|e| e.backward).sum::<usize>()
    }

    pub fn intra(&self) -> usize {
        self.swarms.iter().map(|e| e.intra).sum::<usize>()
    }

    pub fn forward_still_blocks(&self) -> usize {
        self.swarms.iter().filter(|e| e.len == 0).map(|m| m.forward).sum::<usize>()
    }

    pub fn forward_dominant_angle(&self) -> (i16, usize) {
        let mut bins = HashMap::new();

        for mv in &self.swarms {
            if mv.len == 0 {
                continue;
            }
            let entry = bins.entry(mv.angle).or_insert(0);
            (*entry) += mv.forward;
        }

        match bins.iter().max_by_key(|&(k,v)| *v) {
            Some((angle, count)) => (*angle, *count),
            None => (0,0)
        }
    }

    /// can actually return more than the total amount of pixels if the same location is predicted by
    /// multiple vectors
    pub fn pred(&self) -> usize {
        self.swarms.iter().map(MVec::cnt).sum::<usize>()
    }



}

impl Debug for MVInfo {
    fn fmt(&self, f: &mut Formatter) -> Result {
        write!(f, "->{} {}<- ~{} | {:?}", self.past(), self.future(), self.intra(), self.swarms)
    }
}