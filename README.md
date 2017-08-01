# stitch-animation

Linear panning detection, scene extraction and image compositing.

Supports multi-threading, depending on CPU capacity, video resolution and passed arguments
it should run 2-10x faster than normal playback speed.

For the moment the goal is throughput, quality of the composites comes second.
I.e. manual work is likely to outperform this tool on individual scenes.

![video to frame sequence to composite](doc/video,%20frames,%20composite.png)

## Dependencies

*Note:* The ffmpeg and ffmpeg-sys rust crates are currently being updated and require up-to-date FFmpeg libraries.
Building works out of the box on Arch Linux. Ubuntu's FFmpeg packages on the other hand are too old. 

*runtime:*

* ffmpeg >= 3.2 or libav* equivalents
* x86-64 CPU with SSSE3 instruction set. motion detection loops are implemented using x86-specific SIMD instructions 
  
*build time:*

* rustc nightly
* cargo
* gcc/clang
* ffmpeg >= 3.2 or libav* equivalents + headers

For build problems with ffmpeg-sys take a look at their [travis configuration](https://github.com/meh/rust-ffmpeg-sys/blob/master/.travis.yml)
and [linux build script](https://github.com/meh/rust-ffmpeg-sys/blob/master/.travis/install_linux.sh)
 
## Build & Install

```sh
git clone --recursive https://github.com/the8472/stitch-animation.git
cd stitch-animation
# minimum CPU requirements
RUSTFLAGS="-C target-feature=+sse2,+ssse3" cargo install
# optimized for current system
RUSTFLAGS="-C target-cpu=native" cargo install
```

## run

`stitch-animation path/video-name.mkv`

saves composites for found sequences in the current working directory matching the pattern  `./video-name.seq/*.png`

```
$ stitch-animation --help
animation linear panning detection, scene extraction and stitching for videos 0.1.1

USAGE:
    stitch-animation [FLAGS] [OPTIONS] <inputs>...

FLAGS:
    -h, --help        Prints help information
        --nostitch    do not create composite images
    -V, --version     Prints version information

OPTIONS:
    -n <N>                   process at most N frames, after seeking
    -p, --pictures <pics>    save individual frames [default: null]  [values: png, jpg]
    -s <seek_to>             seek to frame number [currently inaccurate, specify a lower number than the desired actual frame]

ARGS:
    <inputs>...    videos files to process. specify '-' to read a newline-separated list from stdin.
                   Example: find /media/videos -type f -name '*.mkv' | stitch-animation -
```

## Current limitations

* x86 only
* any kind of non-linear motion is not actively supported. they just may happen to work anyway.
  this includes zoom, rotations, perspective distortions.
  Long stops during a pan may also lead to disjoint sequences.
  Use image extraction option and an external compositor such as Microsoft's ICE to handle these cases
* letterbox black bars can confuse the motion detector


  

## TODO/Ideas

* improve compositing
  * remove foreground objects and logos (trimmed mean over multiple layers, gradient blending, smarter frame selection)
  * analyze the areas moving into and out of a frame, i.e. how clean the pan is
* zoom and rotation support
  * infer affine transforms from vectors? or search separately for those operations?
* performance improvements
  * skip some frames containing redundant information on perfectly horizontal or vertical pans to reduce encoding load
* skip ranges if they're already decoded on disk

