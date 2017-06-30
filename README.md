# stitch-animation

Linear panning detection and scene extraction using motion vectors provided by video codecs.

For the moment the tool only extracts image sequences and does not align or blend them.
Microsoft ICE, Gimp, Photoshop or similar are recommended to merge them.

The extraction is multi-threaded, depending on CPU performance and video resolution,
it should run 2-10x faster than realtime. 

![video to frame sequence to composite](doc/video,%20frames,%20composite.png)

## Dependencies

*Note:* The ffmpeg and ffmpeg-sys rust crates are currently being updated and require current FFmpeg libraries.
Building works out of the box on Arch Linux. Ubuntu's FFmpeg packages on the other hand are too old. 

*runtime:* ffmpeg >= 3.2 or libav* equivalents
  
*build time:*

* rustc nightly
* cargo
* gcc/clang
* ffmpeg >= 3.2 or libav* equivalents + headers

For build problems with ffmpeg-sys take a look at their [travis configuration](https://github.com/meh/rust-ffmpeg-sys/blob/master/.travis.yml)
and [linux build script](https://github.com/meh/rust-ffmpeg-sys/blob/master/.travis/install_linux.sh)
 
## Build

```sh
git clone --recursive https://github.com/the8472/stitch-animation.git
cd stitch-animation
cargo build --release
cp target/release/stitch-animation ~/bin
## alternative, depending on your $PATH
# cargo install 
```

## run

`stitch-animation path/video-name.mkv`

saves image sequences in the current working directory matching the pattern  `./video-name.seq/*.png`



## Current limitations

* only codecs for which libavcodec exports motion vectors are supported (e.g. the mpeg family)
* I-frames in the middle of a pan can lead to a split pan or premature end of sequence
* any kind of non-linear motion is not actively supported. they just may happen to work anyway. this includes
  zoom, rotations, stops during the pan, sharp turns in the path taken by the camera.
* can not be considered fully automated until it spits out decent composites with minimal artifacts
  

## TODO

* fast planar stitching (opencv? hugin?) 
  * remove foreground objects and logos (trimmed mean over multiple layers, gradient blending, smarter frame selection)
* jpg output
* estimate affine transforms for global motion instead of dominant vector
* detect still frames in the middle of pans by diffing random samples
* use ffmpeg's `mestimate` filter to construct vectors if none are present (would help with I-frames too)
* skip some frames containing redundant information on perfectly horizontal or vertical pans to reduce encoding load
 