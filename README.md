# stitch-animation

Linear panning detection and scene extraction using motion vectors provided by video codecs.

For the moment the tool only extracts image sequences and does not align or blend them.
Microsoft ICE, Gimp, Photoshop or similar are recommended to merge them.

![video to frame sequence to composite](doc/video,%20frames,%20composite.png)

## Dependencies

* ffmpeg 1.3.x
* rustc + cargo (build)
 
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

`stitch-animation path/video name.mkv`

saves image sequences in the current working directory matching the pattern  `./video name.seq/*.png`



## Current limitations

* requires libavcodec support for exporting motion vectors (e.g. the mpeg family)
* I-frames in the middle of a scene currently 
* any kind of non-linear motion is not actively supported. they just may happen to work anyway. this includes
  zoom, rotations, stops during the pan, sharp turns in the path taken by the camera.
* can not be considered fully automated until it spits out decent composites with minimal artifacts
  

## TODO

* fast planar stitching (opencv? hugin?) 
  * remove foreground objects and logos (trimmed mean over multiple layers, gradient blending, smarter frame selection)
* multi-threaded PNG encoding
* estimate affine transforms for global motion instead of dominant vector
* detect still frames by diffing random samples
* use ffmpeg's `mestimate` filter to construct vectors if none are present (would help with I-frames too)
 