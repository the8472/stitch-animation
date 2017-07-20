extern crate gcc;

use std::process::Command;
use std::path::Path;

fn main() {


    gcc::Config::new()
        .cpp(true)
        .file("src/opencv_stitching_detailed.cpp")
        .flag("-lopencv_core")
        .flag("-lopencv_stitching")
        .flag("-lopencv_imgproc")
        .flag("-lopencv_imgcodecs")
        .flag("-lopencv_highgui")
        .compile("libstitch.a");

    println!("cargo:rustc-link-lib=opencv_core");
    println!("cargo:rustc-link-lib=opencv_stitching");
    println!("cargo:rustc-link-lib=opencv_imgproc");
    println!("cargo:rustc-link-lib=opencv_imgcodecs");
    println!("cargo:rustc-link-lib=opencv_highgui");

    /*
    gcc::compile_library("libstitch.a", &["src/opencv_stitching_detailed.cpp"]);

    let status = Command::new("c++")
        .args(
            &[
                "-lopencv_core",
                "-lopencv_stitching",
                "-lopencv_imgproc",
                "-lopencv_imgcodecs",
                "-lopencv_highgui",
                "src/opencv_stitching_detailed.cpp",
                "-o"
            ],
        )
        .arg(Path::new(&std::env::var("OUT_DIR").expect("target dir")).join("image-stitcher"))
        .status()
        .expect("c++ compiler");

    if !status.success() {
        eprintln!("c++ compilaton failed");
        std::process::exit(1);
    }*/
}
