extern crate libc;

use std::path::PathBuf;

extern {
    fn opencv_stitch(argc: libc::c_int, argv: *const *const libc::c_char ) -> libc::c_int;
}

pub fn stitch(files: Vec<PathBuf>, outfile: PathBuf) -> bool {
    let opt_string = "--warp affine --matcher affine --estimator affine --features surf --ba no --wave_correct no --blend_strength 3 --expos_comp no --seam gc_colorgrad --output";

    use std::ffi::CString;

    let mut opts : Vec<CString> = vec![CString::new("arg0").unwrap()];

    for s in opt_string.split(' ').map(|o| CString::new(o).unwrap()) { opts.push(s); }

    opts.push(CString::new(outfile.as_os_str().to_str().unwrap()).unwrap());

    for file_in in files {
        opts.push(CString::new(file_in.as_os_str().to_str().unwrap()).unwrap())
    }

    //println!("{:?}", opts);

    let mut argv = vec![];

    for opt in &opts {
        argv.push(opt.as_ptr());
    }

    let exit_code = unsafe {
        opencv_stitch(argv.len() as libc::c_int , argv.as_slice().as_ptr() as *const *const libc::c_char)
    };

    exit_code == 0
}
