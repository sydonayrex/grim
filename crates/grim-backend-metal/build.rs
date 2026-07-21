use std::env;
use std::path::PathBuf;
use std::process::Command;

fn main() {
    println!("cargo::rustc-check-cfg=cfg(embed_metallib)");
    println!("cargo:rerun-if-changed=src/kernels.msl");

    let target_os = env::var("CARGO_CFG_TARGET_OS").unwrap_or_default();
    if target_os == "macos" {
        let out_dir = PathBuf::from(env::var("OUT_DIR").unwrap());
        let msl_path = PathBuf::from("src/kernels.msl");
        let air_path = out_dir.join("kernels.air");
        let metallib_path = out_dir.join("kernels.metallib");

        // Try compiling to AIR
        let status_air = Command::new("xcrun")
            .args(&[
                "-sdk",
                "macosx",
                "metal",
                "-c",
                msl_path.to_str().unwrap(),
                "-o",
                air_path.to_str().unwrap(),
            ])
            .status();

        if let Ok(status) = status_air {
            if status.success() {
                // Try compiling to metallib
                let status_lib = Command::new("xcrun")
                    .args(&[
                        "-sdk",
                        "macosx",
                        "metallib",
                        air_path.to_str().unwrap(),
                        "-o",
                        metallib_path.to_str().unwrap(),
                    ])
                    .status();

                if let Ok(s_lib) = status_lib {
                    if s_lib.success() {
                        println!("cargo:rustc-cfg=embed_metallib");
                    }
                }
            }
        }
    }
}
