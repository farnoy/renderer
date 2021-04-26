use std::{env, fs, path::Path, process::Command};

use rayon::prelude::*;

fn main() {
    let jobserver = unsafe { jobserver::Client::from_env().expect("failed to obtain jobserver from cargo") };
    let src = env::var("CARGO_MANIFEST_DIR").unwrap();
    let src = Path::new(&src);
    let dest = env::var("OUT_DIR").unwrap();
    let dest = Path::new(&dest);
    let shaders = &[
        "debug_aabb.frag",
        "debug_aabb.vert",
        "depth_pipe.vert",
        "generate_work.comp",
        "compact_draw_stream.comp",
        "gltf_mesh.frag",
        "gltf_mesh.vert",
        "imgui_pipe.frag",
        "imgui_pipe.vert",
    ];

    let latest_helper = fs::read_dir(src.join("src/shaders/helpers"))
        .unwrap()
        .map(|f| {
            let f = f.unwrap();
            println!(
                "cargo:rerun-if-changed=src/shaders/helpers/{}",
                f.file_name().to_str().unwrap()
            );
            fs::metadata(f.path()).unwrap().modified().unwrap()
        })
        .max()
        .unwrap();

    let stale_shaders = shaders
        .iter()
        .filter(|shader| {
            println!("cargo:rerun-if-changed=src/shaders/{}", shader);

            let src_path = src.join(format!("src/shaders/{}", shader));
            let output_path = dest.join(format!("{}.spv", shader));
            let src_mtime = fs::metadata(&src_path)
                .unwrap_or_else(|_| panic!("Shader missing {}", shader))
                .modified()
                .unwrap();

            fs::metadata(&output_path)
                .map(|m| m.modified().unwrap())
                .map(|dest_mtime| latest_helper > dest_mtime || src_mtime > dest_mtime)
                .unwrap_or(true)
        })
        .collect::<Vec<_>>();

    stale_shaders.par_iter().for_each(|shader| {
        let _job_slot = jobserver.acquire().expect("failed to acquire job slot");

        let src_path = src.join(format!("src/shaders/{}", shader));
        let output_path = dest.join(format!("{}.spv", shader));

        let result = Command::new("glslc")
            .args(&[
                "-g",
                "--target-env=vulkan1.2",
                "-o",
                output_path.to_str().unwrap(),
                src_path.to_str().unwrap(),
            ])
            .spawn()
            .unwrap()
            .wait()
            .unwrap()
            .success();

        assert!(result, "failed to compile shader {:?}", &src_path);
    });
}
