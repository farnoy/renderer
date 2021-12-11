use std::{env, path::Path, process::Command};

use itertools::Itertools;
use rayon::prelude::*;

fn main() {
    let jobserver = unsafe { jobserver::Client::from_env().expect("failed to obtain jobserver from cargo") };
    let src = env::var("CARGO_MANIFEST_DIR").unwrap();
    let src = Path::new(&src);
    let dest = env::var("OUT_DIR").unwrap();
    let dest = Path::new(&dest);

    println!("cargo:rerun-if-changed=src");

    let mut input = match renderer_macro_lib::analyze() {
        Err(errs) => {
            println!("cargo:warning=Analysis failed: {}", errs);
            return;
        }
        Ok(data) => data,
    };

    // Compile pipelines
    input
        .pipelines
        .values()
        .flat_map(|pipe| pipe.permutations())
        .collect::<Vec<_>>()
        .into_par_iter()
        .for_each(|(pipe, stage, conditionals)| {
            let _job_slot = jobserver.acquire().expect("failed to acquire job slot");

            let file_extension = match stage.as_str() {
                "VERTEX" => "vert",
                "COMPUTE" => "comp",
                "FRAGMENT" => "frag",
                _ => unimplemented!("Unknown shader stage"),
            };
            let src_path = src.join(format!("src/shaders/{}.{}", pipe.name, file_extension));
            let conditionals = conditionals.into_iter().collect_vec();
            let output_path = dest.join(format!(
                "{}.{}.[{}].spv",
                pipe.name,
                file_extension,
                conditionals.join(",")
            ));

            let mut args = vec!["-g", "--target-env=vulkan1.2"];
            let mut dyn_args = vec![];

            for cond in conditionals {
                dyn_args.push(format!("-D{}", cond));
            }
            for dyn_arg in dyn_args.iter() {
                args.push(dyn_arg);
            }

            args.extend_from_slice(&["-o", output_path.to_str().unwrap(), src_path.to_str().unwrap()]);

            let result = Command::new("glslc")
                .args(&args)
                .spawn()
                .unwrap()
                .wait()
                .unwrap()
                .success();

            assert!(result, "failed to compile shader {:?}", &src_path);
        });

    // Analyze shader interfaces
    match renderer_macro_lib::analyze_shader_types(&input.descriptor_sets, &input.pipelines) {
        Ok(shader_information) => {
            input.shader_information = shader_information;
        }
        Err(errs) => println!("cargo:warning=Shader type validation errors:\n{}", errs.join("\n")),
    };

    if let Err(err) = renderer_macro_lib::persist(&input) {
        println!("cargo:warning={}", err);
    }
}
