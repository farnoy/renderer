use std::process::Command;
use std::env;
use std::path::Path;

fn main() {
    let src = env::var("CARGO_MANIFEST_DIR").unwrap();
    let dest = env::var("OUT_DIR").unwrap();
    let dest = Path::new(&dest);
    let shaders = [
        "triangle.frag",
        "triangle.vert",
        "simple_color.frag",
        "simple_color.geom",
        "simple_color.vert",
    ];
    for shader in shaders.iter() {
        println!("cargo:rerun-if-changed=shaders/{}", shader);
        let output_path = &dest.join(format!("{}.spv", shader));
        let result = Command::new("glslangValidator")
            .arg("-C")
            .arg("-V")
            .arg("-g")
            .arg("-H")
            .arg("-o")
            .arg(output_path)
            .arg(format!("{}/shaders/{}", src, shader))
            .spawn()
            .unwrap()
            .wait()
            .unwrap()
            .success();
        assert!(result, "failed to compile shader");
    }
}
