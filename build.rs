extern crate bindgen;

use std::process::Command;
use std::env;
use std::path::{Path, PathBuf};

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

    println!("cargo:rustc-link-lib=vulkan");
    println!("cargo:rustc-link-lib=stdc++");
    println!("cargo:rustc-link-lib=static=amd_alloc");
    println!("cargo:rustc-link-search=native={}", src);
    println!("cargo:rerun-if-changed=wrapper.h");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++14")
        .whitelist_type("VmaAllocatorCreateInfo")
        .whitelist_type("VmaAllocatorCreateFlags")
        .whitelist_type("VmaAllocatorCreateFlagBits")
        .bitfield_enum("VmaAllocatorCreateFlagBits")
        .whitelist_function("vmaCreateAllocator")
        .whitelist_function("vmaDestroyAllocator")
        .whitelist_function("vmaSetCurrentFrameIndex")
        .blacklist_type("VmaAllocator")
        .blacklist_type("VkBuffer")
        .blacklist_type("VkFlags")
        .blacklist_type("VkResult")
        .blacklist_type("VkStructureType")
        .blacklist_type("VkDeviceMemory")
        .blacklist_type("VkDevice")
        .blacklist_type("VkPhysicalDevice")
        .blacklist_type("VkImage")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
