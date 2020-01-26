extern crate bindgen;

use std::env;
use std::path::{Path, PathBuf};
use std::process::Command;

fn main() {
    let src = env::var("CARGO_MANIFEST_DIR").unwrap();
    let dest = env::var("OUT_DIR").unwrap();
    let dest = Path::new(&dest);
    let shaders = [
        "debug_aabb.frag",
        "debug_aabb.vert",
        "depth_prepass.vert",
        "generate_work.comp",
        "gltf_mesh.frag",
        "gltf_mesh.vert",
        "gui.frag",
        "gui.vert",
    ];
    for shader in shaders.iter() {
        println!("cargo:rerun-if-changed=src/shaders/{}", shader);
        let output_path = dest.join(format!("{}.spv", shader));
        let result = Command::new("glslangValidator")
            .args(&[
                "-C",
                "-V",
                "-g",
                "--target-env",
                "vulkan1.1",
                "-o",
                output_path.to_str().unwrap(),
                format!("{}/src/shaders/{}", src, shader).as_str(),
            ])
            .spawn()
            .unwrap()
            .wait()
            .unwrap()
            .success();
        assert!(result, "failed to compile shader");
    }

    println!("cargo:rustc-link-lib=amd_alloc");
    if cfg!(unix) {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
    println!("cargo:rustc-link-search=native={}", src);
    println!("cargo:rerun-if-changed=wrapper.h");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .derive_debug(true)
        .derive_default(true)
        .generate_comments(false)
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++14")
        .clang_arg(if cfg!(windows) {
            "-IC:\\VulkanSDK\\1.1.130.0\\Include"
        } else {
            ""
        })
        .whitelist_type("VmaAllocatorCreateInfo")
        .whitelist_type("VmaAllocatorCreateFlags")
        .whitelist_type("VmaAllocatorCreateFlagBits")
        .whitelist_type("VmaAllocation")
        .whitelist_type("VmaAllocationCreateFlagBits")
        .whitelist_type("VmaAllocationInfo")
        .whitelist_type("VmaVulkanFunctions")
        .bitfield_enum("VmaAllocatorCreateFlagBits")
        .bitfield_enum("VmaAllocationCreateFlagBits")
        .rustified_enum("VmaMemoryUsage")
        .whitelist_function("vmaCalculateStats")
        .whitelist_function("vmaCreateAllocator")
        .whitelist_function("vmaDestroyAllocator")
        .whitelist_function("vmaSetCurrentFrameIndex")
        .whitelist_function("vmaMapMemory")
        .whitelist_function("vmaFlushAllocation")
        .whitelist_function("vmaUnmapMemory")
        .whitelist_function("vmaCreateBuffer")
        .whitelist_function("vmaDestroyBuffer")
        .whitelist_function("vmaCreateImage")
        .whitelist_function("vmaDestroyImage")
        .blacklist_type("VmaAllocator")
        .blacklist_type("VmaAllocation")
        .blacklist_type("VmaAllocationCreateInfo")
        .blacklist_type("VmaAllocationInfo")
        .blacklist_type("VkBuffer")
        .blacklist_type("VkBufferCreateInfo")
        .blacklist_type("VkImage")
        .blacklist_type("VkImageCreateInfo")
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
