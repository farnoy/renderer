extern crate bindgen;

use rayon::prelude::*;
use std::{
    env, fs,
    path::{Path, PathBuf},
    process::Command,
};

fn main() {
    let jobserver =
        unsafe { jobserver::Client::from_env().expect("failed to obtain jobserver from cargo") };
    let src = env::var("CARGO_MANIFEST_DIR").unwrap();
    let src = Path::new(&src);
    let dest = env::var("OUT_DIR").unwrap();
    let dest = Path::new(&dest);
    let shaders = &[
        "debug_aabb.frag",
        "debug_aabb.vert",
        "depth_prepass.vert",
        "generate_work.comp",
        "gltf_mesh.frag",
        "gltf_mesh.vert",
        "gui.frag",
        "gui.vert",
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

    println!("cargo:rustc-link-lib=amd_alloc");
    if cfg!(unix) {
        println!("cargo:rustc-link-lib=dylib=stdc++");
    }
    println!("cargo:rustc-link-search=native={}", src.to_str().unwrap());
    println!("cargo:rerun-if-changed=wrapper.h");

    let bindings = bindgen::Builder::default()
        .header("wrapper.h")
        .derive_debug(true)
        .derive_default(true)
        .generate_comments(false)
        .layout_tests(false)
        .clang_arg("-x")
        .clang_arg("c++")
        .clang_arg("-std=c++14")
        .clang_arg(if cfg!(windows) {
            "-IC:\\VulkanSDK\\1.2.135.0\\Include"
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
        .whitelist_function("vmaAllocateMemoryPages")
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
        .blacklist_type("VkInstance")
        .blacklist_type("VkFlags")
        .blacklist_type("VkResult")
        .blacklist_type("VkStructureType")
        .blacklist_type("VkDeviceMemory")
        .blacklist_type("VkDevice")
        .blacklist_type("VkDeviceSize")
        .blacklist_type("VkMemoryRequirements")
        .blacklist_type("VkMemoryRequirements2")
        .new_type_alias("VkMemoryRequirements")
        .blacklist_type("VkPhysicalDevice")
        .generate()
        .expect("Unable to generate bindings");

    let out_path = PathBuf::from(env::var("OUT_DIR").unwrap());
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
