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

    cc::Build::new()
        .cpp(true)
        .flag_if_supported("--std=c++14")
        .flag_if_supported("-w") // don't care for this noise
        .flag_if_supported("/std:c++14")
        .file("amd_alloc.cc")
        .includes(if cfg!(windows) {
            Some("C:\\VulkanSDK\\1.2.170.0\\Include")
        } else {
            None
        })
        .compile("amd_alloc");

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
            "-IC:\\VulkanSDK\\1.2.170.0\\Include"
        } else {
            ""
        })
        .allowlist_type("VmaAllocatorCreateInfo")
        .allowlist_type("VmaAllocatorCreateFlags")
        .allowlist_type("VmaAllocatorCreateFlagBits")
        .allowlist_type("VmaAllocation")
        .allowlist_type("VmaAllocationCreateFlagBits")
        .allowlist_type("VmaAllocationInfo")
        .allowlist_type("VmaVulkanFunctions")
        .bitfield_enum("VmaAllocatorCreateFlagBits")
        .bitfield_enum("VmaAllocationCreateFlagBits")
        .rustified_enum("VmaMemoryUsage")
        .allowlist_function("vmaCalculateStats")
        .allowlist_function("vmaAllocateMemoryPages")
        .allowlist_function("vmaCreateAllocator")
        .allowlist_function("vmaGetAllocationInfo")
        .allowlist_function("vmaDestroyAllocator")
        .allowlist_function("vmaSetCurrentFrameIndex")
        .allowlist_function("vmaMapMemory")
        .allowlist_function("vmaFlushAllocation")
        .allowlist_function("vmaUnmapMemory")
        .allowlist_function("vmaCreateBuffer")
        .allowlist_function("vmaDestroyBuffer")
        .allowlist_function("vmaCreateImage")
        .allowlist_function("vmaDestroyImage")
        .blocklist_type("VmaAllocator")
        .blocklist_type("VmaAllocation")
        .blocklist_type("VmaAllocationCreateInfo")
        .blocklist_type("VmaAllocationInfo")
        .blocklist_type("VkBuffer")
        .blocklist_type("VkBufferCreateInfo")
        .blocklist_type("VkImage")
        .blocklist_type("VkImageCreateInfo")
        .blocklist_type("VkInstance")
        .blocklist_type("VkFlags")
        .blocklist_type("VkResult")
        .blocklist_type("VkStructureType")
        .blocklist_type("VkDeviceMemory")
        .blocklist_type("VkDevice")
        .blocklist_type("VkDeviceSize")
        .blocklist_type("VkMemoryRequirements")
        .blocklist_type("VkMemoryRequirements2")
        .new_type_alias("VkMemoryRequirements")
        .blocklist_type("VkPhysicalDevice")
        .generate()
        .expect("Unable to generate bindings");

    let out_dir = env::var("OUT_DIR").unwrap();
    let out_path = Path::new(&out_dir);
    bindings
        .write_to_file(out_path.join("bindings.rs"))
        .expect("Couldn't write bindings!");
}
