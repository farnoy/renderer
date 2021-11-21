use std::{env, path::Path};

fn main() {
    let sdk_path = env::var("VULKAN_SDK").map(|p| format!("{}\\Include", p));

    cc::Build::new()
        .cpp(true)
        .flag_if_supported("--std=c++14")
        .flag_if_supported("-w") // don't care for this noise
        .flag_if_supported("/std:c++14")
        .file("amd_alloc.cc")
        .includes(if cfg!(windows) {
            sdk_path.as_ref().map(|s| s.as_str()).ok()
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
            sdk_path.as_ref().map(|s| format!("-I{}", s)).unwrap_or("".to_string())
        } else {
            "".to_string()
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
        .allowlist_function("vmaCreateBufferWithAlignment")
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
