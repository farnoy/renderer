use std::{ffi::CStr, mem::transmute, ops::Deref, sync::Arc};

use ash::{
    self,
    extensions::{
        self,
        khr::{AccelerationStructure, Synchronization2},
    },
    vk,
};
use cache_padded::CachePadded;
use crossbeam_utils::Backoff;
use parking_lot::{Mutex, MutexGuard};
use profiling::scope;

mod alloc;
mod buffer;
mod commands;
mod descriptors;
mod double_buffered;
mod framebuffer;
mod image;
mod image_view;
mod mapping;
mod pipeline;
mod sampler;
mod shader;
mod sync;

pub(crate) use self::{
    alloc::VmaMemoryUsage,
    buffer::{Buffer, StaticBuffer},
    commands::{StrictCommandPool, StrictRecordingCommandBuffer},
    descriptors::{DescriptorPool, DescriptorSet, DescriptorSetLayout},
    double_buffered::DoubleBuffered,
    framebuffer::Framebuffer,
    image::Image,
    image_view::ImageView,
    pipeline::{Pipeline, PipelineLayout},
    sampler::Sampler,
    shader::Shader,
    sync::{Fence, Semaphore, TimelineSemaphore},
};
use super::{Instance, Surface};

type AshDevice = ash::Device;

pub(crate) struct Device {
    pub(super) device: AshDevice,
    #[allow(unused)]
    instance: Arc<Instance>,
    pub(super) physical_device: vk::PhysicalDevice,
    allocator: alloc::VmaAllocator,
    pub(crate) limits: vk::PhysicalDeviceLimits,
    pub(crate) graphics_queue_family: u32,
    pub(crate) compute_queue_family: u32,
    pub(crate) transfer_queue_family: u32,
    graphics_queue: CachePadded<Mutex<vk::Queue>>,
    // TODO: temporarily exposed
    pub(crate) compute_queues: Vec<CachePadded<Mutex<vk::Queue>>>,
    // TODO: temporarily exposed
    pub(crate) transfer_queue: Option<CachePadded<Mutex<vk::Queue>>>,
    #[cfg(feature = "crash_debugging")]
    pub(crate) buffer_marker_fn: vk::AmdBufferMarkerFn,
    #[allow(dead_code)]
    pub(crate) extended_dynamic_state_fn: vk::ExtExtendedDynamicStateFn,
    pub(crate) synchronization2: Synchronization2,
    pub(crate) acceleration_structure: AccelerationStructure,
    pub(crate) min_acceleration_structure_scratch_offset_alignment: u32,
}

impl Device {
    pub(crate) fn new(instance: &Arc<Instance>, surface: &Surface) -> Result<Device, vk::Result> {
        let Instance { ref entry, .. } = **instance;

        let pdevices = unsafe { instance.enumerate_physical_devices().expect("Physical device error") };

        let physical_device = pdevices[0];
        let queue_families = unsafe { instance.get_physical_device_queue_family_properties(physical_device) };
        let properties = unsafe { instance.get_physical_device_properties(physical_device) };

        let mut acceleration_structure_properties = vk::PhysicalDeviceAccelerationStructurePropertiesKHR::default();
        let mut properties2 =
            vk::PhysicalDeviceProperties2::builder().push_next(&mut acceleration_structure_properties);
        unsafe { instance.get_physical_device_properties2(physical_device, &mut properties2) };
        /*
        unsafe {
            for format in vk::Format::UNDEFINED.as_raw()..vk::Format::ASTC_12X12_SRGB_BLOCK.as_raw()
            {
                let format = vk::Format::from_raw(format);
                let res = instance.get_physical_device_image_format_properties(
                    physical_device,
                    format,
                    vk::ImageType::TYPE_2D,
                    vk::ImageTiling::OPTIMAL,
                    vk::ImageUsageFlags::SAMPLED,
                    vk::ImageCreateFlags::SPARSE_BINDING & vk::ImageCreateFlags::SPARSE_RESIDENCY,
                );
                if let Ok(vk::ImageFormatProperties { ref max_extent, ..}) = res {
                    dbg!(format, max_extent);
                }
            }
        }
        */
        let graphics_queue_family = {
            queue_families
                .iter()
                .enumerate()
                .filter_map(|(ix, info)| unsafe {
                    let supports_graphic_and_surface = info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                        && surface
                            .ext
                            .get_physical_device_surface_support(physical_device, ix as u32, surface.surface)
                            .unwrap();
                    if supports_graphic_and_surface {
                        Some(ix as u32)
                    } else {
                        None
                    }
                })
                .next()
                .unwrap()
        };
        let compute_queues_spec = {
            queue_families
                .iter()
                .enumerate()
                .filter_map(|(ix, info)| {
                    if info.queue_flags.contains(vk::QueueFlags::COMPUTE)
                        && !info.queue_flags.contains(vk::QueueFlags::GRAPHICS)
                    {
                        Some((ix as u32, info.queue_count))
                    } else {
                        None
                    }
                })
                .next()
        };
        let transfer_queue_family = {
            queue_families
                .iter()
                .enumerate()
                .filter_map(|(ix, info)| {
                    if info.queue_flags.contains(vk::QueueFlags::TRANSFER)
                        && !info.queue_flags.contains(vk::QueueFlags::COMPUTE)
                    {
                        Some(ix as u32)
                    } else {
                        None
                    }
                })
                .next()
        };
        let queues = match (compute_queues_spec, transfer_queue_family) {
            (Some((compute_queue_family, compute_queue_len)), Some(transfer_queue_family)) => vec![
                (graphics_queue_family, 1),
                (compute_queue_family, compute_queue_len),
                (transfer_queue_family, 1),
            ],
            (Some((compute_queue_family, compute_queue_len)), None) => {
                vec![(graphics_queue_family, 1), (compute_queue_family, compute_queue_len)]
            }
            _ => vec![(graphics_queue_family, 1)],
        };
        let device = {
            let device_extension_names_raw = vec![
                extensions::khr::Swapchain::name().as_ptr(),
                #[cfg(feature = "gpu_printf")]
                CStr::from_bytes_with_nul(b"VK_KHR_shader_non_semantic_info\0")
                    .unwrap()
                    .as_ptr(),
                #[cfg(feature = "crash_debugging")]
                vk::AmdBufferMarkerFn::name().as_ptr(),
                vk::ExtRobustness2Fn::name().as_ptr(),
                vk::ExtExtendedDynamicStateFn::name().as_ptr(),
                vk::KhrSynchronization2Fn::name().as_ptr(),
                vk::ExtSubgroupSizeControlFn::name().as_ptr(),
                vk::KhrDeferredHostOperationsFn::name().as_ptr(),
                vk::KhrAccelerationStructureFn::name().as_ptr(),
                vk::KhrRayQueryFn::name().as_ptr(),
            ];
            let features = vk::PhysicalDeviceFeatures {
                shader_clip_distance: 1,
                sampler_anisotropy: 1,
                // sparse_binding: 1,
                // sparse_residency_image2_d: 1,
                depth_bounds: 0,
                multi_draw_indirect: 1,
                vertex_pipeline_stores_and_atomics: 1,
                robust_buffer_access: 1, // TODO: disable at some point?
                fill_mode_non_solid: 1,
                draw_indirect_first_instance: 1,
                shader_storage_buffer_array_dynamic_indexing: 1,
                ..Default::default()
            };
            let mut features_dynamic_state =
                vk::PhysicalDeviceExtendedDynamicStateFeaturesEXT::builder().extended_dynamic_state(true);
            let mut features2 = vk::PhysicalDeviceFeatures2::builder().features(features);
            let mut features12 = vk::PhysicalDeviceVulkan12Features::builder()
                .descriptor_binding_partially_bound(true)
                .draw_indirect_count(true)
                .runtime_descriptor_array(true)
                .separate_depth_stencil_layouts(true)
                .shader_storage_buffer_array_non_uniform_indexing(true)
                .shader_sampled_image_array_non_uniform_indexing(true)
                .shader_storage_image_array_non_uniform_indexing(true)
                .timeline_semaphore(true)
                .imageless_framebuffer(true)
                .scalar_block_layout(true)
                .buffer_device_address(true)
                .descriptor_indexing(true)
                .descriptor_binding_sampled_image_update_after_bind(true);
            let mut features_subgroup_size =
                vk::PhysicalDeviceSubgroupSizeControlFeaturesEXT::builder().subgroup_size_control(true);
            let mut features_acceleration_structure = vk::PhysicalDeviceAccelerationStructureFeaturesKHR::builder()
                .acceleration_structure(true)
                .descriptor_binding_acceleration_structure_update_after_bind(true);
            let mut features_ray_query = vk::PhysicalDeviceRayQueryFeaturesKHR::builder().ray_query(true);
            let mut features_robustness = vk::PhysicalDeviceRobustness2FeaturesEXT::builder()
                .robust_image_access2(true)
                .robust_buffer_access2(true)
                .null_descriptor(true);
            let mut features_synchronization =
                vk::PhysicalDeviceSynchronization2FeaturesKHR::builder().synchronization2(true);
            let mut priorities = vec![];
            let queue_infos = queues
                .iter()
                .map(|&(ref family, ref len)| {
                    priorities.push(vec![1.0; *len as usize]);
                    let p = priorities.last().unwrap();
                    vk::DeviceQueueCreateInfo::builder()
                        .queue_family_index(*family)
                        .queue_priorities(p)
                        .build()
                })
                .collect::<Vec<_>>();

            let device_create_info = vk::DeviceCreateInfo::builder()
                .queue_create_infos(&queue_infos)
                .enabled_extension_names(&device_extension_names_raw)
                .push_next(&mut features_dynamic_state)
                .push_next(&mut features2)
                .push_next(&mut features12)
                .push_next(&mut features_subgroup_size)
                .push_next(&mut features_acceleration_structure)
                .push_next(&mut features_ray_query)
                .push_next(&mut features_robustness)
                .push_next(&mut features_synchronization);

            unsafe { instance.create_device(physical_device, &device_create_info, None)? }
        };

        let allocator = alloc::create(entry.vk(), &**instance, device.handle(), physical_device).unwrap();
        let graphics_queue = unsafe { device.get_device_queue(graphics_queue_family, 0) };
        let compute_queues = match compute_queues_spec {
            Some((compute_queue_family, len)) => (0..len)
                .map(|ix| unsafe { device.get_device_queue(compute_queue_family, ix) })
                .collect::<Vec<_>>(),
            None => vec![],
        };
        let transfer_queue = transfer_queue_family
            .map(|transfer_queue_family| unsafe { device.get_device_queue(transfer_queue_family, 0) });

        let compute_queue_family = compute_queues_spec.map(|a| a.0).unwrap_or(graphics_queue_family);

        let fn_ptr_loader =
            |name: &CStr| unsafe { transmute(instance.get_device_proc_addr(device.handle(), name.as_ptr())) };

        #[cfg(feature = "crash_debugging")]
        let buffer_marker_fn = vk::AmdBufferMarkerFn::load(fn_ptr_loader);

        let extended_dynamic_state_fn = vk::ExtExtendedDynamicStateFn::load(fn_ptr_loader);
        let synchronization2 = Synchronization2::new(instance, &device);
        let acceleration_structure = AccelerationStructure::new(instance, &device);

        let device = Device {
            device,
            instance: Arc::clone(instance),
            physical_device,
            allocator,
            limits: properties.limits,
            graphics_queue_family,
            compute_queue_family,
            transfer_queue_family: transfer_queue_family.unwrap_or(compute_queue_family),
            graphics_queue: CachePadded::new(Mutex::new(graphics_queue)),
            compute_queues: compute_queues
                .iter()
                .cloned()
                .map(|q| CachePadded::new(Mutex::new(q)))
                .collect(),
            transfer_queue: transfer_queue.map(|q| CachePadded::new(Mutex::new(q))),
            #[cfg(feature = "crash_debugging")]
            buffer_marker_fn,
            extended_dynamic_state_fn,
            synchronization2,
            acceleration_structure,
            min_acceleration_structure_scratch_offset_alignment: acceleration_structure_properties
                .min_acceleration_structure_scratch_offset_alignment,
        };
        device.set_object_name(graphics_queue, "Graphics Queue");
        for (ix, compute_queue) in compute_queues.iter().cloned().enumerate() {
            if compute_queue != graphics_queue {
                device.set_object_name(compute_queue, &format!("Compute Queue - {}", ix));
            }
        }
        for transfer_queue in transfer_queue.iter() {
            device.set_object_name(*transfer_queue, "Transfer Queue");
        }

        Ok(device)
    }

    pub(crate) fn graphics_queue(&self) -> &Mutex<vk::Queue> {
        &self.graphics_queue
    }

    pub(crate) fn compute_queue_balanced(&self) -> MutexGuard<vk::Queue> {
        scope!("helpers::compute_queue_balanced");
        debug_assert!(
            !self.compute_queues.is_empty(),
            "No compute queues to acquire in compute_queue_balanced"
        );

        let backoff = Backoff::new();
        loop {
            for queue in self.compute_queues.iter() {
                if let Some(lock) = queue.try_lock() {
                    return lock;
                }
            }
            backoff.spin();
        }
    }

    pub(crate) fn allocation_stats(&self) -> alloc::VmaStats {
        alloc::stats(self.allocator)
    }

    #[allow(unused)]
    pub(crate) fn allocation_info(&self, allocation: alloc::VmaAllocation) -> alloc::VmaAllocationInfo {
        alloc::get_allocation_info(self.allocator, allocation)
    }

    pub(crate) fn new_descriptor_pool(&self, max_sets: u32, pool_sizes: &[vk::DescriptorPoolSize]) -> DescriptorPool {
        DescriptorPool::new(self, max_sets, pool_sizes)
    }

    pub(crate) fn new_descriptor_set_layout(
        &self,
        create_info: &vk::DescriptorSetLayoutCreateInfo,
    ) -> DescriptorSetLayout {
        DescriptorSetLayout::new(self, create_info)
    }

    pub(crate) fn new_semaphore(&self) -> Semaphore {
        Semaphore::new(self)
    }

    pub(crate) fn new_semaphore_timeline(&self, initial_value: u64) -> TimelineSemaphore {
        TimelineSemaphore::new(self, initial_value)
    }

    pub(crate) fn new_fence(&self) -> Fence {
        Fence::new(self)
    }

    pub(crate) fn new_shader(&self, bytes: &[u8]) -> Shader {
        Shader::new(self, bytes)
    }

    pub(crate) fn new_buffer(
        &self,
        buffer_usage: vk::BufferUsageFlags,
        allocation_usage: VmaMemoryUsage,
        size: vk::DeviceSize,
    ) -> Buffer {
        Buffer::new(self, buffer_usage, allocation_usage, size)
    }

    pub(crate) fn new_buffer_aligned(
        &self,
        buffer_usage: vk::BufferUsageFlags,
        allocation_usage: VmaMemoryUsage,
        size: vk::DeviceSize,
        alignment: vk::DeviceSize,
    ) -> Buffer {
        Buffer::new_aligned(self, buffer_usage, allocation_usage, size, alignment)
    }

    pub(crate) fn new_static_buffer<T: Sized>(
        &self,
        buffer_usage: vk::BufferUsageFlags,
        allocation_usage: VmaMemoryUsage,
    ) -> StaticBuffer<T> {
        StaticBuffer::new(self, buffer_usage, allocation_usage)
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_image(
        &self,
        format: vk::Format,
        extent: vk::Extent3D,
        samples: vk::SampleCountFlags,
        tiling: vk::ImageTiling,
        initial_layout: vk::ImageLayout,
        usage: vk::ImageUsageFlags,
        allocation_usage: VmaMemoryUsage,
    ) -> Image {
        Image::new(
            self,
            format,
            extent,
            samples,
            tiling,
            initial_layout,
            usage,
            allocation_usage,
        )
    }

    #[allow(clippy::too_many_arguments)]
    pub(crate) fn new_image_exclusive(
        &self,
        format: vk::Format,
        extent: vk::Extent3D,
        samples: vk::SampleCountFlags,
        tiling: vk::ImageTiling,
        initial_layout: vk::ImageLayout,
        usage: vk::ImageUsageFlags,
        allocation_usage: VmaMemoryUsage,
    ) -> Image {
        Image::new_exclusive(
            self,
            format,
            extent,
            samples,
            tiling,
            initial_layout,
            usage,
            allocation_usage,
        )
    }

    pub(crate) fn new_image_view(&self, create_info: &vk::ImageViewCreateInfo) -> ImageView {
        ImageView::new(self, create_info)
    }

    pub(crate) fn new_sampler(&self, create_info: &vk::SamplerCreateInfo) -> Sampler {
        Sampler::new(self, create_info)
    }

    pub(crate) fn new_pipeline_layout(
        &self,
        descriptor_set_layouts: &[&DescriptorSetLayout],
        push_constant_ranges: &[vk::PushConstantRange],
    ) -> PipelineLayout {
        PipelineLayout::new(self, descriptor_set_layouts, push_constant_ranges)
    }

    pub(crate) fn new_graphics_pipeline(&self, create_info: vk::GraphicsPipelineCreateInfo) -> Pipeline {
        Pipeline::new_graphics_pipeline(self, create_info)
    }

    pub(super) fn new_compute_pipelines(
        &self,
        create_infos: &[vk::ComputePipelineCreateInfoBuilder<'_>],
    ) -> Vec<Pipeline> {
        Pipeline::new_compute_pipelines(self, create_infos)
    }

    pub(super) fn new_renderpass(&self, create_info: &vk::RenderPassCreateInfoBuilder) -> RenderPass {
        RenderPass::new(self, create_info)
    }

    pub(crate) fn vk(&self) -> &AshDevice {
        &self.device
    }

    #[cfg(feature = "vk_names")]
    pub(crate) fn set_object_name<T: vk::Handle>(&self, handle: T, name: &str) {
        use std::ffi::CString;

        let name = CString::new(name).unwrap();
        let name_info = vk::DebugUtilsObjectNameInfoEXT::builder()
            .object_type(T::TYPE)
            .object_handle(handle.as_raw())
            .object_name(&name);

        unsafe {
            self.instance
                .debug_utils()
                .debug_utils_set_object_name(self.device.handle(), &name_info)
                .expect("failed to set object name");
        };
    }

    #[cfg(not(feature = "vk_names"))]
    pub(crate) fn set_object_name<T: vk::Handle>(&self, _handle: T, _name: &str) {}
}

impl Deref for Device {
    type Target = AshDevice;

    fn deref(&self) -> &AshDevice {
        &self.device
    }
}

impl Drop for Device {
    fn drop(&mut self) {
        unsafe {
            self.device.device_wait_idle().unwrap();
            alloc::destroy(self.allocator);
            self.device.destroy_device(None);
        }
    }
}

pub(crate) struct RenderPass {
    pub(crate) handle: vk::RenderPass,
}

impl RenderPass {
    pub(super) fn new(device: &Device, create_info: &vk::RenderPassCreateInfoBuilder) -> RenderPass {
        let handle = unsafe {
            device
                .device
                .create_render_pass(create_info, None)
                .expect("Failed to create renderpass")
        };
        RenderPass { handle }
    }

    pub(crate) fn destroy(mut self, device: &Device) {
        unsafe { device.destroy_render_pass(self.handle, None) }
        self.handle = vk::RenderPass::null();
    }
}

#[cfg(debug_assertions)]
impl Drop for RenderPass {
    fn drop(&mut self) {
        debug_assert_eq!(
            self.handle,
            vk::RenderPass::null(),
            "RenderPass not destroyed before Drop"
        );
    }
}
