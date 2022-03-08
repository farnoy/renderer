use std::{ffi::CStr, iter::once, mem::transmute, ops::Deref, sync::Arc};

use ash::{
    self,
    extensions::{
        self,
        khr::{AccelerationStructure, DynamicRendering, Synchronization2},
    },
    vk,
};
use hashbrown::HashMap;
use itertools::Itertools;
use parking_lot::Mutex;
use smallvec::{smallvec, SmallVec};

mod alloc;
mod buffer;
mod commands;
mod descriptors;
mod double_buffered;
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
    queues: HashMap<u32, Vec<Mutex<vk::Queue>>>,
    #[cfg(feature = "crash_debugging")]
    pub(crate) buffer_marker_fn: vk::AmdBufferMarkerFn,
    #[allow(dead_code)]
    pub(crate) extended_dynamic_state_fn: vk::ExtExtendedDynamicStateFn,
    pub(crate) synchronization2: Synchronization2,
    pub(crate) dynamic_rendering: DynamicRendering,
    pub(crate) acceleration_structure: AccelerationStructure,
    pub(crate) min_acceleration_structure_scratch_offset_alignment: u32,
}

/// If the compute queues are unavailable, they will be mapped to the graphics queues at this
/// virtual offset
const COMPUTE_QUEUE_VIRTUAL_STAGGER: usize = 3;
/// If the transfer queues are unavailable, they will be mapped to the compute queues at this
/// virtual offset
const TRANSFER_QUEUE_VIRTUAL_STAGGER: usize = 4;

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
        fn pick_family<F: Fn(u32, vk::QueueFlags) -> bool>(
            queue_families: &[vk::QueueFamilyProperties],
            f: F,
        ) -> Option<(u32, u32)> {
            queue_families
                .iter()
                .enumerate()
                .filter_map(|(ix, info)| {
                    let ix = ix as u32;
                    if f(ix, info.queue_flags) {
                        Some((ix, info.queue_count))
                    } else {
                        None
                    }
                })
                .next()
        }
        let graphics_queue_spec = pick_family(&queue_families, |ix, flags| unsafe {
            flags.contains(vk::QueueFlags::GRAPHICS)
                && surface
                    .ext
                    .get_physical_device_surface_support(physical_device, ix as u32, surface.surface)
                    .unwrap()
        })
        .expect("could not find a suitable graphics queue");
        let compute_queues_spec = pick_family(&queue_families, |_ix, flags| {
            flags.contains(vk::QueueFlags::COMPUTE)
                && !flags.contains(vk::QueueFlags::GRAPHICS)
                && cfg!(not(feature = "collapse_compute"))
        });
        let transfer_queues_spec = pick_family(&queue_families, |_ix, flags| {
            flags.contains(vk::QueueFlags::TRANSFER)
                && !flags.contains(vk::QueueFlags::COMPUTE)
                && cfg!(not(feature = "collapse_transfer"))
        });
        let queues = once(graphics_queue_spec)
            .chain(compute_queues_spec)
            .chain(transfer_queues_spec)
            .into_grouping_map()
            .max();
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
                DynamicRendering::name().as_ptr(),
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
            let mut features_dynamic_rendering =
                vk::PhysicalDeviceDynamicRenderingFeaturesKHR::builder().dynamic_rendering(true);
            let mut priorities = vec![];
            let queue_infos = queues
                .iter()
                .map(|(&family, &len)| {
                    priorities.push(vec![1.0; len as usize]);
                    let p = priorities.last().unwrap();
                    vk::DeviceQueueCreateInfo::builder()
                        .queue_family_index(family)
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
                .push_next(&mut features_synchronization)
                .push_next(&mut features_dynamic_rendering);

            unsafe { instance.create_device(physical_device, &device_create_info, instance.allocation_callbacks())? }
        };

        let allocator = alloc::create(
            entry.vk(),
            &**instance,
            instance.allocation_callbacks(),
            device.handle(),
            physical_device,
        )
        .unwrap();
        let graphics_queue_family = graphics_queue_spec.0;
        let compute_queue_family = compute_queues_spec.map(|(f, _)| f).unwrap_or(graphics_queue_family);
        let transfer_queue_family = transfer_queues_spec.map(|(f, _)| f).unwrap_or(compute_queue_family);

        let queues = {
            let mut h = HashMap::new();
            for (&family, &count) in queues.iter() {
                for ix in 0..count {
                    h.entry(family)
                        .or_insert(vec![])
                        .push(unsafe { Mutex::new(device.get_device_queue(family, ix)) });
                }
            }
            h
        };

        let fn_ptr_loader =
            |name: &CStr| unsafe { transmute(instance.get_device_proc_addr(device.handle(), name.as_ptr())) };

        #[cfg(feature = "crash_debugging")]
        let buffer_marker_fn = vk::AmdBufferMarkerFn::load(fn_ptr_loader);

        let extended_dynamic_state_fn = vk::ExtExtendedDynamicStateFn::load(fn_ptr_loader);
        let synchronization2 = Synchronization2::new(instance, &device);
        let dynamic_rendering = DynamicRendering::new(instance, &device);
        let acceleration_structure = AccelerationStructure::new(instance, &device);

        let mut device = Device {
            device,
            instance: Arc::clone(instance),
            physical_device,
            allocator,
            limits: properties.limits,
            graphics_queue_family,
            compute_queue_family,
            transfer_queue_family,
            queues,
            #[cfg(feature = "crash_debugging")]
            buffer_marker_fn,
            extended_dynamic_state_fn,
            synchronization2,
            dynamic_rendering,
            acceleration_structure,
            min_acceleration_structure_scratch_offset_alignment: acceleration_structure_properties
                .min_acceleration_structure_scratch_offset_alignment,
        };
        if cfg!(feature = "vk_names") {
            #[allow(clippy::needless_collect)]
            let queue_labels: Vec<(vk::Queue, String)> = device
                .queues
                .iter_mut()
                .flat_map(|(&family, queues)| {
                    queues.iter_mut().enumerate().map(move |(ix, q)| {
                        let mut usage: SmallVec<[&'static str; 3]> = smallvec![];
                        if device.graphics_queue_family == family {
                            usage.push("Graphics");
                        }
                        if device.compute_queue_family == family {
                            usage.push("Compute");
                        }
                        if device.transfer_queue_family == family {
                            usage.push("Transfer");
                        }
                        let usage = usage.join("|");
                        (*q.get_mut(), format!("Queue[{usage}][{ix}]"))
                    })
                })
                .collect();
            for (q, label) in queue_labels.into_iter() {
                device.set_object_name(q, &label);
            }
        }

        Ok(device)
    }

    pub(crate) fn graphics_queue(&self) -> &Mutex<vk::Queue> {
        self.graphics_queue_virtualized(0)
    }

    /// Translates the virtual queue index into the final effective index that will be used,
    /// compensated for both collapsed queue families and runtime queue counts
    pub(crate) fn graphics_queue_virtualized_to_effective_ix(&self, virt_ix: usize) -> usize {
        let queues = self.queues.get(&self.graphics_queue_family).unwrap();
        virt_ix % queues.len()
    }

    /// Translates the virtual queue index into the final effective index that will be used,
    /// compensated for both collapsed queue families and runtime queue counts
    pub(crate) fn compute_queue_virtualized_to_effective_ix(&self, virt_ix: usize) -> usize {
        if self.compute_queue_family == self.graphics_queue_family {
            self.graphics_queue_virtualized_to_effective_ix(COMPUTE_QUEUE_VIRTUAL_STAGGER + virt_ix)
        } else {
            let queues = self.queues.get(&self.compute_queue_family).unwrap();
            virt_ix % queues.len()
        }
    }

    /// Translates the virtual queue index into the final effective index that will be used,
    /// compensated for both collapsed queue families and runtime queue counts
    pub(crate) fn transfer_queue_virtualized_to_effective_ix(&self, virt_ix: usize) -> usize {
        if self.transfer_queue_family == self.compute_queue_family {
            self.compute_queue_virtualized_to_effective_ix(TRANSFER_QUEUE_VIRTUAL_STAGGER + virt_ix)
        } else {
            let queues = self.queues.get(&self.transfer_queue_family).unwrap();
            virt_ix % queues.len()
        }
    }

    pub(crate) fn graphics_queue_virtualized(&self, virt_ix: usize) -> &Mutex<vk::Queue> {
        let queues = self.queues.get(&self.graphics_queue_family).unwrap();
        &queues[virt_ix % queues.len()]
    }

    pub(crate) fn compute_queue_virtualized(&self, virt_ix: usize) -> &Mutex<vk::Queue> {
        if self.compute_queue_family == self.graphics_queue_family {
            self.graphics_queue_virtualized(COMPUTE_QUEUE_VIRTUAL_STAGGER + virt_ix)
        } else {
            let queues = self.queues.get(&self.compute_queue_family).unwrap();
            &queues[virt_ix % queues.len()]
        }
    }

    pub(crate) fn transfer_queue_virtualized(&self, virt_ix: usize) -> &Mutex<vk::Queue> {
        if self.transfer_queue_family == self.compute_queue_family {
            self.compute_queue_virtualized(TRANSFER_QUEUE_VIRTUAL_STAGGER + virt_ix)
        } else {
            let queues = self.queues.get(&self.transfer_queue_family).unwrap();
            &queues[virt_ix % queues.len()]
        }
    }

    pub(crate) fn allocation_callbacks(&self) -> Option<&vk::AllocationCallbacks> {
        self.instance.allocation_callbacks()
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
            self.device.destroy_device(self.instance.allocation_callbacks());
        }
    }
}
