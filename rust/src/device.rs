//! wgpu adapter/device acquisition + capability probing.

use std::sync::Arc;

/// A wgpu device handle plus the adapter facts the engine reports.
#[derive(Clone)]
pub struct GpuContext {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub info: GpuInfo,
}

#[derive(Clone, Debug)]
pub struct GpuInfo {
    pub name: String,
    pub backend: String,
    pub device_type: String,
    pub max_buffer_size: u64,
    pub max_storage_buffer_binding_size: u64,
}

/// Acquire a high-performance adapter and a device with the largest buffer
/// limits the adapter offers; Gemma-4 weight tensors are large (the per-layer
/// embedding table alone is ~560 MB packed).
pub fn request_device() -> Result<GpuContext, String> {
    let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle_from_env());
    let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
        power_preference: wgpu::PowerPreference::HighPerformance,
        force_fallback_adapter: false,
        compatible_surface: None,
        apply_limit_buckets: false,
    }))
    .map_err(|e| format!("no suitable GPU adapter found: {e}"))?;

    let al = adapter.limits();
    let want_bytes = al.max_storage_buffer_binding_size.min(al.max_buffer_size);
    let mut limits = wgpu::Limits::defaults();
    limits.max_storage_buffer_binding_size = want_bytes;
    limits.max_buffer_size = want_bytes;
    limits.max_compute_workgroup_storage_size = al.max_compute_workgroup_storage_size;
    limits.max_compute_invocations_per_workgroup = al.max_compute_invocations_per_workgroup;
    limits.max_compute_workgroups_per_dimension = al.max_compute_workgroups_per_dimension;
    limits.max_uniform_buffer_binding_size = al.max_uniform_buffer_binding_size;
    // Never request more than the adapter has for the remaining fields.
    let limits = limits.using_resolution(al.clone());

    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        label: Some("gemma-wgpu"),
        required_features: wgpu::Features::empty(),
        required_limits: limits,
        experimental_features: wgpu::ExperimentalFeatures::disabled(),
        memory_hints: wgpu::MemoryHints::Performance,
        trace: wgpu::Trace::Off,
    }))
    .map_err(|e| format!("request_device failed: {e}"))?;

    let ai = adapter.get_info();
    let info = GpuInfo {
        name: ai.name.clone(),
        backend: format!("{:?}", ai.backend),
        device_type: format!("{:?}", ai.device_type),
        max_buffer_size: want_bytes,
        max_storage_buffer_binding_size: want_bytes,
    };
    Ok(GpuContext { device: Arc::new(device), queue: Arc::new(queue), info })
}
