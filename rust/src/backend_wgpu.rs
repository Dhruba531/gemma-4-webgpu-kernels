//! wgpu backend.
//!
//! Implements exactly the same op surface as [`crate::backend_cpu::CpuBackend`],
//! so `model::Gemma4Model` runs unchanged. Tensors live in GPU storage buffers;
//! ops record compute dispatches and return a handle synchronously. Only
//! `readback()` blocks.
//!
//! f32 throughout; weights stay packed (2/4/8-bit) and are dequantized inside
//! the matmul/embed kernels.
//!
//! **Batching.** Ops record into ONE shared command encoder and ONE open
//! compute pass (WebGPU auto-synchronizes dependent dispatches within a pass),
//! submitted on `flush()` — which `readback()` triggers. Copies that used to
//! split the pass (KV appends, row slices) are dispatches of a tiny copy
//! kernel, so a whole forward pass is a single pass in a single submit.
//!
//! **Host overhead.** Every kernel has an explicit bind group layout whose
//! uniform binding uses a dynamic offset into a persistent 256-byte-slot ring,
//! so a bind group depends only on (kernel, storage buffers). Bind groups are
//! cached by those keys — activation buffers are recycled through a pool, so
//! steady-state decode creates none — and uniform bytes are staged straight
//! into the ring and uploaded with one `write_buffer` per segment per flush.
//! Nothing on the per-dispatch path allocates.
//!
//! **Memory lifecycle.** The model brackets each forward pass with
//! `begin_frame()`/`end_frame()`. Every intermediate activation `alloc()`'d
//! inside the frame is recycled at `end_frame()` except the tensors the caller
//! asks to keep (the live KV cache), so GPU memory stays bounded per step.
//! Recycled activation buffers go to a pow2-bucketed pool and are reused by
//! later frames. Weight uploads are never frame-tracked — they live for the
//! model's lifetime. `end_frame` runs right after a blocking readback, which
//! guarantees the GPU has finished every submitted command, so recycling is
//! race-free. Index uploads (token ids, positions) also come from the pool and
//! are written with `queue.write_buffer`, which is ordered before the frame's
//! submit.

use std::cell::RefCell;
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::ops::Deref;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;

use crate::backend::{AttnOpts, Backend, KvAppend, QuantWeight, TensorShape};
use crate::device::GpuContext;
use crate::kernels as K;

/// A device buffer tagged with a process-unique id (wgpu 30 exposes none), used
/// to key the bind-group cache. Ids are never reused, so a cache entry can
/// never alias a buffer that was destroyed and replaced.
pub struct GpuBuffer {
    raw: wgpu::Buffer,
    pub id: u64,
}

impl Deref for GpuBuffer {
    type Target = wgpu::Buffer;
    fn deref(&self) -> &wgpu::Buffer {
        &self.raw
    }
}

impl std::fmt::Debug for GpuBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "GpuBuffer#{} ({} B)", self.id, self.raw.size())
    }
}

pub type Buf = Arc<GpuBuffer>;

static NEXT_BUFFER_ID: AtomicU64 = AtomicU64::new(1);

fn tag(raw: wgpu::Buffer) -> Buf {
    Arc::new(GpuBuffer { raw, id: NEXT_BUFFER_ID.fetch_add(1, Ordering::Relaxed) })
}

/// Quantized-weight metadata that rides along with a packed GPU buffer.
#[derive(Debug)]
pub struct GpuQuant {
    /// f32 scales, `[N * scale_stride]`.
    pub scale_buffer: Buf,
    pub bits: u32,
    pub signed_i8: bool,
    pub group_size: usize,
    pub per_byte: usize,
}

/// A GPU tensor: an f32 storage buffer (or a packed quantized weight) + shape.
#[derive(Clone, Debug)]
pub struct GpuTensor {
    pub buffer: Buf,
    pub shape: Vec<usize>,
    /// Capacity in rows for KV-cache tensors that grow in place.
    pub cap_rows: Option<usize>,
    pub quant: Option<Arc<GpuQuant>>,
}

impl TensorShape for GpuTensor {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
}

impl GpuTensor {
    fn new(buffer: Buf, shape: &[usize]) -> Self {
        Self { buffer, shape: shape.to_vec(), cap_rows: None, quant: None }
    }
}

/// Little-endian uniform values, packed 4 bytes each (16-byte aligned struct).
#[derive(Clone, Copy)]
pub enum U {
    U(u32),
    F(f32),
}

impl U {
    fn bits(self) -> u32 {
        match self {
            U::U(u) => u,
            U::F(f) => f.to_bits(),
        }
    }
}

// --- kernel table ------------------------------------------------------------

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
#[repr(u8)]
enum Kernel {
    Elementwise,
    Embed,
    EmbedQ,
    RmsNorm,
    Matmul,
    Gemv,
    MatmulQ,
    MatmulQTiled,
    MatmulQScalar,
    Rope,
    RmsNormRope,
    Attention,
    AttnSplit,
    AttnCombine,
    SliceCols,
    Copy,
}

/// Binding convention: `ro` read-only storage buffers, then `rw` read-write
/// storage buffers, then the uniform params (dynamic offset).
struct KernelDef {
    name: &'static str,
    wgsl: &'static str,
    ro: u32,
    rw: u32,
}

const KERNELS: [KernelDef; 16] = [
    KernelDef { name: "elementwise", wgsl: K::ELEMENTWISE, ro: 2, rw: 1 },
    KernelDef { name: "embed", wgsl: K::EMBED, ro: 2, rw: 1 },
    KernelDef { name: "embed_q", wgsl: K::EMBED_Q, ro: 3, rw: 1 },
    KernelDef { name: "rmsnorm", wgsl: K::RMSNORM, ro: 4, rw: 1 },
    KernelDef { name: "matmul", wgsl: K::MATMUL, ro: 2, rw: 1 },
    KernelDef { name: "gemv", wgsl: K::GEMV, ro: 2, rw: 1 },
    KernelDef { name: "matmul_q", wgsl: K::MATMUL_Q, ro: 6, rw: 1 },
    KernelDef { name: "matmul_q_tiled", wgsl: K::MATMUL_Q_TILED, ro: 6, rw: 1 },
    KernelDef { name: "matmul_q_scalar", wgsl: K::MATMUL_Q_SCALAR, ro: 3, rw: 1 },
    KernelDef { name: "rope", wgsl: K::ROPE, ro: 2, rw: 1 },
    KernelDef { name: "rmsnorm_rope", wgsl: K::RMSNORM_ROPE, ro: 3, rw: 1 },
    KernelDef { name: "attention", wgsl: K::ATTENTION, ro: 5, rw: 1 },
    KernelDef { name: "attn_split", wgsl: K::ATTN_SPLIT, ro: 5, rw: 2 },
    KernelDef { name: "attn_combine", wgsl: K::ATTN_COMBINE, ro: 1, rw: 1 },
    KernelDef { name: "slice_cols", wgsl: K::SLICE_COLS, ro: 1, rw: 1 },
    KernelDef { name: "copy", wgsl: K::COPY, ro: 1, rw: 1 },
];

const MAX_BINDINGS: usize = 8;

/// Identity of a bind group: kernel + storage buffers + uniform ring segment +
/// uniform struct size. The uniform *offset* is a dynamic offset, so it is not
/// part of the key.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct BgKey {
    kernel: Kernel,
    seg: u16,
    usize_: u32,
    n: u8,
    bufs: [u64; MAX_BINDINGS],
}

const POOL_MAX_BYTES: u64 = 128 << 20; // cap on recycled activation buffers
const BIND_GROUP_CACHE_MAX: usize = 16384;
const UNIFORM_ALIGN: u64 = 256;
const RING_MIN_CAP: u64 = 1 << 18;
const ATTN_TILE: usize = 64; // must match TILE in attention_split.wgsl
const ATTN_MAX_SPLITS: usize = 16;

fn storage_usage() -> wgpu::BufferUsages {
    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST
}

struct RingSegment {
    buf: wgpu::Buffer,
    data: Vec<u8>,
    off: u64,
    cap: u64,
}

struct Layouts {
    bind: wgpu::BindGroupLayout,
    pipeline: wgpu::PipelineLayout,
}

/// Mutable recording state, behind a `RefCell` so ops take `&self`.
struct State {
    pipelines: HashMap<(Kernel, u32), Arc<wgpu::ComputePipeline>>,
    layouts: Vec<Option<Layouts>>,
    bind_groups: HashMap<BgKey, wgpu::BindGroup>,
    enc: Option<wgpu::CommandEncoder>,
    pass: Option<wgpu::ComputePass<'static>>,
    last_pipeline: Option<(Kernel, u32)>,
    rings: Vec<RingSegment>,
    /// Buffers `alloc()`'d inside the current frame (buffer, pool-eligible).
    frame: Option<Vec<(Buf, bool)>>,
    /// Recycled activation buffers: pow2 byte size -> buffers.
    pool: HashMap<u64, Vec<Buf>>,
    pooled_bytes: u64,
    /// Per-frame memo of u32 index uploads keyed by contents, so the same
    /// positions array isn't re-uploaded for each of the 35 layers.
    u32_memo: HashMap<Vec<u32>, Buf>,
    /// Placeholder bound to storage slots a kernel variant does not read.
    dummy: Buf,
    /// Reusable MAP_READ staging buffer for `readback`.
    staging: Option<wgpu::Buffer>,
    /// Ids of buffers destroyed since the bind-group cache was last purged.
    dead_ids: Vec<u64>,
}

pub struct WgpuBackend {
    pub ctx: GpuContext,
    st: RefCell<State>,
    trusted: bool,
    /// Fixed MATMUL_Q lanes-per-row override (tuning); `None` picks by N.
    matmul_lanes: Option<usize>,
}

impl WgpuBackend {
    pub fn new(ctx: GpuContext) -> Self {
        // Kernels are compiled without naga's runtime checks unless
        // GEMMA_WGPU_TRUSTED_SHADERS=0 (see `with_trusted_shaders`).
        let trusted = std::env::var("GEMMA_WGPU_TRUSTED_SHADERS").map(|v| v != "0").unwrap_or(true);
        let matmul_lanes = std::env::var("GEMMA_WGPU_MATMUL_LANES").ok().and_then(|v| v.parse().ok()).filter(|l: &usize| l.is_power_of_two() && (4..=64).contains(l));
        let dummy = tag(ctx.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("dummy"),
            size: 256,
            usage: storage_usage(),
            mapped_at_creation: false,
        }));
        Self {
            ctx,
            trusted,
            matmul_lanes,
            st: RefCell::new(State {
                pipelines: HashMap::new(),
                layouts: (0..KERNELS.len()).map(|_| None).collect(),
                bind_groups: HashMap::new(),
                enc: None,
                pass: None,
                last_pipeline: None,
                rings: Vec::new(),
                frame: None,
                pool: HashMap::new(),
                pooled_bytes: 0,
                u32_memo: HashMap::new(),
                dummy,
                staging: None,
                dead_ids: Vec::new(),
            }),
        }
    }

    /// Acquire a device and wrap it.
    pub fn create() -> Result<Self, String> {
        Ok(Self::new(crate::device::request_device()?))
    }

    /// Whether kernels compile without naga's per-access buffer bounds checks
    /// and loop-bounding counters (`wgpu::ShaderRuntimeChecks::unchecked()`).
    ///
    /// Every kernel here indexes with explicit dims from its uniform params
    /// and is validated against the CPU reference in both modes, so the
    /// checks only cost throughput (roughly 2x at decode, 6x at prefill, since
    /// they land in the tiled GEMM's workgroup-memory inner loops; Chrome's
    /// Tint clamps instead). Trusted is the default; pass `false` (or set
    /// `GEMMA_WGPU_TRUSTED_SHADERS=0`) to keep the checks. Only affects
    /// pipelines compiled after the call.
    pub fn with_trusted_shaders(mut self, trusted: bool) -> Self {
        self.trusted = trusted;
        self
    }

    pub fn trusted_shaders(&self) -> bool {
        self.trusted
    }

    fn device(&self) -> &wgpu::Device {
        &self.ctx.device
    }

    // --- shared encoder / batching ------------------------------------------

    fn end_pass(st: &mut State) {
        st.pass = None; // dropping the pass ends it
        st.last_pipeline = None;
    }

    fn encoder<'a>(&self, st: &'a mut State) -> &'a mut wgpu::CommandEncoder {
        if st.enc.is_none() {
            st.enc = Some(self.device().create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gemma-batch") }));
        }
        st.enc.as_mut().unwrap()
    }

    fn ensure_pass(&self, st: &mut State) {
        if st.pass.is_none() {
            let enc = self.encoder(st);
            let pass = enc
                .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("gemma-ops"), timestamp_writes: None })
                .forget_lifetime();
            st.pass = Some(pass);
            st.last_pipeline = None;
        }
    }

    /// Record a raw buffer-to-buffer copy into the shared encoder (splits the pass).
    fn copy_raw(&self, st: &mut State, src: &wgpu::Buffer, src_off: u64, dst: &wgpu::Buffer, dst_off: u64, bytes: u64) {
        if bytes == 0 {
            return;
        }
        Self::end_pass(st);
        self.encoder(st).copy_buffer_to_buffer(src, src_off, dst, dst_off, bytes);
    }

    /// Copy `n` f32 elements as a dispatch, keeping the compute pass open.
    fn copy_elems(&self, st: &mut State, src: &Buf, src_off: usize, dst: &Buf, dst_off: usize, n: usize) {
        if n == 0 {
            return;
        }
        let wg = n.div_ceil(256);
        if wg > 65535 {
            self.copy_raw(st, src, (src_off * 4) as u64, dst, (dst_off * 4) as u64, (n * 4) as u64);
            return;
        }
        self.dispatch(st, Kernel::Copy, 0, &[], &[src, dst], &[U::U(n as u32), U::U(src_off as u32), U::U(dst_off as u32), U::U(0)], [wg as u32, 1, 1]);
    }

    fn flush_inner(&self, st: &mut State) {
        Self::end_pass(st);
        let Some(enc) = st.enc.take() else { return };
        for seg in st.rings.iter_mut() {
            if seg.off > 0 {
                self.ctx.queue.write_buffer(&seg.buf, 0, &seg.data[..seg.off as usize]);
            }
            seg.off = 0; // safe to reuse: queue operations execute in order
        }
        self.ctx.queue.submit([enc.finish()]);
    }

    // --- allocation ---------------------------------------------------------

    fn raw_buf(&self, byte_len: u64, usage: wgpu::BufferUsages, label: &str) -> wgpu::Buffer {
        self.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: byte_len.max(16).div_ceil(4) * 4,
            usage,
            mapped_at_creation: false,
        })
    }

    /// Activation allocation: pow2-bucketed so frames can recycle buffers.
    /// Kernels never rely on buffer length (all use explicit dims) nor on
    /// zero-initialization (every op fully writes its output), so a recycled
    /// buffer with stale contents is fine.
    fn alloc_in(&self, st: &mut State, shape: &[usize]) -> GpuTensor {
        let n: usize = shape.iter().product();
        let want = (n as u64 * 4).max(16);
        let mut size = 256u64;
        while size < want {
            size <<= 1;
        }
        let buf = match st.pool.get_mut(&size).and_then(|l| l.pop()) {
            Some(b) => {
                st.pooled_bytes -= size;
                b
            }
            None => tag(self.raw_buf(size, storage_usage(), "activation")),
        };
        if let Some(f) = st.frame.as_mut() {
            f.push((buf.clone(), true));
        }
        GpuTensor::new(buf, shape)
    }

    fn recycle(st: &mut State, buf: Buf, pooled: bool) {
        let size = buf.size();
        if !pooled || st.pooled_bytes + size > POOL_MAX_BYTES {
            Self::destroy(st, &buf);
            return;
        }
        st.pool.entry(size).or_default().push(buf);
        st.pooled_bytes += size;
    }

    fn destroy(st: &mut State, buf: &Buf) {
        buf.destroy();
        st.dead_ids.push(buf.id);
    }

    /// Drop cached bind groups that reference destroyed buffers.
    fn purge_dead(st: &mut State) {
        if st.dead_ids.is_empty() {
            return;
        }
        let dead: std::collections::HashSet<u64> = st.dead_ids.drain(..).collect();
        st.bind_groups.retain(|k, _| !k.bufs[..k.n as usize].iter().any(|id| dead.contains(id)));
    }

    fn upload_bytes(&self, bytes: &[u8], label: &str) -> wgpu::Buffer {
        let padded = (bytes.len() as u64).div_ceil(4) * 4;
        let buf = self.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some(label),
            size: padded.max(16),
            usage: storage_usage(),
            mapped_at_creation: true,
        });
        buf.slice(..).get_mapped_range_mut().expect("mapped at creation").slice(..bytes.len()).copy_from_slice(bytes);
        buf.unmap();
        buf
    }

    /// Upload raw packed weight bytes as a storage buffer addressable as `array<u32>`.
    pub fn bytes_buffer(&self, bytes: &[u8]) -> Buf {
        tag(self.upload_bytes(bytes, "packed-weight"))
    }

    /// Build a quant tensor from an already-uploaded GPU byte buffer.
    pub fn quant_tensor_from_gpu(&self, buffer: Buf, scales: &[f32], shape: [usize; 2], bits: u32, signed_i8: bool, group_size: usize) -> GpuTensor {
        let per_byte = if signed_i8 { 1 } else { (8 / bits) as usize };
        let scale_buffer = tag(self.upload_bytes(bytemuck::cast_slice(scales), "quant-scale"));
        GpuTensor {
            buffer,
            shape: shape.to_vec(),
            cap_rows: None,
            quant: Some(Arc::new(GpuQuant { scale_buffer, bits, signed_i8, group_size, per_byte })),
        }
    }

    /// A u32 index buffer (token ids / positions): pooled, frame-tracked and
    /// memoized per frame by contents. The write is queued before the frame's
    /// submit, and a pooled buffer is idle by construction (see `end_frame`).
    fn u32_buf(&self, st: &mut State, arr: &[u32]) -> Buf {
        if let Some(b) = st.u32_memo.get(arr) {
            return b.clone();
        }
        let t = self.alloc_in(st, &[arr.len().max(1)]);
        self.ctx.queue.write_buffer(&t.buffer, 0, bytemuck::cast_slice(arr));
        st.u32_memo.insert(arr.to_vec(), t.buffer.clone());
        t.buffer
    }

    /// Stage a uniform struct into the ring: returns (segment, offset, size).
    fn uniform_slot(&self, st: &mut State, vals: &[U]) -> (usize, u64, u64) {
        let size = (vals.len() as u64 * 4).div_ceil(16) * 16;
        let need = size.div_ceil(UNIFORM_ALIGN) * UNIFORM_ALIGN;
        let mut idx = st.rings.iter().position(|s| s.off + need <= s.cap);
        if idx.is_none() {
            let cap = need.max(RING_MIN_CAP);
            st.rings.push(RingSegment {
                buf: self.raw_buf(cap, wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST, "uniform-ring"),
                data: vec![0u8; cap as usize],
                off: 0,
                cap,
            });
            idx = Some(st.rings.len() - 1);
        }
        let i = idx.unwrap();
        let seg = &mut st.rings[i];
        let off = seg.off as usize;
        for (j, v) in vals.iter().enumerate() {
            seg.data[off + j * 4..off + j * 4 + 4].copy_from_slice(&v.bits().to_le_bytes());
        }
        for b in &mut seg.data[off + vals.len() * 4..off + size as usize] {
            *b = 0;
        }
        seg.off += need;
        (i, off as u64, size)
    }

    fn layouts<'a>(&self, st: &'a mut State, kernel: Kernel) -> &'a Layouts {
        let ki = kernel as usize;
        if st.layouts[ki].is_none() {
            let def = &KERNELS[ki];
            let mut entries = Vec::new();
            for b in 0..def.ro + def.rw {
                entries.push(wgpu::BindGroupLayoutEntry {
                    binding: b,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: b < def.ro },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                });
            }
            entries.push(wgpu::BindGroupLayoutEntry {
                binding: def.ro + def.rw,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer { ty: wgpu::BufferBindingType::Uniform, has_dynamic_offset: true, min_binding_size: None },
                count: None,
            });
            let bind = self.device().create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor { label: Some(def.name), entries: &entries });
            let pipeline = self.device().create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some(def.name),
                bind_group_layouts: &[Some(&bind)],
                immediate_size: 0,
            });
            st.layouts[ki] = Some(Layouts { bind, pipeline });
        }
        st.layouts[ki].as_ref().unwrap()
    }

    fn pipeline(&self, st: &mut State, kernel: Kernel, variant: u32, constants: &[(&str, f64)]) -> Arc<wgpu::ComputePipeline> {
        if let Some(p) = st.pipelines.get(&(kernel, variant)) {
            return p.clone();
        }
        let def = &KERNELS[kernel as usize];
        let desc = wgpu::ShaderModuleDescriptor { label: Some(def.name), source: wgpu::ShaderSource::Wgsl(def.wgsl.into()) };
        let module = if self.trusted {
            // SAFETY: these kernels never index past the buffers they are
            // bound to (all bounds come from explicit uniform dims) and
            // every loop is bounded by those dims; see `with_trusted_shaders`.
            unsafe { self.device().create_shader_module_trusted(desc, wgpu::ShaderRuntimeChecks::unchecked()) }
        } else {
            self.device().create_shader_module(desc)
        };
        let layout = &self.layouts(st, kernel).pipeline;
        let p = Arc::new(self.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some(def.name),
            layout: Some(layout),
            module: &module,
            entry_point: Some("main"),
            compilation_options: wgpu::PipelineCompilationOptions { constants, zero_initialize_workgroup_memory: false },
            cache: None,
        }));
        st.pipelines.insert((kernel, variant), p.clone());
        p
    }

    /// Record one dispatch. `bufs` are the storage bindings in layout order;
    /// `uniform` is the params struct.
    fn dispatch(&self, st: &mut State, kernel: Kernel, variant: u32, constants: &[(&str, f64)], bufs: &[&Buf], uniform: &[U], workgroups: [u32; 3]) {
        debug_assert_eq!(bufs.len() as u32, KERNELS[kernel as usize].ro + KERNELS[kernel as usize].rw);
        let (seg, offset, usize_) = self.uniform_slot(st, uniform);
        let pipeline = self.pipeline(st, kernel, variant, constants);
        let mut key = BgKey { kernel, seg: seg as u16, usize_: usize_ as u32, n: bufs.len() as u8, bufs: [0; MAX_BINDINGS] };
        for (i, b) in bufs.iter().enumerate() {
            key.bufs[i] = b.id;
        }
        if !st.bind_groups.contains_key(&key) {
            if st.bind_groups.len() >= BIND_GROUP_CACHE_MAX {
                st.bind_groups.clear();
            }
            self.layouts(st, kernel);
            let mut entries: Vec<wgpu::BindGroupEntry> = bufs.iter().enumerate().map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() }).collect();
            entries.push(wgpu::BindGroupEntry {
                binding: bufs.len() as u32,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding { buffer: &st.rings[seg].buf, offset: 0, size: NonZeroU64::new(usize_) }),
            });
            let layout = &st.layouts[kernel as usize].as_ref().unwrap().bind;
            let bg = self.device().create_bind_group(&wgpu::BindGroupDescriptor { label: Some(KERNELS[kernel as usize].name), layout, entries: &entries });
            st.bind_groups.insert(key, bg);
        }
        self.ensure_pass(st);
        let set_pipeline = st.last_pipeline != Some((kernel, variant));
        let bg = st.bind_groups.get(&key).unwrap();
        let pass = st.pass.as_mut().unwrap();
        if set_pipeline {
            pass.set_pipeline(&pipeline);
        }
        pass.set_bind_group(0, bg, &[offset as u32]);
        pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
        st.last_pipeline = Some((kernel, variant));
    }

    fn ew(&self, a: &GpuTensor, b: Option<&GpuTensor>, op: u32, k: f32) -> GpuTensor {
        let mut st = self.st.borrow_mut();
        let st = &mut *st;
        let out = self.alloc_in(st, &a.shape);
        let n = a.size();
        self.dispatch(st, Kernel::Elementwise, 0, &[], &[&a.buffer, &b.unwrap_or(a).buffer, &out.buffer], &[U::U(n as u32), U::U(op), U::F(k), U::U(0)], [(n as u32).div_ceil(256), 1, 1]);
        out
    }

    fn rms_norm_impl(&self, x: &GpuTensor, w: Option<&GpuTensor>, residual: Option<&GpuTensor>, scale_by: Option<&GpuTensor>, eps: f32) -> GpuTensor {
        let mut st = self.st.borrow_mut();
        let st = &mut *st;
        let d = *x.shape.last().unwrap();
        let rows = x.size() / d;
        let out = self.alloc_in(st, &x.shape);
        let flags = (w.is_some() as u32) | ((residual.is_some() as u32) << 1) | ((scale_by.is_some() as u32) << 2);
        self.dispatch(
            st,
            Kernel::RmsNorm,
            0,
            &[],
            &[&x.buffer, &w.unwrap_or(x).buffer, &residual.unwrap_or(x).buffer, &scale_by.unwrap_or(x).buffer, &out.buffer],
            &[U::U(rows as u32), U::U(d as u32), U::F(eps), U::U(flags)],
            [rows as u32, 1, 1],
        );
        out
    }

    /// Whether `w` (quantized) can go through the vec4/whole-word MATMUL_Q path
    /// for activations of width `k` and `m` rows: whole u32 words per row,
    /// per-row scales, workgroup grid within limits.
    fn quant_fast_path(q: &GpuQuant, k: usize, m: usize, mrows: usize) -> bool {
        let bits = if q.signed_i8 { 8 } else { q.bits };
        let vals_per_word = (32 / bits) as usize;
        k % vals_per_word == 0 && q.group_size == k && m.div_ceil(mrows) <= 65535
    }

    fn mrows_for(m: usize) -> usize {
        if m == 1 {
            1
        } else if m <= 4 {
            4
        } else {
            8
        }
    }

    /// MATMUL_Q dispatch. `mode` 0/1/2 as documented in the kernel; `w2` is the
    /// second weight for mode 1, `bs` the `[M, stride]` multiplicand for mode 2.
    #[allow(clippy::too_many_arguments)]
    fn matmul_q(&self, st: &mut State, x: &GpuTensor, w: &GpuTensor, q: &GpuQuant, mode: u32, w2: Option<(&GpuTensor, &GpuQuant)>, bs: Option<(&GpuTensor, usize)>) -> GpuTensor {
        let m = x.shape[0];
        let kd = x.shape[1];
        let n = w.shape[0];
        let out = self.alloc_in(st, &[m, n]);
        let bits = if q.signed_i8 { 8 } else { q.bits };
        let vals_per_word = (32 / bits) as usize;
        let mrows = Self::mrows_for(m);
        let (w2b, s2b) = match w2 {
            Some((t, q2)) => (&t.buffer, &q2.scale_buffer),
            None => (&w.buffer, &q.scale_buffer),
        };
        let (bsb, b_off, b_stride) = match bs {
            Some((t, off)) => (&t.buffer, off, t.shape[1]),
            None => (&x.buffer, 0, 0),
        };
        let uniform = [U::U(m as u32), U::U(kd as u32), U::U(n as u32), U::U((kd / vals_per_word) as u32), U::U(b_off as u32), U::U(b_stride as u32), U::U(0), U::U(0)];
        if m >= 16 && kd % 32 == 0 && n.div_ceil(64) <= 65535 && m.div_ceil(64) <= 65535 {
            self.dispatch(
                st,
                Kernel::MatmulQTiled,
                bits | (mode << 16),
                &[("BITS", bits as f64), ("MODE", mode as f64)],
                &[&x.buffer, &w.buffer, &q.scale_buffer, w2b, s2b, bsb, &out.buffer],
                &uniform,
                [(n as u32).div_ceil(64), (m as u32).div_ceil(64), 1],
            );
            return out;
        }
        // Lanes per output row: narrow projections (k/v at N = 256) would
        // otherwise be a handful of workgroups on a 10-core GPU.
        let lanes: usize = match self.matmul_lanes {
            Some(l) => l,
            None if n >= 2048 => 16,
            None if n >= 512 => 32,
            None => 64,
        };
        let outs = 256 / lanes;
        let variant = bits | ((mrows as u32) << 8) | (mode << 16) | ((lanes as u32) << 24);
        self.dispatch(
            st,
            Kernel::MatmulQ,
            variant,
            &[("BITS", bits as f64), ("MROWS", mrows as f64), ("MODE", mode as f64), ("LANES", lanes as f64)],
            &[&x.buffer, &w.buffer, &q.scale_buffer, w2b, s2b, bsb, &out.buffer],
            &uniform,
            [(n as u32).div_ceil(outs as u32), (m as u32).div_ceil(mrows as u32), 1],
        );
        out
    }

    /// Block until the GPU has finished all submitted work.
    pub fn wait_idle(&self) {
        self.flush();
        let _ = self.device().poll(wgpu::PollType::wait_indefinitely());
    }
}

impl Backend for WgpuBackend {
    type Tensor = GpuTensor;

    fn name(&self) -> &str {
        "wgpu"
    }

    fn tensor(&self, data: &[f32], shape: &[usize]) -> GpuTensor {
        GpuTensor::new(tag(self.upload_bytes(bytemuck::cast_slice(data), "f32-tensor")), shape)
    }

    fn quant_tensor(&self, q: QuantWeight) -> GpuTensor {
        let buffer = self.bytes_buffer(&q.qbytes);
        self.quant_tensor_from_gpu(buffer, &q.scales, q.shape, q.bits, q.signed_i8, q.group_size)
    }

    fn zeros(&self, shape: &[usize]) -> GpuTensor {
        let mut st = self.st.borrow_mut();
        let st = &mut *st;
        let t = self.alloc_in(st, shape);
        Self::end_pass(st);
        let bytes = (t.size() as u64 * 4).max(4);
        self.encoder(st).clear_buffer(&t.buffer, 0, Some(bytes));
        t
    }

    fn reshape(&self, x: &GpuTensor, shape: &[usize]) -> GpuTensor {
        GpuTensor { buffer: x.buffer.clone(), shape: shape.to_vec(), cap_rows: None, quant: x.quant.clone() }
    }

    fn embed_rows(&self, table: &GpuTensor, ids: &[u32], scale: f32) -> GpuTensor {
        let mut st = self.st.borrow_mut();
        let st = &mut *st;
        let d = table.shape[1];
        let t = ids.len();
        let out = self.alloc_in(st, &[t, d]);
        let idbuf = self.u32_buf(st, ids);
        let wg = ((t * d) as u32).div_ceil(256);
        if let Some(q) = &table.quant {
            let s_stride = d.div_ceil(q.group_size);
            self.dispatch(
                st,
                Kernel::EmbedQ,
                0,
                &[],
                &[&table.buffer, &idbuf, &q.scale_buffer, &out.buffer],
                &[U::U(t as u32), U::U(d as u32), U::U(q.bits), U::U(q.per_byte as u32), U::U(q.group_size as u32), U::U(s_stride as u32), U::F(scale), U::U(0)],
                [wg, 1, 1],
            );
            return out;
        }
        self.dispatch(st, Kernel::Embed, 0, &[], &[&table.buffer, &idbuf, &out.buffer], &[U::U(t as u32), U::U(d as u32), U::F(scale), U::U(0)], [wg, 1, 1]);
        out
    }

    // weight = None runs the unweighted variant.
    fn rms_norm(&self, x: &GpuTensor, w: Option<&GpuTensor>, eps: f32) -> GpuTensor {
        self.rms_norm_impl(x, w, None, None, eps)
    }

    fn add_rms_norm(&self, residual: &GpuTensor, x: &GpuTensor, w: &GpuTensor, eps: f32, scale_by: Option<&GpuTensor>) -> GpuTensor {
        self.rms_norm_impl(x, Some(w), Some(residual), scale_by, eps)
    }

    fn linear(&self, x: &GpuTensor, w: &GpuTensor) -> GpuTensor {
        let mut st = self.st.borrow_mut();
        let st = &mut *st;
        let m = x.shape[0];
        let kd = x.shape[1];
        let n = w.shape[0];
        if let Some(q) = &w.quant {
            if Self::quant_fast_path(q, kd, m, Self::mrows_for(m)) {
                return self.matmul_q(st, x, w, q, 0, None, None);
            }
            let out = self.alloc_in(st, &[m, n]);
            self.dispatch(
                st,
                Kernel::MatmulQScalar,
                0,
                &[],
                &[&x.buffer, &w.buffer, &q.scale_buffer, &out.buffer],
                &[U::U(m as u32), U::U(kd as u32), U::U(n as u32), U::U(q.bits), U::U(q.per_byte as u32), U::U(q.signed_i8 as u32), U::U(0), U::U(0)],
                [(n as u32).div_ceil(64), m as u32, 1],
            );
            return out;
        }
        let out = self.alloc_in(st, &[m, n]);
        if m == 1 && n <= 65535 {
            // decode-time f32 matvec: one workgroup per output row, lanes stride K.
            self.dispatch(st, Kernel::Gemv, 0, &[], &[&x.buffer, &w.buffer, &out.buffer], &[U::U(kd as u32), U::U(n as u32), U::U(0), U::U(0)], [n as u32, 1, 1]);
            return out;
        }
        self.dispatch(
            st,
            Kernel::Matmul,
            0,
            &[],
            &[&x.buffer, &w.buffer, &out.buffer],
            &[U::U(m as u32), U::U(kd as u32), U::U(n as u32), U::U(0)],
            [(n as u32).div_ceil(16), (m as u32).div_ceil(16), 1],
        );
        out
    }

    fn linear_geglu(&self, x: &GpuTensor, w_gate: &GpuTensor, w_up: &GpuTensor) -> GpuTensor {
        if let (Some(qg), Some(qu)) = (&w_gate.quant, &w_up.quant) {
            let m = x.shape[0];
            let kd = x.shape[1];
            let same = w_gate.shape == w_up.shape && qg.bits == qu.bits && qg.signed_i8 == qu.signed_i8 && qg.group_size == qu.group_size;
            if same && Self::quant_fast_path(qg, kd, m, Self::mrows_for(m)) {
                let mut st = self.st.borrow_mut();
                return self.matmul_q(&mut st, x, w_gate, qg, 1, Some((w_up, qu)), None);
            }
        }
        let g = self.linear(x, w_gate);
        let u = self.linear(x, w_up);
        self.geglu(&g, &u)
    }

    fn linear_gelu_mul_cols(&self, x: &GpuTensor, w: &GpuTensor, b: &GpuTensor, offset: usize) -> GpuTensor {
        if let Some(q) = &w.quant {
            let m = x.shape[0];
            let kd = x.shape[1];
            if b.shape.len() == 2 && b.shape[0] == m && Self::quant_fast_path(q, kd, m, Self::mrows_for(m)) {
                let mut st = self.st.borrow_mut();
                return self.matmul_q(&mut st, x, w, q, 2, None, Some((b, offset)));
            }
        }
        let g = self.linear(x, w);
        let s = self.slice_cols(b, offset, w.shape[0]);
        self.geglu(&g, &s)
    }

    fn add(&self, a: &GpuTensor, b: &GpuTensor) -> GpuTensor {
        self.ew(a, Some(b), 0, 0.0)
    }
    fn mul(&self, a: &GpuTensor, b: &GpuTensor) -> GpuTensor {
        self.ew(a, Some(b), 1, 0.0)
    }
    fn scale(&self, a: &GpuTensor, s: f32) -> GpuTensor {
        self.ew(a, None, 2, s)
    }
    fn scale_by_tensor(&self, a: &GpuTensor, s: &GpuTensor) -> GpuTensor {
        self.ew(a, Some(s), 6, 0.0)
    }
    fn gelu_tanh(&self, a: &GpuTensor) -> GpuTensor {
        self.ew(a, None, 3, 0.0)
    }
    fn geglu(&self, gate: &GpuTensor, up: &GpuTensor) -> GpuTensor {
        self.ew(gate, Some(up), 4, 0.0)
    }
    fn softcap(&self, a: &GpuTensor, cap: f32) -> GpuTensor {
        self.ew(a, None, 5, cap)
    }

    fn rope(&self, x: &GpuTensor, positions: &[u32], theta: f32, head_dim: usize, heads: usize, rotary_dim: usize) -> GpuTensor {
        let mut st = self.st.borrow_mut();
        let st = &mut *st;
        let t = x.shape[0];
        let out = self.alloc_in(st, &x.shape);
        let posbuf = self.u32_buf(st, positions);
        self.dispatch(
            st,
            Kernel::Rope,
            0,
            &[],
            &[&x.buffer, &posbuf, &out.buffer],
            &[U::U(t as u32), U::U(heads as u32), U::U(head_dim as u32), U::U(rotary_dim as u32), U::F(theta), U::F(0.0), U::F(0.0), U::F(0.0)],
            [((t * heads * head_dim) as u32).div_ceil(256), 1, 1],
        );
        out
    }

    fn head_norm_rope(&self, x: &GpuTensor, w: &GpuTensor, eps: f32, positions: &[u32], theta: f32, head_dim: usize, heads: usize, rotary_dim: usize) -> GpuTensor {
        let mut st = self.st.borrow_mut();
        let st = &mut *st;
        let t = x.shape[0];
        let out = self.alloc_in(st, &x.shape);
        let posbuf = self.u32_buf(st, positions);
        self.dispatch(
            st,
            Kernel::RmsNormRope,
            0,
            &[],
            &[&x.buffer, &w.buffer, &posbuf, &out.buffer],
            &[U::U(t as u32), U::U(heads as u32), U::U(head_dim as u32), U::U(rotary_dim as u32), U::F(theta), U::F(eps), U::F(0.0), U::F(0.0)],
            [(t * heads) as u32, 1, 1],
        );
        out
    }

    fn attention(&self, q: &GpuTensor, k: &GpuTensor, v: &GpuTensor, o: &AttnOpts) -> GpuTensor {
        let mut st = self.st.borrow_mut();
        let st = &mut *st;
        let (t, hq, dh) = (q.shape[0], q.shape[1], q.shape[2]);
        let s = k.shape[0];
        let hkv = k.shape[1];
        let out = self.alloc_in(st, &[t, hq * dh]);
        let qp = self.u32_buf(st, o.q_pos);
        let kp = self.u32_buf(st, o.k_pos);

        let group = hq / hkv;
        let grouped = hkv > 0 && hq % hkv == 0 && group <= 8 && dh % 4 == 0 && (group * dh) % 256 == 0 && (group * dh) / 256 <= 16;
        if !grouped || t == 0 {
            self.dispatch(
                st,
                Kernel::Attention,
                0,
                &[],
                &[&q.buffer, &k.buffer, &v.buffer, &qp, &kp, &out.buffer],
                &[
                    U::U(t as u32), U::U(s as u32), U::U(hq as u32), U::U(hkv as u32),
                    U::U(dh as u32), U::U(o.sliding_window), U::U(0), U::U(0),
                    U::F(o.scale), U::F(0.0), U::F(0.0), U::F(0.0),
                ],
                [t as u32, hq as u32, 1],
            );
            return out;
        }

        // Active key range shared by every query in the step (k_pos ascending):
        // keys after the newest query are causally masked for all of them, and
        // with a sliding window keys at or before (oldest query − window) are
        // masked for all of them. Sliding layers therefore stay O(window) at
        // decode however long the cache is, before any tile-level skip.
        let qmin = o.q_pos.iter().copied().min().unwrap_or(0);
        let qmax = o.q_pos.iter().copied().max().unwrap_or(0);
        let k_end = o.k_pos.partition_point(|&kp| kp <= qmax).min(s);
        let k_start = if o.sliding_window > 0 && qmin >= o.sliding_window {
            o.k_pos.partition_point(|&kp| kp + o.sliding_window <= qmin).min(k_end)
        } else {
            0
        };
        let nkeys = k_end - k_start;
        let splits = nkeys.div_ceil(ATTN_TILE).clamp(1, ATTN_MAX_SPLITS);
        let keys_per_split = nkeys.div_ceil(splits).div_ceil(ATTN_TILE).max(1) * ATTN_TILE;
        let splits = if nkeys == 0 { 1 } else { nkeys.div_ceil(keys_per_split) };
        let dpt = (group * dh) / 256;

        let part = if splits > 1 { self.alloc_in(st, &[t * hq * splits * (dh + 2)]).buffer } else { st.dummy.clone() };
        self.dispatch(
            st,
            Kernel::AttnSplit,
            dpt as u32,
            &[("DPT", dpt as f64)],
            &[&q.buffer, &k.buffer, &v.buffer, &qp, &kp, &part, &out.buffer],
            &[
                U::U(t as u32), U::U(s as u32), U::U(hq as u32), U::U(hkv as u32),
                U::U(dh as u32), U::U(o.sliding_window), U::U(k_start as u32), U::U(k_end as u32),
                U::U(splits as u32), U::U(keys_per_split as u32), U::U(0), U::U(0),
                U::F(o.scale), U::F(0.0), U::F(0.0), U::F(0.0),
            ],
            [t as u32, hkv as u32, splits as u32],
        );
        if splits > 1 {
            self.dispatch(
                st,
                Kernel::AttnCombine,
                0,
                &[],
                &[&part, &out.buffer],
                &[U::U(t as u32), U::U(hq as u32), U::U(splits as u32), U::U(dh as u32)],
                [((t * hq * dh) as u32).div_ceil(256), 1, 1],
            );
        }
        out
    }

    fn slice_rows(&self, x: &GpuTensor, start: usize, count: usize) -> GpuTensor {
        let mut st = self.st.borrow_mut();
        let st = &mut *st;
        let dim = x.shape[1];
        let out = self.alloc_in(st, &[count, dim]);
        self.copy_elems(st, &x.buffer, start * dim, &out.buffer, 0, count * dim);
        out
    }

    fn slice_cols(&self, x: &GpuTensor, offset: usize, width: usize) -> GpuTensor {
        let mut st = self.st.borrow_mut();
        let st = &mut *st;
        let (rows, stride) = (x.shape[0], x.shape[1]);
        let out = self.alloc_in(st, &[rows, width]);
        self.dispatch(
            st,
            Kernel::SliceCols,
            0,
            &[],
            &[&x.buffer, &out.buffer],
            &[U::U(rows as u32), U::U(stride as u32), U::U(offset as u32), U::U(width as u32)],
            [((rows * width) as u32).div_ceil(256), 1, 1],
        );
        out
    }

    fn concat_rows(&self, a: Option<&GpuTensor>, b: &GpuTensor) -> GpuTensor {
        let mut st = self.st.borrow_mut();
        let st = &mut *st;
        let Some(a) = a else {
            let out = self.alloc_in(st, &b.shape);
            self.copy_elems(st, &b.buffer, 0, &out.buffer, 0, b.size());
            return out;
        };
        let mut shape = a.shape.clone();
        shape[0] += b.shape[0];
        let out = self.alloc_in(st, &shape);
        self.copy_elems(st, &a.buffer, 0, &out.buffer, 0, a.size());
        self.copy_elems(st, &b.buffer, 0, &out.buffer, a.size(), b.size());
        out
    }

    /// Append `add` (`[rows, ...]`) to the KV tensor `existing` in place, growing
    /// a capacity buffer geometrically. Returns a view with the new logical row
    /// count (sharing the capacity buffer) plus the retired old buffer when
    /// growth reallocated. Turns the per-step O(S) cache copy into O(new rows).
    fn kv_append(&self, existing: Option<&GpuTensor>, add: &GpuTensor) -> KvAppend<GpuTensor> {
        let mut st = self.st.borrow_mut();
        let st = &mut *st;
        let rows = add.shape[0];
        let dim = add.size() / rows;
        let Some(existing) = existing else {
            let cap = rows.max(256);
            let buf = tag(self.raw_buf((cap * dim * 4) as u64, storage_usage(), "kv-cache"));
            if let Some(f) = st.frame.as_mut() {
                f.push((buf.clone(), false));
            }
            self.copy_elems(st, &add.buffer, 0, &buf, 0, add.size());
            let mut t = GpuTensor::new(buf, &add.shape);
            t.cap_rows = Some(cap);
            return KvAppend { tensor: t, dead: None };
        };
        let len = existing.shape[0];
        let cap_rows = existing.cap_rows.unwrap_or(len);
        let mut dst = existing.clone();
        let mut dead = None;
        if len + rows > cap_rows {
            let cap = (cap_rows * 2).max(len + rows);
            let buf = tag(self.raw_buf((cap * dim * 4) as u64, storage_usage(), "kv-cache"));
            if let Some(f) = st.frame.as_mut() {
                f.push((buf.clone(), false));
            }
            self.copy_elems(st, &existing.buffer, 0, &buf, 0, len * dim);
            dst = GpuTensor::new(buf, &existing.shape);
            dst.cap_rows = Some(cap);
            dead = Some(existing.clone());
        }
        self.copy_elems(st, &add.buffer, 0, &dst.buffer, len * dim, add.size());
        let mut shape = add.shape.clone();
        shape[0] = len + rows;
        let mut view = GpuTensor::new(dst.buffer.clone(), &shape);
        view.cap_rows = dst.cap_rows;
        KvAppend { tensor: view, dead }
    }

    fn readback(&self, t: &GpuTensor) -> Vec<f32> {
        let bytes = (t.size() * 4) as u64;
        let staging = {
            let mut st = self.st.borrow_mut();
            let st = &mut *st;
            if st.staging.as_ref().is_none_or(|s| s.size() < bytes) {
                if let Some(old) = st.staging.take() {
                    old.destroy();
                }
                st.staging = Some(self.raw_buf(bytes.max(1 << 16), wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, "staging"));
            }
            let staging = st.staging.as_ref().unwrap().clone();
            self.copy_raw(st, &t.buffer, 0, &staging, 0, bytes);
            self.flush_inner(st);
            staging
        };
        let (tx, rx) = std::sync::mpsc::channel();
        staging.slice(0..bytes).map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.device().poll(wgpu::PollType::wait_indefinitely()).expect("device poll");
        rx.recv().expect("map callback").expect("buffer map");
        let out: Vec<f32> = {
            let view = staging.slice(0..bytes).get_mapped_range().expect("mapped range");
            bytemuck::cast_slice(&view).to_vec()
        };
        staging.unmap();
        out
    }

    fn flush(&self) {
        let mut st = self.st.borrow_mut();
        self.flush_inner(&mut st);
    }

    fn begin_frame(&self) {
        let mut st = self.st.borrow_mut();
        st.frame = Some(Vec::new());
        st.u32_memo.clear();
    }

    /// Recycle everything allocated since `begin_frame()` except `keep`, plus
    /// the `also_free` tensors (buffers from earlier frames the caller has
    /// retired, e.g. KV tensors superseded by this step's growth).
    ///
    /// Only call this after a blocking `readback()`: the map resolving means
    /// every previously submitted command finished, so the GPU cannot still be
    /// using any buffer this frame touched.
    fn end_frame(&self, keep: &[GpuTensor], also_free: Vec<GpuTensor>) {
        let mut st = self.st.borrow_mut();
        let st = &mut *st;
        self.flush_inner(st);
        let keep_ids: std::collections::HashSet<u64> = keep.iter().map(|t| t.buffer.id).collect();
        if let Some(frame) = st.frame.take() {
            for (buf, pooled) in frame {
                if !keep_ids.contains(&buf.id) {
                    Self::recycle(st, buf, pooled);
                }
            }
        }
        for t in also_free {
            if !keep_ids.contains(&t.buffer.id) {
                Self::destroy(st, &t.buffer);
            }
        }
        st.u32_memo.clear();
        Self::purge_dead(st);
    }

    fn free_tensor(&self, t: GpuTensor) {
        let mut st = self.st.borrow_mut();
        Self::destroy(&mut st, &t.buffer);
        Self::purge_dead(&mut st);
    }
}
