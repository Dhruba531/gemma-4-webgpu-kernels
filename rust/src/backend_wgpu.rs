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
//! **Batching.** Ops record into ONE shared command encoder (and, where
//! possible, one open compute pass — WebGPU auto-synchronizes dependent
//! dispatches within a pass), submitted on `flush()` — which `readback()`
//! triggers. A forward pass is therefore a single `queue.submit` instead of one
//! per op (~900/token), which removes almost all CPU-side driver overhead.
//! Per-op uniforms are suballocated from a persistent ring buffer at 256-byte
//! offsets and written with one `queue.write_buffer` per segment per flush.
//!
//! **Memory lifecycle.** The model brackets each forward pass with
//! `begin_frame()`/`end_frame()`. Every intermediate activation `alloc()`'d
//! inside the frame is recycled at `end_frame()` except the tensors the caller
//! asks to keep (the live KV cache), so GPU memory stays bounded per step.
//! Recycled activation buffers go to a pow2-bucketed pool and are reused by
//! later frames. Weight uploads are never frame-tracked — they live for the
//! model's lifetime. `end_frame` runs right after a blocking readback, which
//! guarantees the GPU has finished every submitted command, so recycling is
//! race-free.

use std::cell::RefCell;
use std::collections::HashMap;
use std::num::NonZeroU64;
use std::sync::Arc;

use crate::backend::{AttnOpts, Backend, KvAppend, QuantWeight, TensorShape};
use crate::device::GpuContext;
use crate::kernels as K;

/// Quantized-weight metadata that rides along with a packed GPU buffer.
#[derive(Debug)]
pub struct GpuQuant {
    /// f32 scales, `[N * scale_stride]`.
    pub scale_buffer: Arc<wgpu::Buffer>,
    pub bits: u32,
    pub signed_i8: bool,
    pub group_size: usize,
    pub per_byte: usize,
}

/// A GPU tensor: an f32 storage buffer (or a packed quantized weight) + shape.
#[derive(Clone, Debug)]
pub struct GpuTensor {
    pub buffer: Arc<wgpu::Buffer>,
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
    fn new(buffer: Arc<wgpu::Buffer>, shape: &[usize]) -> Self {
        Self { buffer, shape: shape.to_vec(), cap_rows: None, quant: None }
    }
}

/// Little-endian uniform packing (16-byte aligned), mirroring the JS `_u32f32`.
#[derive(Clone, Copy)]
pub enum U {
    U(u32),
    F(f32),
}

fn pack_uniform(values: &[U]) -> Vec<u8> {
    let padded = values.len().div_ceil(4) * 4;
    let mut out = vec![0u8; padded * 4];
    for (i, v) in values.iter().enumerate() {
        let bytes = match v {
            U::U(u) => u.to_le_bytes(),
            U::F(f) => f.to_le_bytes(),
        };
        out[i * 4..i * 4 + 4].copy_from_slice(&bytes);
    }
    out
}

const POOL_MAX_BYTES: u64 = 128 << 20; // cap on recycled activation buffers
const UNIFORM_ALIGN: u64 = 256;
const RING_MIN_CAP: u64 = 1 << 18;

fn storage_usage() -> wgpu::BufferUsages {
    wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::COPY_DST
}

struct RingSegment {
    buf: wgpu::Buffer,
    data: Vec<u8>,
    off: u64,
    cap: u64,
}

struct UniformSlot {
    seg: usize,
    offset: u64,
    size: u64,
}

/// Mutable recording state, behind a `RefCell` so ops take `&self`.
struct State {
    pipelines: HashMap<String, wgpu::ComputePipeline>,
    enc: Option<wgpu::CommandEncoder>,
    pass: Option<wgpu::ComputePass<'static>>,
    rings: Vec<RingSegment>,
    /// Buffers `alloc()`'d inside the current frame (buffer, pool-eligible).
    frame: Option<Vec<(Arc<wgpu::Buffer>, bool)>>,
    /// Recycled activation buffers: pow2 byte size -> buffers.
    pool: HashMap<u64, Vec<Arc<wgpu::Buffer>>>,
    pooled_bytes: u64,
    /// Per-frame memo of u32 index uploads keyed by contents, so the same
    /// positions array isn't re-uploaded for each of the 35 layers.
    u32_memo: HashMap<Vec<u32>, Arc<wgpu::Buffer>>,
}

pub struct WgpuBackend {
    pub ctx: GpuContext,
    st: RefCell<State>,
    trusted: bool,
}

impl WgpuBackend {
    pub fn new(ctx: GpuContext) -> Self {
        let trusted = std::env::var("GEMMA_WGPU_TRUSTED_SHADERS").map(|v| v == "1").unwrap_or(false);
        Self {
            ctx,
            trusted,
            st: RefCell::new(State {
                pipelines: HashMap::new(),
                enc: None,
                pass: None,
                rings: Vec::new(),
                frame: None,
                pool: HashMap::new(),
                pooled_bytes: 0,
                u32_memo: HashMap::new(),
            }),
        }
    }

    /// Acquire a device and wrap it.
    pub fn create() -> Result<Self, String> {
        Ok(Self::new(crate::device::request_device()?))
    }

    /// Compile kernels without naga's per-access buffer bounds checks and
    /// loop-bounding counters (`wgpu::ShaderRuntimeChecks::unchecked()`).
    ///
    /// Every kernel here indexes with explicit dims from its uniform params
    /// and is validated against the CPU reference, so the checks only cost
    /// throughput in the inner loops (Chrome's Tint clamps instead). Only
    /// affects pipelines compiled after the call. Also enabled by the
    /// `GEMMA_WGPU_TRUSTED_SHADERS=1` environment variable.
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
    }

    fn encoder<'a>(&self, st: &'a mut State) -> &'a mut wgpu::CommandEncoder {
        if st.enc.is_none() {
            st.enc = Some(self.device().create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("gemma-batch") }));
        }
        st.enc.as_mut().unwrap()
    }

    fn compute_pass<'a>(&self, st: &'a mut State) -> &'a mut wgpu::ComputePass<'static> {
        if st.pass.is_none() {
            let enc = self.encoder(st);
            let pass = enc
                .begin_compute_pass(&wgpu::ComputePassDescriptor { label: Some("gemma-ops"), timestamp_writes: None })
                .forget_lifetime();
            st.pass = Some(pass);
        }
        st.pass.as_mut().unwrap()
    }

    /// Record a buffer-to-buffer copy into the shared encoder (splits the pass).
    fn copy(&self, src: &wgpu::Buffer, src_off: u64, dst: &wgpu::Buffer, dst_off: u64, bytes: u64) {
        if bytes == 0 {
            return;
        }
        let mut st = self.st.borrow_mut();
        Self::end_pass(&mut st);
        self.encoder(&mut st).copy_buffer_to_buffer(src, src_off, dst, dst_off, bytes);
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
    fn alloc(&self, shape: &[usize]) -> GpuTensor {
        let n: usize = shape.iter().product();
        let want = (n as u64 * 4).max(16);
        let mut size = 256u64;
        while size < want {
            size <<= 1;
        }
        let mut st = self.st.borrow_mut();
        let buf = match st.pool.get_mut(&size).and_then(|l| l.pop()) {
            Some(b) => {
                st.pooled_bytes -= size;
                b
            }
            None => Arc::new(self.raw_buf(size, storage_usage(), "activation")),
        };
        if let Some(f) = st.frame.as_mut() {
            f.push((buf.clone(), true));
        }
        GpuTensor::new(buf, shape)
    }

    fn recycle(st: &mut State, buf: Arc<wgpu::Buffer>, pooled: bool) {
        let size = buf.size();
        if !pooled || st.pooled_bytes + size > POOL_MAX_BYTES {
            buf.destroy();
            return;
        }
        st.pool.entry(size).or_default().push(buf);
        st.pooled_bytes += size;
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
    pub fn bytes_buffer(&self, bytes: &[u8]) -> Arc<wgpu::Buffer> {
        Arc::new(self.upload_bytes(bytes, "packed-weight"))
    }

    /// Build a quant tensor from an already-uploaded GPU byte buffer.
    pub fn quant_tensor_from_gpu(&self, buffer: Arc<wgpu::Buffer>, scales: &[f32], shape: [usize; 2], bits: u32, signed_i8: bool, group_size: usize) -> GpuTensor {
        let per_byte = if signed_i8 { 1 } else { (8 / bits) as usize };
        let scale_buffer = Arc::new(self.upload_bytes(bytemuck::cast_slice(scales), "quant-scale"));
        GpuTensor {
            buffer,
            shape: shape.to_vec(),
            cap_rows: None,
            quant: Some(Arc::new(GpuQuant { scale_buffer, bits, signed_i8, group_size, per_byte })),
        }
    }

    /// Upload a u32 index buffer (token ids / positions); memoized per frame.
    fn u32_buf(&self, arr: &[u32]) -> Arc<wgpu::Buffer> {
        {
            let st = self.st.borrow();
            if let Some(b) = st.u32_memo.get(arr) {
                return b.clone();
            }
        }
        let buf = Arc::new(self.upload_bytes(bytemuck::cast_slice(arr), "u32-index"));
        self.st.borrow_mut().u32_memo.insert(arr.to_vec(), buf.clone());
        buf
    }

    /// Suballocate a uniform slot from the ring; contents are staged CPU-side
    /// and written in one `queue.write_buffer` per segment at flush.
    fn uniform_slot(&self, st: &mut State, bytes: &[u8]) -> UniformSlot {
        let need = (bytes.len() as u64).div_ceil(UNIFORM_ALIGN) * UNIFORM_ALIGN;
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
        let off = seg.off;
        seg.data[off as usize..off as usize + bytes.len()].copy_from_slice(bytes);
        seg.off += need;
        UniformSlot { seg: i, offset: off, size: bytes.len() as u64 }
    }

    fn pipeline<'a>(&self, st: &'a mut State, key: &str, wgsl: &str, constants: &[(&str, f64)]) -> &'a wgpu::ComputePipeline {
        if !st.pipelines.contains_key(key) {
            let desc = wgpu::ShaderModuleDescriptor { label: Some(key), source: wgpu::ShaderSource::Wgsl(wgsl.into()) };
            let module = if self.trusted {
                // SAFETY: these kernels never index past the buffers they are
                // bound to (all bounds come from explicit uniform dims) and
                // every loop is bounded by those dims; see `with_trusted_shaders`.
                unsafe { self.device().create_shader_module_trusted(desc, wgpu::ShaderRuntimeChecks::unchecked()) }
            } else {
                self.device().create_shader_module(desc)
            };
            let p = self.device().create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(key),
                layout: None,
                module: &module,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions { constants, zero_initialize_workgroup_memory: false },
                cache: None,
            });
            st.pipelines.insert(key.to_string(), p);
        }
        st.pipelines.get(key).unwrap()
    }

    fn dispatch(&self, key: &str, wgsl: &str, storage: &[&wgpu::Buffer], uniform: &[U], workgroups: [u32; 3], constants: &[(&str, f64)]) {
        let mut st = self.st.borrow_mut();
        let st = &mut *st;
        let ubytes = pack_uniform(uniform);
        let slot = self.uniform_slot(st, &ubytes);
        let pipeline = self.pipeline(st, key, wgsl, constants).clone();
        let mut entries: Vec<wgpu::BindGroupEntry> = storage
            .iter()
            .enumerate()
            .map(|(i, b)| wgpu::BindGroupEntry { binding: i as u32, resource: b.as_entire_binding() })
            .collect();
        entries.push(wgpu::BindGroupEntry {
            binding: storage.len() as u32,
            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                buffer: &st.rings[slot.seg].buf,
                offset: slot.offset,
                size: NonZeroU64::new(slot.size),
            }),
        });
        let bg = self.device().create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(key),
            layout: &pipeline.get_bind_group_layout(0),
            entries: &entries,
        });
        let pass = self.compute_pass(st);
        pass.set_pipeline(&pipeline);
        pass.set_bind_group(0, &bg, &[]);
        pass.dispatch_workgroups(workgroups[0], workgroups[1], workgroups[2]);
    }

    fn ew(&self, a: &GpuTensor, b: Option<&GpuTensor>, op: u32, k: f32) -> GpuTensor {
        let out = self.alloc(&a.shape);
        let n = a.size();
        self.dispatch(
            "ew",
            K::ELEMENTWISE,
            &[&a.buffer, &b.unwrap_or(a).buffer, &out.buffer],
            &[U::U(n as u32), U::U(op), U::F(k), U::U(0)],
            [(n as u32).div_ceil(256), 1, 1],
            &[],
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
        GpuTensor::new(Arc::new(self.upload_bytes(bytemuck::cast_slice(data), "f32-tensor")), shape)
    }

    fn quant_tensor(&self, q: QuantWeight) -> GpuTensor {
        let buffer = self.bytes_buffer(&q.qbytes);
        self.quant_tensor_from_gpu(buffer, &q.scales, q.shape, q.bits, q.signed_i8, q.group_size)
    }

    fn zeros(&self, shape: &[usize]) -> GpuTensor {
        let t = self.alloc(shape);
        let mut st = self.st.borrow_mut();
        Self::end_pass(&mut st);
        let bytes = (t.size() as u64 * 4).max(4);
        self.encoder(&mut st).clear_buffer(&t.buffer, 0, Some(bytes));
        t
    }

    fn reshape(&self, x: &GpuTensor, shape: &[usize]) -> GpuTensor {
        GpuTensor { buffer: x.buffer.clone(), shape: shape.to_vec(), cap_rows: None, quant: x.quant.clone() }
    }

    fn embed_rows(&self, table: &GpuTensor, ids: &[u32], scale: f32) -> GpuTensor {
        let d = table.shape[1];
        let t = ids.len();
        let out = self.alloc(&[t, d]);
        let idbuf = self.u32_buf(ids);
        let wg = ((t * d) as u32).div_ceil(256);
        if let Some(q) = &table.quant {
            let s_stride = d.div_ceil(q.group_size);
            self.dispatch(
                "embedq",
                K::EMBED_Q,
                &[&table.buffer, &idbuf, &q.scale_buffer, &out.buffer],
                &[U::U(t as u32), U::U(d as u32), U::U(q.bits), U::U(q.per_byte as u32), U::U(q.group_size as u32), U::U(s_stride as u32), U::F(scale), U::U(0)],
                [wg, 1, 1],
                &[],
            );
            return out;
        }
        self.dispatch("embed", K::EMBED, &[&table.buffer, &idbuf, &out.buffer], &[U::U(t as u32), U::U(d as u32), U::F(scale), U::U(0)], [wg, 1, 1], &[]);
        out
    }

    // weight = None runs the unweighted variant (the kernel still needs a valid
    // w binding of >= D elements, so bind x itself as a dummy).
    fn rms_norm(&self, x: &GpuTensor, w: Option<&GpuTensor>, eps: f32) -> GpuTensor {
        let d = *x.shape.last().unwrap();
        let rows = x.size() / d;
        let out = self.alloc(&x.shape);
        self.dispatch(
            "rmsnorm",
            K::RMSNORM,
            &[&x.buffer, &w.unwrap_or(x).buffer, &out.buffer],
            &[U::U(rows as u32), U::U(d as u32), U::F(eps), U::U(w.is_some() as u32)],
            [rows as u32, 1, 1],
            &[],
        );
        out
    }

    fn linear(&self, x: &GpuTensor, w: &GpuTensor) -> GpuTensor {
        let m = x.shape[0];
        let kd = x.shape[1];
        let n = w.shape[0];
        let out = self.alloc(&[m, n]);
        if let Some(q) = &w.quant {
            let bits = if q.signed_i8 { 8 } else { q.bits };
            let vals_per_word = (32 / bits) as usize;
            // Fast path: whole u32 words per lane, per-row scales. Covers every
            // real checkpoint shape; the scalar kernel remains for odd K / grouped scales.
            if kd % vals_per_word == 0 && q.group_size == kd && m <= 262140 {
                let mrows: u32 = if m == 1 { 1 } else { 4 };
                self.dispatch(
                    &format!("matmulq_b{bits}m{mrows}"),
                    K::MATMUL_Q,
                    &[&x.buffer, &w.buffer, &q.scale_buffer, &out.buffer],
                    &[U::U(m as u32), U::U(kd as u32), U::U(n as u32), U::U(bits), U::U((kd / vals_per_word) as u32), U::U(0), U::U(0), U::U(0)],
                    [(n as u32).div_ceil(32), (m as u32).div_ceil(mrows), 1],
                    &[("MROWS", mrows as f64), ("BITS", bits as f64)],
                );
                return out;
            }
            self.dispatch(
                "matmulq_scalar",
                K::MATMUL_Q_SCALAR,
                &[&x.buffer, &w.buffer, &q.scale_buffer, &out.buffer],
                &[U::U(m as u32), U::U(kd as u32), U::U(n as u32), U::U(q.bits), U::U(q.per_byte as u32), U::U(q.signed_i8 as u32), U::U(0), U::U(0)],
                [(n as u32).div_ceil(64), m as u32, 1],
                &[],
            );
            return out;
        }
        if m == 1 && n <= 65535 {
            // decode-time f32 matvec: one workgroup per output row, lanes stride K.
            self.dispatch("gemv", K::GEMV, &[&x.buffer, &w.buffer, &out.buffer], &[U::U(kd as u32), U::U(n as u32), U::U(0), U::U(0)], [n as u32, 1, 1], &[]);
            return out;
        }
        self.dispatch(
            "matmul",
            K::MATMUL,
            &[&x.buffer, &w.buffer, &out.buffer],
            &[U::U(m as u32), U::U(kd as u32), U::U(n as u32), U::U(0)],
            [(n as u32).div_ceil(16), (m as u32).div_ceil(16), 1],
            &[],
        );
        out
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
        let t = x.shape[0];
        let out = self.alloc(&x.shape);
        let posbuf = self.u32_buf(positions);
        self.dispatch(
            "rope",
            K::ROPE,
            &[&x.buffer, &posbuf, &out.buffer],
            &[U::U(t as u32), U::U(heads as u32), U::U(head_dim as u32), U::U(rotary_dim as u32), U::F(theta), U::F(0.0), U::F(0.0), U::F(0.0)],
            [((t * heads * head_dim) as u32).div_ceil(256), 1, 1],
            &[],
        );
        out
    }

    fn attention(&self, q: &GpuTensor, k: &GpuTensor, v: &GpuTensor, o: &AttnOpts) -> GpuTensor {
        let (t, hq, dh) = (q.shape[0], q.shape[1], q.shape[2]);
        let s = k.shape[0];
        let hkv = k.shape[1];
        let out = self.alloc(&[t, hq * dh]);
        let qp = self.u32_buf(o.q_pos);
        let kp = self.u32_buf(o.k_pos);
        self.dispatch(
            "attn",
            K::ATTENTION,
            &[&q.buffer, &k.buffer, &v.buffer, &qp, &kp, &out.buffer],
            &[
                U::U(t as u32), U::U(s as u32), U::U(hq as u32), U::U(hkv as u32),
                U::U(dh as u32), U::U(o.sliding_window), U::U(0), U::U(0),
                U::F(o.scale), U::F(0.0), U::F(0.0), U::F(0.0),
            ],
            [t as u32, hq as u32, 1],
            &[],
        );
        out
    }

    fn slice_rows(&self, x: &GpuTensor, start: usize, count: usize) -> GpuTensor {
        let dim = x.shape[1];
        let out = self.alloc(&[count, dim]);
        self.copy(&x.buffer, (start * dim * 4) as u64, &out.buffer, 0, (count * dim * 4) as u64);
        out
    }

    fn slice_cols(&self, x: &GpuTensor, offset: usize, width: usize) -> GpuTensor {
        let (rows, stride) = (x.shape[0], x.shape[1]);
        let out = self.alloc(&[rows, width]);
        self.dispatch(
            "slicecols",
            K::SLICE_COLS,
            &[&x.buffer, &out.buffer],
            &[U::U(rows as u32), U::U(stride as u32), U::U(offset as u32), U::U(width as u32)],
            [((rows * width) as u32).div_ceil(256), 1, 1],
            &[],
        );
        out
    }

    fn concat_rows(&self, a: Option<&GpuTensor>, b: &GpuTensor) -> GpuTensor {
        let Some(a) = a else {
            let out = self.alloc(&b.shape);
            self.copy(&b.buffer, 0, &out.buffer, 0, (b.size() * 4) as u64);
            return out;
        };
        let mut shape = a.shape.clone();
        shape[0] += b.shape[0];
        let out = self.alloc(&shape);
        self.copy(&a.buffer, 0, &out.buffer, 0, (a.size() * 4) as u64);
        self.copy(&b.buffer, 0, &out.buffer, (a.size() * 4) as u64, (b.size() * 4) as u64);
        out
    }

    /// Append `add` (`[rows, ...]`) to the KV tensor `existing` in place, growing
    /// a capacity buffer geometrically. Returns a view with the new logical row
    /// count (sharing the capacity buffer) plus the retired old buffer when
    /// growth reallocated. Turns the per-step O(S) cache copy into O(new rows).
    fn kv_append(&self, existing: Option<&GpuTensor>, add: &GpuTensor) -> KvAppend<GpuTensor> {
        let rows = add.shape[0];
        let dim = add.size() / rows;
        let Some(existing) = existing else {
            let cap = rows.max(256);
            let buf = Arc::new(self.raw_buf((cap * dim * 4) as u64, storage_usage(), "kv-cache"));
            if let Some(f) = self.st.borrow_mut().frame.as_mut() {
                f.push((buf.clone(), false));
            }
            self.copy(&add.buffer, 0, &buf, 0, (add.size() * 4) as u64);
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
            let buf = Arc::new(self.raw_buf((cap * dim * 4) as u64, storage_usage(), "kv-cache"));
            if let Some(f) = self.st.borrow_mut().frame.as_mut() {
                f.push((buf.clone(), false));
            }
            self.copy(&existing.buffer, 0, &buf, 0, (len * dim * 4) as u64);
            dst = GpuTensor::new(buf, &existing.shape);
            dst.cap_rows = Some(cap);
            dead = Some(existing.clone());
        }
        self.copy(&add.buffer, 0, &dst.buffer, (len * dim * 4) as u64, (add.size() * 4) as u64);
        let mut shape = add.shape.clone();
        shape[0] = len + rows;
        let mut view = GpuTensor::new(dst.buffer.clone(), &shape);
        view.cap_rows = dst.cap_rows;
        KvAppend { tensor: view, dead }
    }

    fn readback(&self, t: &GpuTensor) -> Vec<f32> {
        let bytes = (t.size() * 4) as u64;
        let staging = self.raw_buf(bytes, wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ, "staging");
        self.copy(&t.buffer, 0, &staging, 0, bytes);
        self.flush();
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
        staging.destroy();
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
        self.flush_inner(&mut st);
        let keep_ptrs: std::collections::HashSet<*const wgpu::Buffer> = keep.iter().map(|t| Arc::as_ptr(&t.buffer)).collect();
        if let Some(frame) = st.frame.take() {
            for (buf, pooled) in frame {
                if !keep_ptrs.contains(&Arc::as_ptr(&buf)) {
                    Self::recycle(&mut st, buf, pooled);
                }
            }
        }
        for t in also_free {
            if !keep_ptrs.contains(&Arc::as_ptr(&t.buffer)) {
                t.buffer.destroy();
            }
        }
        st.u32_memo.clear();
    }

    fn free_tensor(&self, t: GpuTensor) {
        t.buffer.destroy();
    }
}
