//! Backend abstraction.
//!
//! The model never touches wgpu directly. It calls the [`Backend`] op surface,
//! which two implementations provide identically:
//!
//! - [`crate::backend_cpu::CpuBackend`]: plain-Rust reference. Slow, exact, debuggable.
//! - [`crate::backend_wgpu::WgpuBackend`]: records WGSL compute passes; tensors are GPU buffers.
//!
//! Both share the same tensor-shape semantics, so `model::Gemma4Model` is written
//! once and runs on either. That is what lets the test-suite assert GPU ≡ CPU.

/// A weight kept in its packed, quantized form.
///
/// `qbytes` is the raw payload (U8 sub-byte packed LSB-first, or native I8);
/// `scales` is one f32 per (row, group). `shape` is the logical `[N, K]`.
#[derive(Clone, Debug)]
pub struct QuantWeight {
    pub qbytes: Vec<u8>,
    pub scales: Vec<f32>,
    pub shape: [usize; 2],
    pub bits: u32,
    pub signed_i8: bool,
    pub group_size: usize,
}

impl QuantWeight {
    /// Values per byte for the packed layout (1 for I8).
    pub fn per_byte(&self) -> usize {
        if self.signed_i8 { 1 } else { (8 / self.bits) as usize }
    }

    /// Bytes per packed row.
    pub fn row_bytes(&self) -> usize {
        let k = self.shape[1];
        if self.signed_i8 { k } else { k / self.per_byte() }
    }

    /// Scale entries per row.
    pub fn scale_stride(&self) -> usize {
        self.shape[1].div_ceil(self.group_size)
    }

    /// Dequantize a single element `[n, k]`, matching the WGSL unpack exactly.
    #[inline]
    pub fn deq(&self, n: usize, k: usize) -> f32 {
        let qv: i32 = if self.signed_i8 {
            self.qbytes[n * self.row_bytes() + k] as i8 as i32
        } else {
            let per_byte = self.per_byte();
            let b = self.qbytes[n * self.row_bytes() + k / per_byte] as u32;
            let shift = ((k % per_byte) as u32) * self.bits;
            // offset binary: value = raw - zeroPoint, zeroPoint = 2^(bits-1)
            (((b >> shift) & ((1u32 << self.bits) - 1)) as i32) - (1i32 << (self.bits - 1))
        };
        qv as f32 * self.scales[n * self.scale_stride() + k / self.group_size]
    }
}

/// Options for [`Backend::attention`].
#[derive(Clone, Debug)]
pub struct AttnOpts<'a> {
    pub q_pos: &'a [u32],
    pub k_pos: &'a [u32],
    pub scale: f32,
    /// 0 = no sliding window (full causal attention).
    pub sliding_window: u32,
    /// 0 = no attention soft-capping (Gemma-3/4 dropped it).
    pub attn_softcap: f32,
}

/// Result of [`Backend::kv_append`].
pub struct KvAppend<T> {
    /// View with the new logical row count.
    pub tensor: T,
    /// The retired old buffer when growth reallocated (free at a safe point).
    pub dead: Option<T>,
}

/// Shape accessors every backend tensor provides.
pub trait TensorShape {
    fn shape(&self) -> &[usize];
    fn size(&self) -> usize {
        self.shape().iter().product()
    }
}

/// The op surface the Gemma-4 graph is written against.
pub trait Backend {
    type Tensor: Clone + TensorShape;

    fn name(&self) -> &str;

    // --- construction ---------------------------------------------------
    fn tensor(&self, data: &[f32], shape: &[usize]) -> Self::Tensor;
    fn quant_tensor(&self, q: QuantWeight) -> Self::Tensor;
    fn zeros(&self, shape: &[usize]) -> Self::Tensor;
    /// Reinterpret with a new shape (no copy).
    fn reshape(&self, x: &Self::Tensor, shape: &[usize]) -> Self::Tensor;

    // --- ops --------------------------------------------------------------
    /// `out[i,:] = table[ids[i],:] * scale`; table is `[V, D]` (f32 or quantized).
    fn embed_rows(&self, table: &Self::Tensor, ids: &[u32], scale: f32) -> Self::Tensor;
    /// `y = x / sqrt(mean(x²)+eps) * w`, per row. `w = None` runs the unweighted variant.
    fn rms_norm(&self, x: &Self::Tensor, w: Option<&Self::Tensor>, eps: f32) -> Self::Tensor;
    /// `y = x · Wᵀ`; x `[T, K]`, W `[N, K]` (PyTorch nn.Linear layout).
    fn linear(&self, x: &Self::Tensor, w: &Self::Tensor) -> Self::Tensor;
    fn add(&self, a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;
    fn mul(&self, a: &Self::Tensor, b: &Self::Tensor) -> Self::Tensor;
    fn scale(&self, a: &Self::Tensor, s: f32) -> Self::Tensor;
    /// `a * s[0]` (learned `layer_scalar`).
    fn scale_by_tensor(&self, a: &Self::Tensor, s: &Self::Tensor) -> Self::Tensor;
    fn gelu_tanh(&self, a: &Self::Tensor) -> Self::Tensor;
    /// `gelu(gate) * up`
    fn geglu(&self, gate: &Self::Tensor, up: &Self::Tensor) -> Self::Tensor;
    /// `cap * tanh(x / cap)`
    fn softcap(&self, a: &Self::Tensor, cap: f32) -> Self::Tensor;
    /// Half-split partial RoPE on `[T, heads, head_dim]`; rotates the first `rotary_dim` dims.
    fn rope(&self, x: &Self::Tensor, positions: &[u32], theta: f32, head_dim: usize, heads: usize, rotary_dim: usize) -> Self::Tensor;
    /// q `[T,Hq,Dh]`, k `[S,Hkv,Dh]`, v `[S,Hkv,Dh]` → `[T, Hq*Dh]`.
    fn attention(&self, q: &Self::Tensor, k: &Self::Tensor, v: &Self::Tensor, opts: &AttnOpts) -> Self::Tensor;
    /// Rows `[start, start+count)` of a `[rows, dim]` tensor.
    fn slice_rows(&self, x: &Self::Tensor, start: usize, count: usize) -> Self::Tensor;
    /// `[:, offset:offset+width]` of a `[rows, stride]` tensor.
    fn slice_cols(&self, x: &Self::Tensor, offset: usize, width: usize) -> Self::Tensor;
    /// Concatenate along dim 0. `a = None` copies `b`.
    fn concat_rows(&self, a: Option<&Self::Tensor>, b: &Self::Tensor) -> Self::Tensor;
    /// Append `add` rows to the KV tensor `existing` (in place where possible).
    fn kv_append(&self, existing: Option<&Self::Tensor>, add: &Self::Tensor) -> KvAppend<Self::Tensor>;

    // --- sync / lifecycle ------------------------------------------------
    /// Blocking read of a tensor's f32 contents (submits pending work first).
    fn readback(&self, t: &Self::Tensor) -> Vec<f32>;
    /// Submit everything recorded so far (no-op on CPU).
    fn flush(&self) {}
    /// Bracket a forward pass: intermediates allocated inside are recycled at `end_frame`.
    fn begin_frame(&self) {}
    /// Recycle frame allocations except `keep`, and free `also_free`.
    fn end_frame(&self, _keep: &[Self::Tensor], _also_free: Vec<Self::Tensor>) {}
    /// Free a tensor the caller guarantees is no longer in use by the GPU.
    fn free_tensor(&self, _t: Self::Tensor) {}
}
