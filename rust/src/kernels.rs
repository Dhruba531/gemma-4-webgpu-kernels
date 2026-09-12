//! WGSL compute kernels for the Gemma-4 forward pass, embedded verbatim.
//!
//! Each constant is a complete shader module with a `main` entry point. They
//! mirror, op-for-op, the CPU reference in `backend_cpu.rs`, so GPU and CPU
//! outputs agree. Everything is f32 (no shader-f16 feature needed).
//!
//! Binding convention per dispatch: storage buffers first (read, then
//! read_write), then a single uniform "params" buffer last.

/// Tiled f32 matmul `C[M,N] = A[M,K] · Bᵀ`, 16×16 shared-memory tiles.
pub const MATMUL: &str = include_str!("kernels/matmul.wgsl");
/// f32 matvec for `M == 1`: one workgroup per output row, lanes stride K.
pub const GEMV: &str = include_str!("kernels/gemv.wgsl");
/// Dequant-on-the-fly matmul (2/4/8-bit), whole-u32 loads, `BITS`/`MROWS` overrides.
pub const MATMUL_Q: &str = include_str!("kernels/matmul_q.wgsl");
/// Shape-agnostic quantized matmul fallback (odd K), byte-at-a-time unpack.
pub const MATMUL_Q_SCALAR: &str = include_str!("kernels/matmul_q_scalar.wgsl");
/// Quantized embedding gather + dequant + scale (grouped scales).
pub const EMBED_Q: &str = include_str!("kernels/embed_q.wgsl");
/// f32 embedding gather + scale.
pub const EMBED: &str = include_str!("kernels/embed.wgsl");
/// Column slice `out[r, 0:W] = in[r, offset:offset+W]`.
pub const SLICE_COLS: &str = include_str!("kernels/slice_cols.wgsl");
/// RMSNorm, one workgroup per row, tree reduce; `hasW = 0` is the unweighted variant.
pub const RMSNORM: &str = include_str!("kernels/rmsnorm.wgsl");
/// Elementwise add/mul/scale/gelu/geglu/softcap/scale-by-tensor selected by `op`.
pub const ELEMENTWISE: &str = include_str!("kernels/elementwise.wgsl");
/// Half-split partial RoPE.
pub const ROPE: &str = include_str!("kernels/rope.wgsl");
/// Tiled online-softmax attention (causal + sliding window, GQA), 128 keys per tile.
pub const ATTENTION: &str = include_str!("kernels/attention.wgsl");
