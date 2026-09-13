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
/// Dequant-on-the-fly matmul (2/4/8-bit): vec4 activations, XOR sign-trick
/// unpack, `BITS`/`MROWS`/`MODE` overrides (MODE fuses GeGLU / gelu·slice epilogues).
pub const MATMUL_Q: &str = include_str!("kernels/matmul_q.wgsl");
/// Prefill GEMM (M >= 16): 64×64 output tiles, A and dequantized W staged in
/// workgroup memory, 4×4 register blocks; same `BITS`/`MODE` overrides.
pub const MATMUL_Q_TILED: &str = include_str!("kernels/matmul_q_tiled.wgsl");
/// Shape-agnostic quantized matmul fallback (odd K), byte-at-a-time unpack.
pub const MATMUL_Q_SCALAR: &str = include_str!("kernels/matmul_q_scalar.wgsl");
/// Quantized embedding gather + dequant + scale (grouped scales).
pub const EMBED_Q: &str = include_str!("kernels/embed_q.wgsl");
/// f32 embedding gather + scale.
pub const EMBED: &str = include_str!("kernels/embed.wgsl");
/// Column slice `out[r, 0:W] = in[r, offset:offset+W]`.
pub const SLICE_COLS: &str = include_str!("kernels/slice_cols.wgsl");
/// Offset copy `out[dst + i] = in[src + i]` as a dispatch (keeps the pass open).
pub const COPY: &str = include_str!("kernels/copy.wgsl");
/// RMSNorm, one workgroup per row, tree reduce; flags fuse `hasW`, `+ residual`, `· s[0]`.
pub const RMSNORM: &str = include_str!("kernels/rmsnorm.wgsl");
/// Elementwise add/mul/scale/gelu/geglu/softcap/scale-by-tensor selected by `op`.
pub const ELEMENTWISE: &str = include_str!("kernels/elementwise.wgsl");
/// Half-split partial RoPE.
pub const ROPE: &str = include_str!("kernels/rope.wgsl");
/// Per-head RMSNorm fused with partial RoPE (q/k path), one workgroup per (token, head).
pub const RMSNORM_ROPE: &str = include_str!("kernels/rmsnorm_rope.wgsl");
/// Generic tiled online-softmax attention (causal + sliding window, GQA), one
/// workgroup per (query, head), 128 keys per tile. Fallback for shapes the
/// grouped kernel does not cover.
pub const ATTENTION: &str = include_str!("kernels/attention.wgsl");
/// Grouped split-K attention: one workgroup per (query, kv head, key split)
/// serving the whole GQA group; partials merged by [`ATTN_COMBINE`].
pub const ATTN_SPLIT: &str = include_str!("kernels/attention_split.wgsl");
/// Merges [`ATTN_SPLIT`] partials `(m, l, acc)` into normalized outputs.
pub const ATTN_COMBINE: &str = include_str!("kernels/attention_combine.wgsl");
