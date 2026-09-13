# gemma-wgpu · Gemma 4 E2B on wgpu, in Rust

A Rust port of the browser engine in the parent directory. Same design: every
compute kernel is hand-written WGSL, the transformer, tokenizer, safetensors
loader, KV cache, sampler and chat template are all implemented here, and
there is no inference library underneath. The kernels started as byte-identical
extracts of `../src/engine/kernels.js` and have since been reworked for
throughput (see "Kernel design" below); the graph, the packing conventions and
the CPU-vs-GPU parity test-suite are shared with the browser engine.

The host side runs on [`wgpu`](https://wgpu.rs) 30 (Metal on macOS, Vulkan /
DX12 elsewhere) instead of the browser's WebGPU, so it runs natively from a
terminal with the 2.46 GB checkpoint memory-mapped from disk.

## Build, test, run

```bash
cd rust
cargo test                      # CPU tests + wgpu≡CPU parity on your GPU (skips GPU tests with no adapter)
cargo build --release
./target/release/gemma-wgpu info                     # which adapter will be used
./target/release/gemma-wgpu bench --kernels          # real E2B shapes, synthetic weights, no download
./target/release/gemma-wgpu bench --backend cpu --vdiv 8 --prefill 8 --decode 3   # the threaded CPU reference
```

Running the real model:

```bash
# The repo is gated: accept the license on huggingface.co, then
export HF_TOKEN=hf_...
./target/release/gemma-wgpu download --dir models/gemma-4-e2b      # 2.46 GB
./target/release/gemma-wgpu chat --dir models/gemma-4-e2b
./target/release/gemma-wgpu generate --dir models/gemma-4-e2b --prompt "Explain RMSNorm in two sentences." --temperature 0
```

`chat` is a REPL (`/reset`, `/quit`); both commands stream tokens and print
prefill / decode timing. `--backend cpu` runs the same model on the reference
backend (threaded across cores, still ~100x slower than the GPU — for
debugging only; `GEMMA_CPU_THREADS=1` gives the sequential baseline).

## Layout

```
src/
  kernels/*.wgsl      the 16 compute shaders
  kernels.rs          include_str! bindings + per-kernel docs
  backend.rs          Backend trait: the op surface the model is written against (incl. fused ops)
  backend_cpu.rs      plain-Rust reference implementation of every op (row-parallel across cores)
  backend_wgpu.rs     wgpu implementation (single pass per step, bind-group cache, uniform ring, buffer pool, KV growth)
  device.rs           adapter/device acquisition with large-buffer limits
  safetensors.rs      mmap reader, F32/F16/BF16 decode, in-memory builder for tests
  dequant.rs          2/4/8-bit pack/unpack + dequant helpers
  weights.rs          logical name -> on-disk tensor (+ scale), lazy upload, cache
  config.rs           config.json -> flat, layer-aware config (quant rules via fancy-regex)
  model.rs            the Gemma-4 decoder graph, backend-agnostic, weights resolved once per layer
  kvcache.rs          per-layer cache with sharing and in-place growth
  sampler.rs          greedy / temperature / top-k / top-p / repetition penalty
  tokenizer.rs        Unigram (Viterbi) + BPE from tokenizer.json, streaming decoder
  template.rs         Gemma-4 chat template
  engine.rs           Gemma4Mobile: load a model dir, stream generate()
  hub.rs              Hugging Face download (HF_TOKEN aware)
  main.rs             CLI: download / chat / generate / bench / info
tests/
  common/mod.rs       synthetic + real-format tiny checkpoints, deterministic RNGs
  model.rs            CPU graph: incremental KV decode == full-sequence forward, sampler
  tokenizer.rs        encode/decode, byte fallback, specials, BPE, chat template
  weights.rs          safetensors dtypes, pack/unpack, dequant, name resolution + bit widths
  gpu.rs              every kernel vs CPU (~1e-7), quant kernels + every MATMUL_Q tile/epilogue
                      variant, grouped split-K attention at real head shapes, fused ops,
                      full model, real format
  engine.rs           load a tiny real-format checkpoint from disk and stream a reply
```

## Kernel design

| kernel | does | parallelism |
|---|---|---|
| `MATMUL_Q` | decode / short-batch dequant matmul (2/4/8-bit), per-row scales; `MODE` fuses the GeGLU (gate + up + gelu·up in one pass, second weight bound) and the PLE gate (gelu·slice) epilogues | LANES lanes per output row (8..64, chosen by N so narrow k/v projections still fill the GPU), vec4 activations, word loop unrolled 4× with loads hoisted; sub-byte values unpacked with one XOR per word + a signed bit-field extract per value |
| `MATMUL_Q_TILED` | prefill GEMM (M ≥ 16), same `BITS`/`MODE` | 64×64 output tiles, K in chunks of 32; A and the dequantized W tile staged in workgroup memory, 4×4 register blocks (2 vec4 loads per 16 MACs); each packed value unpacked once per 64 activation rows |
| `MATMUL_Q_SCALAR` | shape-agnostic fallback (odd K / grouped scales) | 1 thread/output |
| `MATMUL` / `GEMV` | f32 paths (`per_layer_model_projection`) | 16×16 tiles / 1 workgroup per row |
| `ATTN_SPLIT` + `ATTN_COMBINE` | grouped split-K online-softmax attention (causal + sliding window) | 1 workgroup per (query, **kv head**, key split) serving the whole GQA group — K/V read once per kv head, 8× less traffic for MQA; the host trims the key range to what any query can see (sliding layers stay O(window) at any context length) and splits it over up to 16 workgroups so decode fills the GPU; partials `(m, l, acc)` merged by the combine kernel |
| `ATTENTION` | generic per-(query, head) kernel | fallback for shapes the grouped kernel does not cover |
| `RMSNORM` | row normalize; flags fuse `+ residual` and `· layer_scalar` | every Gemma-4 post-norm + residual (+ PLE scalar) is one dispatch |
| `RMSNORM_ROPE` | per-head q/k norm fused with partial RoPE | 1 workgroup per (token, head) |
| `EMBED` / `EMBED_Q` / `ROPE` / `ELEMENTWISE` / `SLICE_COLS` / `COPY` | gathers, rope, elementwise, slices; `COPY` replaces `copyBufferToBuffer` for KV appends and row slices so a step never leaves its compute pass | 1 thread/element |

Host side (`backend_wgpu.rs`): a forward pass records into one compute pass in
one command buffer. Every kernel has an explicit bind group layout whose
uniform binding takes a dynamic offset into a 256-byte-slot ring, so bind
groups depend only on (kernel, storage buffers) and are cached — activation
buffers are recycled through a pow2 pool and index uploads come from the same
pool, so steady-state decode creates no wgpu objects at all. Nothing on the
per-dispatch path allocates. A decode step is ~600 dispatches (down from ~950)
after fusing gate/up/GeGLU, post-norm/residual/layer-scalar, PLE gate/slice
and q/k norm/RoPE.

## What differs from the browser engine

- **Loading.** No streaming assembler: the checkpoint is `mmap`ed and each text
  tensor is uploaded straight from the map (vision/audio towers are never
  touched). Peak host memory is the page cache, not a JS heap.
- **Synchronous API.** `forward()` blocks on the logits readback; `generate()`
  returns an `Iterator<Item = Step>` instead of an async generator.
- **Runtime checks.** naga (wgpu's shader compiler) can insert per-access
  buffer bounds checks and loop-bounding counters; Chrome's Tint clamps
  indices instead. The kernels only ever index with explicit uniform dims and
  the parity suite runs in both modes, so by default they are compiled with
  `ShaderRuntimeChecks::unchecked()`. `--checked-shaders` (or
  `GEMMA_WGPU_TRUSTED_SHADERS=0`) keeps the checks; they cost ~2x at decode and
  ~6x at prefill because they land in the tiled GEMM's inner loops.
- **Backend trait instead of duck typing.** `Backend` has an associated
  `Tensor` type; `Gemma4Model<B, W>` is generic over the backend and the
  weight source, so CPU and GPU share one graph and one test-suite. The fused
  ops (`linear_geglu`, `add_rms_norm`, `linear_gelu_mul_cols`,
  `head_norm_rope`) have default implementations that compose the primitives,
  so the CPU backend stays the plain reference.
- **Kernels.** The browser engine still runs the original per-head attention,
  4-row-tile matmul and unfused ops; the reworked kernels above are Rust-only.

Everything else — the `wNa8o8` packing conventions, the unweighted v-norm,
`layer_scalar`, the unapplied logit soft-cap, in-place KV growth, per-frame
buffer recycling — is shared, and the `ARCHITECTURE.md` in the parent
directory describes the graph this crate runs.

## Numbers (Apple M4, Metal, synthetic weights at real shapes, full 262k vocab)

`bench --prefill 64 --decode 24`, median per token:

| | prefill (64 tok) | decode |
|---|---|---|
| before this rework, `--trusted-shaders` | 26.5 ms/token | 33.4 ms/token (30 tok/s) |
| now, default (trusted) | **3.2 ms/token** | **16.0 ms/token (62 tok/s)** |
| now, `--checked-shaders` | 20.1 ms/token | 29.0 ms/token |

Isolated kernels (`bench --kernels`, ms/op): lm_head 2-bit [262144×1536]
4.1 → 2.1; MLP gate 4-bit [6144×1536] 0.123 → 0.059; 2-bit prefill gate
[12288×1536] at M=64 → 1.67; attention Dh256 at S=512 0.223 → 0.060, and
0.070 at S=8192 with the 512 window (was O(S)); global-layer attention Dh512
at S=8192 0.60. At decode the remaining budget is ~75% quantized matmul
(~50-60 GB/s effective on the 2-bit layers) and ~600 dispatches of a few µs
each. Per-dispatch latency on Metal (`add [1,1536] chained`) is ~6 µs.

CPU reference (`bench --backend cpu --vdiv 8`, 10 cores): decode 5.7 s →
1.4 s per token, prefill 1.43 s → 0.33 s per token vs `GEMMA_CPU_THREADS=1`.

GPU≡CPU parity on the tiny model is ~1.5e-6 max abs diff for prefill and
incremental decode; individual kernels ≤ 1.4e-5 (I8 matmul) and mostly ~1e-7.
