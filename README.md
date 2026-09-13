# Gemma 4 E2B · WebGPU kernels (from scratch)

A from-scratch, in-browser inference engine for **Gemma 4 E2B** (`google/gemma-4-E2B-it-qat-mobile-transformers`), built to reproduce the Hugging Face Space
[`webml-community/gemma-4-webgpu-kernels`](https://huggingface.co/spaces/webml-community/gemma-4-webgpu-kernels).

Every compute kernel is hand-written WGSL — there is **no inference library** (no
transformers.js, no ONNX Runtime). The transformer, tokenizer, safetensors loader,
KV cache, sampler, and chat template are all implemented here.

⚡ Landing page + streaming chat UI included, with a Three.js "compute grid" hero.

## Run it

```bash
npm install        # only needed for the Node test-suite (headless WebGPU via Dawn)
npm run serve      # static server on http://localhost:8080
```

Open <http://localhost:8080>, click **Enter**, choose a backend, and **Load model**.
The weights stream from the HF Hub on first load (large download). Requires a
WebGPU-capable browser (Chrome/Edge 113+, Safari 18+).

## Verify it

The whole engine is backend-agnostic: model code runs on either a **CPU reference
backend** (plain JS) or the **WebGPU backend** (WGSL). That makes the math testable
without a GPU, and lets us assert GPU≡CPU parity.

```bash
npm test
```

- `test/model.test.mjs` — runs the full Gemma-4 graph on CPU; asserts incremental
  KV-cache decoding equals a full-sequence forward (zero diff), checks sliding
  window, partial rotary, MQA, PLE, soft-cap, sampler.
- `test/tokenizer.test.mjs` — Unigram/BPE encode-decode, byte fallback, specials, chat template.
- `test/weights.test.mjs` — safetensors F32/F16/BF16 decode; 2/4/8-bit pack/unpack & dequant.
- `test/hub.test.mjs` — exact byte-range handling and cancellation propagation.
- `test/stream.test.mjs` — chunked tensor assembly, truncation detection, and streamed weights.
- `test/gpu.test.mjs` — **every WGSL kernel vs. the CPU reference** (~1e-6), plus a
  full GPU forward-pass parity check, on a headless Dawn device.
- `test/browser.html` — the same parity checks running in your actual browser
  (open `http://localhost:8080/test/browser.html`). Confirmed on Chrome + Apple Metal-3.
- `test/browser-quant.html` — quant kernels + the **full streaming quantized model**
  (GPU≡CPU) in the browser, i.e. the real load path on synthetic real-format data.

`npm run test:gpu-integration` runs the GPU integration tests in Node (Dawn). Note:
headless Dawn occasionally segfaults on tiny buffers — a known Node-binding quirk,
not a logic bug; the browser tests are the authoritative GPU check.

## Running the real weights

The checkpoint is a single 2.46 GB `model.safetensors` in Google's mobile QAT
format (`wNa8o8`): weights are packed **U8 (2/4-bit)** or **I8 (8-bit)** with an
f32 `*_scale` sibling (per output row, or per-layer-block for the per-layer
embeddings), under `model.language_model.*` names. Two things make running it
in-browser non-trivial, both handled here:

- **Streaming load.** A 2.46 GB `ArrayBuffer` can't be allocated in a tab
  (`RangeError: Array buffer allocation failed`). `stream-loader.js` reads the
  header via range requests, then streams the body and assembles **one tensor at
  a time**, uploading each straight to the GPU and freeing the JS bytes — peak JS
  heap stays at the largest single tensor (~1.17 GB), and the unused vision/audio
  towers are skipped.
- **Weights stay quantized on the GPU.** Dequantizing to f32 would need ~16 GB.
  Instead packed weights live in GPU buffers and are dequantized **inside** the
  matmul/embed kernels (`MATMUL_Q`, `EMBED_Q`), so GPU memory ≈ the 2.46 GB
  download.

Validated in real Chrome (Apple Metal-3) via `test/browser-quant.html`: the full
quantized model, loaded through the streaming path, matches the CPU reference to
~1e-6.

> Fidelity note: Google hasn't published the exact `wNa8o8` packing (zero-point /
> bit order) or the precise per-layer-embedding wiring. Those are implemented
> best-effort and isolated to `dequant.js`/`weights.js`/`stream-loader.js` and the
> PLE block in `gemma4.js`; everything else is independent of those choices.

## Rust port

`rust/` contains a native port of this engine on [`wgpu`](https://wgpu.rs):
the same backend-agnostic graph and the same CPU-vs-GPU parity test-suite,
plus a CLI (`download` / `chat` / `generate` / `bench`). Its kernels started
as byte-for-byte copies of `src/engine/kernels.js` and have since been
reworked for throughput (fused ops, grouped split-K attention, a tiled prefill
GEMM, a cached-bind-group host path): 3.2 ms/token prefill and 16 ms/token
decode (62 tok/s) on an M4, vs ~17 / ~26 in the browser. See
[rust/README.md](rust/README.md); those kernel changes have not been ported
back to the JS engine.

## What's implemented

The real Gemma-4-E2B is a Gemma-3n-class architecture. From its `config.json`:

| Feature | Detail |
|---|---|
| Layers / hidden | 35 / 1536 |
| Attention | MQA: 8 query heads, 1 KV head; head_dim 256 (local) / 512 (global) |
| Layer types | sliding (window 512) + full attention every 5th layer |
| RoPE | dual base: θ=1e6 global (partial rotary 0.25) / θ=1e4 local |
| MLP | GeGLU, `gelu_pytorch_tanh`, intermediate 6144 |
| Norms | RMSNorm with `(1+w)` gain; pre+post around attn & ffn; per-head q/k norm |
| Per-Layer Embeddings | gate → gelu → ×ple → project → norm → add, 256-dim/layer |
| KV sharing | last 20 layers reuse same-type earlier KV |
| Quant | mixed 2/4/8-bit QAT (symmetric, per-group dequant) |
| Output | tied-off lm_head + final logit soft-cap 30 |

See [ARCHITECTURE.md](ARCHITECTURE.md) for the full data flow and where any
assumptions live.

## Layout

```
index.html              landing + chat UI
landing.js              Three.js hero animation
serve.mjs               static dev server
src/
  main.js               UI ↔ engine wiring (streaming)
  engine/
    device.js           WebGPU adapter/device
    kernels.js          all WGSL compute shaders
    backend-webgpu.js   GPU op dispatch (mirrors the CPU API)
    backend-cpu.js      CPU reference implementation of every op
    safetensors.js      header parse + F32/F16/BF16 decode
    dequant.js          2/4/8-bit symmetric grouped dequant
    weights.js          name → backend tensor (non-streaming / tests)
    stream-loader.js    stream 2.46 GB → one tensor at a time → GPU
    hub.js              HF Hub fetch (range + streaming progress)
    gemma.js            Gemma4Mobile: load + streaming generate()
  model/
    config.js           config.json → flat, layer-aware config
    gemma4.js           the decoder forward graph (backend-agnostic)
    kvcache.js          per-layer cache with sharing
    sampler.js          greedy / temperature / top-k / top-p
  tokenizer/
    tokenizer.js        Unigram + BPE from tokenizer.json
    template.js         Gemma chat template
test/                   node + browser test harnesses
reference/              the original Space files, for comparison
```

## Notes & honesty

- Activations are **f32**; weights stay **quantized** on the GPU and are
  dequantized in-kernel (unpack loops are specialized per bit-width via pipeline
  overrides). The next optimization is f16 activations — the architecture is
  unchanged by it.
- Attention is a tiled online-softmax kernel (128 keys per tile, tile-level
  skips keep sliding-window layers O(window) at decode). A forward pass records
  into a single command submission; activations are pooled across steps and the
  KV cache appends in place (see ARCHITECTURE.md).
- `test/bench.mjs` / `test/bench.html` benchmark decode/prefill at real model
  shapes with synthetic weights (no download). On an M4 MacBook (browser,
  full 262k vocab): ~26 ms/token decode, ~17 ms/token prefill — vs 215/71
  before the batching + kernel rework. The Node Dawn binding (`webgpu@0.4`)
  can segfault on multi-GiB runs; use `npm run bench` (quarter vocab) there,
  or the browser page for full size.
- The QAT packing and several model conventions were verified against the original
  working implementation (`reference/gemma-4-e2b.js`) and the real checkpoint header
  (`reference/header.json`): sub-byte weights are OFFSET-BINARY
  (`value = (raw − 2^(bits−1)) · scale`), norm weights store the full multiplier
  (plain `w`, not HF-Gemma's `1+w`), attention uses scale 1.0 (no `1/√head_dim` —
  the per-head q-norm covers it), v gets an unweighted per-head RMSNorm, every
  layer ends with a learned `layer_scalar` multiply, and the config's
  `final_logit_softcapping` is never applied. Getting any of these wrong produces
  fluent-speed gibberish while all CPU/GPU parity tests still pass — the parity
  suite validates consistency, not checkpoint fidelity.
- Validated end-to-end: the real 2.46 GB checkpoint generates coherent chat text
  in-browser at ~20-30 tok/s on an M4.
