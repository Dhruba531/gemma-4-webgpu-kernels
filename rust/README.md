# gemma-wgpu · Gemma 4 E2B on wgpu, in Rust

A Rust port of the browser engine in the parent directory. Same design, same
kernels: every compute kernel is hand-written WGSL (the `.wgsl` files under
`src/kernels/` are byte-identical to `../src/engine/kernels.js`), the
transformer, tokenizer, safetensors loader, KV cache, sampler and chat template
are all implemented here, and there is no inference library underneath.

The host side runs on [`wgpu`](https://wgpu.rs) 30 (Metal on macOS, Vulkan /
DX12 elsewhere) instead of the browser's WebGPU, so it runs natively from a
terminal with the 2.46 GB checkpoint memory-mapped from disk.

## Build, test, run

```bash
cd rust
cargo test                      # CPU tests + wgpu≡CPU parity on your GPU (skips GPU tests with no adapter)
cargo build --release
./target/release/gemma-wgpu info                     # which adapter will be used
./target/release/gemma-wgpu bench --trusted-shaders  # real E2B shapes, synthetic weights, no download
```

Running the real model:

```bash
# The repo is gated: accept the license on huggingface.co, then
export HF_TOKEN=hf_...
./target/release/gemma-wgpu download --dir models/gemma-4-e2b      # 2.46 GB
./target/release/gemma-wgpu chat --dir models/gemma-4-e2b --trusted-shaders
./target/release/gemma-wgpu generate --dir models/gemma-4-e2b --prompt "Explain RMSNorm in two sentences." --temperature 0
```

`chat` is a REPL (`/reset`, `/quit`); both commands stream tokens and print
prefill / decode timing. `--backend cpu` runs the same model on the reference
backend (seconds per token — for debugging only).

## Layout

```
src/
  kernels/*.wgsl      the 11 compute shaders (verbatim from the JS engine)
  kernels.rs          include_str! bindings + per-kernel docs
  backend.rs          Backend trait: the op surface the model is written against
  backend_cpu.rs      plain-Rust reference implementation of every op
  backend_wgpu.rs     wgpu implementation (batching, uniform ring, buffer pool, KV growth)
  device.rs           adapter/device acquisition with large-buffer limits
  safetensors.rs      mmap reader, F32/F16/BF16 decode, in-memory builder for tests
  dequant.rs          2/4/8-bit pack/unpack + dequant helpers
  weights.rs          logical name -> on-disk tensor (+ scale), lazy upload, cache
  config.rs           config.json -> flat, layer-aware config (quant rules via fancy-regex)
  model.rs            the Gemma-4 decoder graph, backend-agnostic
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
  gpu.rs              every kernel vs CPU (~1e-7), quant kernels, full model, real format
  engine.rs           load a tiny real-format checkpoint from disk and stream a reply
```

## What differs from the browser engine

- **Loading.** No streaming assembler: the checkpoint is `mmap`ed and each text
  tensor is uploaded straight from the map (vision/audio towers are never
  touched). Peak host memory is the page cache, not a JS heap.
- **Synchronous API.** `forward()` blocks on the logits readback; `generate()`
  returns an `Iterator<Item = Step>` instead of an async generator.
- **Runtime checks.** naga (wgpu's shader compiler) inserts per-access buffer
  bounds checks and loop-bounding counters by default; Chrome's Tint clamps
  indices instead. `--trusted-shaders` (or `GEMMA_WGPU_TRUSTED_SHADERS=1`)
  compiles the kernels with `ShaderRuntimeChecks::unchecked()`. The kernels
  only ever index with explicit uniform dims and the parity suite runs in both
  modes, so this is the mode to use for real inference.
- **Backend trait instead of duck typing.** `Backend` has an associated
  `Tensor` type; `Gemma4Model<B, W>` is generic over the backend and the
  weight source, so CPU and GPU share one graph and one test-suite.

Everything else — the batching into one command submission per forward, the
256-byte uniform ring, pow2-bucketed activation recycling per frame, in-place
KV capacity growth, `BITS`/`MROWS` pipeline overrides for the quantized matmul,
the `wNa8o8` packing conventions, the unweighted v-norm, `layer_scalar`, the
unapplied logit soft-cap — is a one-to-one port, and the `ARCHITECTURE.md` in
the parent directory describes this crate as well.

## Numbers (Apple M4, Metal, synthetic weights at real shapes, full 262k vocab)

| mode | prefill (64 tok) | decode |
|---|---|---|
| default (checked shaders) | 43.4 ms/token | 47.7 ms/token (21 tok/s) |
| `--trusted-shaders` | 26.5 ms/token | 33.4 ms/token (30 tok/s) |

The browser engine measured ~17 / ~26 ms on the same machine; the remaining
gap is host-side per-op cost in wgpu (bind group creation and validation for
the ~900 dispatches a forward records). GPU≡CPU parity on the tiny model is
~2e-7 max abs diff for both prefill and incremental decode.
