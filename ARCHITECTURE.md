# Architecture

How a chat turn flows through the engine, and where each piece lives.

## 1. Backend abstraction

The model never touches WebGPU directly. It calls an **op interface** that two
backends implement identically:

- `backend-cpu.js` — plain-JS reference. Slow, but exact and debuggable.
- `backend-webgpu.js` — enqueues WGSL compute passes; tensors are GPU buffers.

Both expose the same methods (`linear`, `rmsNorm`, `attention`, `rope`, `embedRows`,
`geglu`, `softcap`, `sliceRows`, `sliceCols`, `concatRows`, …) and the same tensor
shape semantics. So `model/gemma4.js` is written once and runs on either. This is
what makes correctness testable without a GPU and lets `test/gpu.test.mjs` assert
GPU≡CPU to ~1e-6.

## 2. Loading (`engine/gemma.js` → `Gemma4Mobile.load`)

1. acquire a WebGPU device (`engine/device.js`)
2. fetch `config.json` → `Gemma4Config` (`model/config.js`)
3. fetch `tokenizer.json` (+ `tokenizer_config`, `generation_config`) → `Tokenizer`
4. fetch the safetensors shard(s) with a streaming progress bar (`engine/hub.js`),
   parse headers (`engine/safetensors.js`), wrap in `ModelWeights` (`engine/weights.js`)
5. build `Gemma4Model`

`ModelWeights.get(name)` lazily decodes a tensor (F32/F16/BF16) or dequantizes a
QAT payload (`engine/dequant.js`), uploads it to the backend, and caches it.

## 3. A forward pass (`model/gemma4.js`)

```
ids ──▶ embed_tokens · √hidden ───────────────┐
ids ──▶ embed_tokens_per_layer · √256 ──▶ ple  │
                                               ▼
for each of 35 layers:
  ┌─ attention block ───────────────────────────────────────────────┐
  │ x = RMSNorm_in(h)                                                 │
  │ q = Linear(x); k,v = Linear(x)   # MQA: 8 q-heads, 1 kv-head;     │
  │                                  # shared layers skip k/v          │
  │ q,k = per-head RMSNorm(q,k);  v = per-head RMSNorm(v) UNWEIGHTED  │
  │ q,k = RoPE(q,k)          # θ + partial-rotary depend on layer     │
  │ (k,v) → KV cache         # shared layers read an earlier layer    │
  │ o = Attention(q, kcache, vcache)  # scale = 1.0 (no 1/√Dh!);      │
  │                                   # causal (+sliding for local)   │
  │ h = h + RMSNorm_postattn(Linear_o(o))                            │
  └──────────────────────────────────────────────────────────────────┘
  ┌─ feed-forward block (GeGLU) ─────────────────────────────────────┐
  │ x = RMSNorm_preffn(h)                                             │
  │ h = h + RMSNorm_postffn(Linear_down( gelu(Linear_gate x) * Linear_up x )) │
  └──────────────────────────────────────────────────────────────────┘
  ┌─ Per-Layer Embedding injection ──────────────────────────────────┐
  │ g  = gelu(Linear_gate_ple(h))      # → 256                        │
  │ h  = h + RMSNorm_ple(Linear_proj_ple( g * ple[layer] ))  # → 1536 │
  │ h  = h · layer_scalar              # learned per-layer scalar     │
  └──────────────────────────────────────────────────────────────────┘

h = RMSNorm_final(h)
logits = lm_head(h_last)              # last token only
# final_logit_softcapping is in the config but the runtime never applies it
```

All RMSNorms multiply by the stored weight directly — this checkpoint bakes the
Gemma `1 + w` into the tensor (verified against the reference runtime).

### Attention details
- **GQA/MQA**: query head `h` reads kv head `h / (Hq/Hkv)`.
- **Masking**: causal always; local layers additionally mask keys older than
  `sliding_window` (512).
- **head_dim varies by layer**: 256 on sliding layers, 512 on full-attention layers.
- **partial rotary**: full-attention layers rotate only the first 25% of head_dim
  (`partial_rotary_factor 0.25`); the tail passes through unrotated.
- **KV sharing**: layers in the last `num_kv_shared_layers` block don't compute their
  own K/V — they reuse the most recent earlier layer **of the same attention type**
  (so cached head_dim/RoPE base always match). See `Gemma4Config.kvSourceLayer`.

## 4. Decoding (`Gemma4Mobile.generate`)

Prefill the whole prompt once, then loop: sample the next token
(`model/sampler.js`), append it to the KV cache, forward just that one token.
`generate` is an async generator yielding `{ tokenId, token, text }` so the UI can
stream. A stateful tokenizer decoder appends only the new piece (including UTF-8
byte-fallback sequences), avoiding a full re-decode of all generated tokens on
every step. Stops on any EOS or `maxNewTokens`.

## 4b. Streaming weight load (`engine/stream-loader.js`)

The real checkpoint is one 2.46 GB `model.safetensors`. We never hold it whole:

1. range-fetch the 8-byte header length, then the JSON header
2. `fetch()` the body and read it as a chunk stream
3. `assembleFromStream` keeps a cursor over the header's tensors (sorted by
   offset); as each needed tensor's bytes complete it is uploaded immediately —
   packed weights (`U8`/`I8`) to a GPU byte buffer, floats (`BF16`→f32) to a GPU
   tensor — and the JS bytes are dropped. Vision/audio tensors are skipped.
4. `StreamedWeights` wraps the resulting GPU handles with the same `get(name)`
   contract the model expects, building `quantTensorFromGpu` lazily.

Range responses are required to be exact HTTP 206 responses, and the assembler
rejects incomplete tensor payloads, so a server that ignores ranges or a dropped
download cannot silently become a corrupt model load.

Quantized weights therefore live on the GPU in their packed form (~2.46 GB) and
are dequantized **inside** the kernels — never expanded to f32 (which would be
~16 GB).

### Quant format (`wNa8o8`) — verified against reference/gemma-4-e2b.js + header.json
- `U8` payload, sub-byte values packed LSB-first; OFFSET BINARY:
  `value = (raw - 2^(bits-1)) · scale` (zero-point subtraction, NOT two's-complement
  sign extension — raw 0 is the most negative value)
- `I8` payload (8-bit), native signed: `value = q · scale`
- one f32 scale per output row (`[N,1]` for every linear; per layer-block of 256
  for `embed_tokens_per_layer`, the only grouped scale)
- bit width per tensor comes from `config.quantization_config` (`config.bitsFor`);
  the 2-bit MLP layers (15-34) double their intermediate width to 12288
- the `*_activation_scale` / `*_cache_scale` scalars simulate int8 activations in
  the mobile runtime; this engine runs activations in f32 and ignores them

## 5. WGSL kernels (`engine/kernels.js`)

| kernel | does | parallelism |
|---|---|---|
| `MATMUL` | `C = A·Bᵀ`, f32 | 16×16 tiles, both tile loads coalesced along K |
| `GEMV` | f32 matvec (M = 1, e.g. `per_layer_model_projection` at decode) | 1 workgroup/output row, 256 lanes stride K + tree reduce |
| `MATMUL_Q` | dequant-on-the-fly matmul (2/4/8-bit), per-row scales | 8 lanes × 32 rows per workgroup; whole-u32 loads; `BITS`/`MROWS` pipeline overrides unroll the unpack loop (decode `MROWS=1`, prefill `MROWS=4` reuses each word across 4 A-rows) |
| `MATMUL_Q_SCALAR` | shape-agnostic fallback (odd K) | 1 thread/output, byte unpack |
| `EMBED` / `EMBED_Q` | gather rows, f32 or dequant | 1 thread/element |
| `RMSNORM` | row normalize · `(1+w)` | 1 workgroup/row, tree reduce |
| `ELEMENTWISE` | add/mul/scale/gelu/geglu/softcap | 1 thread/element |
| `ROPE` | half-split partial rotary | 1 thread/element |
| `ATTENTION` | tiled online-softmax, causal+sliding, GQA | 1 workgroup/(query,head); keys in tiles of 128 (one score/thread, two tree reductions per tile); ascending `kpos` enables tile-level skips, so sliding layers stay O(window) at decode |
| `SLICE_COLS` | per-layer PLE slice | 1 thread/element |

**gelu numerical stability.** The `gelu_pytorch_tanh` cubic term can drive the
`tanh` argument into the thousands; WGSL `tanh` computes `eˣ`, which overflows
f32 to `inf` → `inf/inf = NaN` (CPU `Math.tanh` saturates and stays finite). The
kernel clamps the argument (`tanh_safe`) before calling `tanh`. This was a real
GPU-only NaN that only surfaced with large activations in the full model.

`concatRows` / `sliceRows` are `copyBufferToBuffer` (rows are contiguous in
row-major). Uniform params are packed little-endian (16-byte aligned) by
`WebGpuBackend._u32f32`.

**Command batching.** Ops record into one shared command encoder (and one open
compute pass where possible — WebGPU auto-synchronizes dependent dispatches
within a pass); `flush()` submits everything, and `readback()` folds its
staging copy into the same submit. A forward pass is one `queue.submit` instead
of ~900. Per-op uniforms are suballocated from a persistent ring buffer at
256-byte offsets and written with one `queue.writeBuffer` per flush; index
uploads (`_u32`) are memoized per frame by array identity so the positions
arrays upload twice per step, not per layer.

**GPU memory lifecycle.** Each `forward()` runs inside a backend memory frame
(`beginFrame`/`endFrame`): every intermediate activation allocated during the
pass is recycled at the end of it into a pow2-bucketed buffer pool (kernels
fully overwrite their outputs and never read buffer length, so stale contents
are harmless), keeping only the live KV cache (plus weights, which are never
frame-tracked). `endFrame` runs right after the awaited logits readback, which
guarantees the GPU has finished every submitted command, so recycling is
race-free. Native objects created while recording (bind groups, command
buffers, index buffers) are also retained until `endFrame` — some WebGPU
bindings (Node/Dawn) let JS GC destroy natives the GPU is still consuming. Net
effect: GPU memory is bounded per step, and steady-state decode allocates
almost nothing.

**KV cache growth.** `kvAppend` writes new K/V rows in place into a capacity
buffer that doubles when full (`kvcache.js` keeps a logical-length view), so a
decode step copies O(new rows), not the whole history. Retired capacity
buffers are freed at `endFrame` via the cache's `dead` list.

## Assumptions, isolated on purpose

- **QAT packing**: the symmetric grouped layout in `dequant.js` and the
  `<name>` + `<name>_scale` convention in `weights.js` are the standard scheme;
  Google's exact packing for this checkpoint isn't published. If it differs, only
  those two files change.
- **PLE injection point** follows the published Gemma-3n reference structure.
- **f32 everywhere** — see README "Notes".
