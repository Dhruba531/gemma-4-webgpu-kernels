// Streaming weight loader.
//
// The full model.safetensors is 2.46 GB — too large to hold as one ArrayBuffer
// (browsers cap ArrayBuffer near 2 GB and the contiguous allocation fails). This
// streams the HTTP body and assembles ONE tensor at a time, uploading each to
// the GPU (or decoding to f32) and freeing the JS bytes before the next. Peak JS
// heap stays at the size of the single largest tensor (~1.17 GB), and tensors
// for the unused vision/audio towers are skipped entirely.
//
// Quantized payloads (U8/I8) are uploaded packed and paired with their f32 scale
// sibling; floats (BF16/F16/F32) are decoded to f32. The result is a name→tensor
// map consumed by StreamedWeights.

import { decodeToF32 } from "./safetensors.js";
import { resolveTensorNames } from "./weights.js";

const isScale = (n) => /(\.|_)scale[s]?$|embedding_scale$/.test(n);

// Which on-disk tensors the text model needs (skip vision_tower/audio_tower/…).
export function isTextTensor(name) {
  return name.startsWith("model.language_model.") || name.startsWith("lm_head");
}

/**
 * Assemble tensors from an ordered async stream of byte chunks.
 * @param chunks async iterable of Uint8Array (the file body, in order)
 * @param header parsed safetensors header
 * @param dataStart byte offset where the payload begins
 * @param backend
 * @param onProgress (loadedBytes)=>void
 * @returns Map<realName, { kind:'quant'|'float', ... }>
 */
export async function assembleFromStream({ chunks, header, dataStart, backend, want = isTextTensor, onProgress = () => {} }) {
  const needed = Object.entries(header)
    .filter(([k]) => k !== "__metadata__" && want(k))
    .map(([name, m]) => ({ name, abs0: dataStart + m.data_offsets[0], abs1: dataStart + m.data_offsets[1], dtype: m.dtype, shape: m.shape, buf: null, filled: 0 }))
    .sort((a, b) => a.abs0 - b.abs0);

  // raw payloads, keyed by name, ready to be turned into tensors. On GPU
  // backends we upload immediately and keep only GPU handles, so the big weight
  // bytes are freed from the JS heap as soon as each tensor completes.
  const qbytes = new Map();  // name -> { bytes|gpuBuffer, byteLength, dtype, shape }
  const scales = new Map();  // name -> Float32Array (small)
  const floats = new Map();  // name -> { data|tensor, shape }
  const gpu = typeof backend._bytesBuf === "function";

  const finish = (T) => {
    if (T.dtype === "U8" || T.dtype === "I8") {
      if (gpu) qbytes.set(T.name, { gpuBuffer: backend._bytesBuf(T.buf), byteLength: T.buf.length, dtype: T.dtype, shape: T.shape });
      else qbytes.set(T.name, { bytes: T.buf, dtype: T.dtype, shape: T.shape });
    } else if (isScale(T.name)) {
      scales.set(T.name, decodeToF32(T.dtype, T.buf));
    } else {
      const data = decodeToF32(T.dtype, T.buf);
      floats.set(T.name, gpu ? { tensor: backend.tensor(data, T.shape), shape: T.shape } : { data, shape: T.shape });
    }
    T.buf = null; // allow GC
  };

  let filePos = 0;
  let head = 0; // first not-yet-complete needed tensor
  let loaded = 0;
  for await (const chunk of chunks) {
    const chunkStart = filePos;
    const chunkEnd = filePos + chunk.byteLength;
    for (let i = head; i < needed.length && needed[i].abs0 < chunkEnd; i++) {
      const T = needed[i];
      if (T.abs1 <= chunkStart || T.filled === -1) continue; // already done / before window
      if (!T.buf) T.buf = new Uint8Array(T.abs1 - T.abs0);
      const copyStart = Math.max(T.abs0, chunkStart);
      const copyEnd = Math.min(T.abs1, chunkEnd);
      T.buf.set(chunk.subarray(copyStart - chunkStart, copyEnd - chunkStart), copyStart - T.abs0);
      T.filled += copyEnd - copyStart;
      if (T.filled === T.buf.length) { finish(T); T.filled = -1; }
    }
    while (head < needed.length && needed[head].filled === -1) head++;
    filePos = chunkEnd;
    loaded += chunk.byteLength;
    onProgress(loaded);
  }

  if (head < needed.length) {
    const T = needed[head];
    const received = Math.max(0, T.filled);
    const expected = T.abs1 - T.abs0;
    throw new Error(`truncated weight stream at ${T.name}: received ${received} of ${expected} bytes`);
  }

  return { qbytes, scales, floats };
}

// Weight provider over assembled stream parts. Same get(logical) contract as
// ModelWeights, but reads from the in-memory part maps and builds GPU tensors
// lazily (so quant uploads happen on first use, spreading GPU allocation).
export class StreamedWeights {
  constructor(parts, config, backend) {
    this.parts = parts;
    this.config = config;
    this.B = backend;
    this.cache = new Map();
  }

  get(logical) {
    if (this.cache.has(logical)) return this.cache.get(logical);
    const { weight, scale, bare } = resolveTensorNames(logical);
    const base = logical.replace(/\.weight$/, "");
    let tensor = null;

    const floatTensor = (f) => (f.tensor ? f.tensor : this.B.tensor(f.data, f.shape));

    if (this.parts.qbytes.has(weight)) {
      const q = this.parts.qbytes.get(weight);
      const signedI8 = q.dtype === "I8";
      const bits = signedI8 ? 8 : this.config.bitsFor(base);
      const perByte = signedI8 ? 1 : 8 / bits;
      const N = q.shape[0];
      const K = q.shape[1] * perByte;
      const scaleData = this.parts.scales.get(scale);
      if (!scaleData) throw new Error(`quantized tensor ${weight} is missing its scale tensor ${scale}`);
      const scaleCols = Math.max(1, Math.round(scaleData.length / N));
      const spec = {
        scales: scaleData, shape: [N, K], bits,
        signedI8: signedI8 ? 1 : 0,
        groupSize: K / scaleCols,
      };
      tensor = q.gpuBuffer
        ? this.B.quantTensorFromGpu({ ...spec, buffer: q.gpuBuffer })
        : this.B.quantTensor({ ...spec, qbytes: q.bytes });
    } else if (this.parts.floats.has(weight)) {
      tensor = floatTensor(this.parts.floats.get(weight));
    } else if (bare && this.parts.floats.has(bare)) {
      tensor = floatTensor(this.parts.floats.get(bare));
    }

    this.cache.set(logical, tensor);
    return tensor;
  }
}
