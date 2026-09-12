// CPU reference backend.
//
// Every op the model needs is implemented here in plain JS. It is the ground
// truth the WebGPU backend is checked against, and it lets the whole transformer
// run (slowly) with no GPU — which is how the Node test-suite validates the math.
//
// Tensors are { data: Float32Array, shape: number[] }.

export class CpuTensor {
  constructor(data, shape) {
    this.data = data;
    this.shape = shape;
  }
  get size() {
    return this.data.length;
  }
}

const t = (data, shape) => new CpuTensor(data, shape);
const numel = (shape) => shape.reduce((a, b) => a * b, 1);

export class CpuBackend {
  constructor() {
    this.name = "cpu";
  }

  async init() {}
  async readback(x) {
    return x.data;
  }
  destroy() {}

  tensor(data, shape) {
    return t(data instanceof Float32Array ? data : Float32Array.from(data), shape);
  }
  zeros(shape) {
    return t(new Float32Array(numel(shape)), shape);
  }

  // rows[i] = table[ids[i]] * scale ; table is [V, D] (f32 or quantized).
  embedRows(table, ids, scale = 1.0) {
    const D = table.shape[1];
    const out = new Float32Array(ids.length * D);
    if (table.quant) {
      for (let i = 0; i < ids.length; i++)
        for (let d = 0; d < D; d++) out[i * D + d] = this._deq(table.quant, ids[i], d) * scale;
      return t(out, [ids.length, D]);
    }
    for (let i = 0; i < ids.length; i++) {
      const src = ids[i] * D;
      for (let d = 0; d < D; d++) out[i * D + d] = table.data[src + d] * scale;
    }
    return t(out, [ids.length, D]);
  }

  // RMSNorm: y = x/rms(x) * weight. The qat-mobile checkpoint stores norm
  // weights as the FULL multiplier (the reference runtime applies plain `w`,
  // not HF-Gemma's `1 + w`). Pass weight = null for an unweighted norm
  // (used on v ahead of attention).
  rmsNorm(x, weight, eps) {
    const [T, D] = x.shape;
    const out = new Float32Array(T * D);
    for (let i = 0; i < T; i++) {
      let ss = 0;
      const base = i * D;
      for (let d = 0; d < D; d++) {
        const v = x.data[base + d];
        ss += v * v;
      }
      const inv = 1 / Math.sqrt(ss / D + eps);
      for (let d = 0; d < D; d++) {
        out[base + d] = x.data[base + d] * inv * (weight ? weight.data[d] : 1);
      }
    }
    return t(out, [T, D]);
  }

  // A quantized weight kept un-expanded. `q` = { qbytes:Uint8Array, scales:
  // Float32Array, shape:[N,K], bits, signedI8, groupSize }.
  quantTensor(q) {
    const perByte = q.signedI8 ? 1 : 8 / q.bits;
    return { quant: { ...q, perByte }, shape: q.shape, size: q.shape[0] * q.shape[1] };
  }

  // Dequantize a single weight element matching the WGSL unpack exactly.
  _deq(q, n, k) {
    const { qbytes, scales, shape, bits, signedI8, groupSize, perByte } = q;
    const K = shape[1];
    const rowBytes = signedI8 ? K : K / perByte;
    let qv;
    if (signedI8) {
      qv = (qbytes[n * rowBytes + k] << 24) >> 24;
    } else {
      const b = qbytes[n * rowBytes + ((k / perByte) | 0)];
      const shift = (k % perByte) * bits;
      // offset binary: value = raw - zeroPoint, zeroPoint = 2^(bits-1)
      qv = ((b >> shift) & ((1 << bits) - 1)) - (1 << (bits - 1));
    }
    const sStride = Math.ceil(K / groupSize);
    return qv * scales[n * sStride + ((k / groupSize) | 0)];
  }

  // y = x @ W^T ; x is [T, K], W is [N, K] (row-major like PyTorch nn.Linear).
  linear(x, W) {
    const [T, K] = x.shape;
    const N = W.shape[0];
    const out = new Float32Array(T * N);
    if (W.quant) {
      for (let i = 0; i < T; i++)
        for (let n = 0; n < N; n++) {
          let acc = 0;
          for (let k = 0; k < K; k++) acc += x.data[i * K + k] * this._deq(W.quant, n, k);
          out[i * N + n] = acc;
        }
      return t(out, [T, N]);
    }
    for (let i = 0; i < T; i++) {
      const xb = i * K;
      for (let n = 0; n < N; n++) {
        const wb = n * K;
        let acc = 0;
        for (let k = 0; k < K; k++) acc += x.data[xb + k] * W.data[wb + k];
        out[i * N + n] = acc;
      }
    }
    return t(out, [T, N]);
  }

  add(a, b) {
    const out = new Float32Array(a.size);
    for (let i = 0; i < out.length; i++) out[i] = a.data[i] + b.data[i];
    return t(out, a.shape.slice());
  }
  mul(a, b) {
    const out = new Float32Array(a.size);
    for (let i = 0; i < out.length; i++) out[i] = a.data[i] * b.data[i];
    return t(out, a.shape.slice());
  }
  scale(a, s) {
    const out = new Float32Array(a.size);
    for (let i = 0; i < out.length; i++) out[i] = a.data[i] * s;
    return t(out, a.shape.slice());
  }
  // Multiply by a learned scalar that lives in a [1] tensor (layer_scalar).
  scaleByTensor(a, s) {
    return this.scale(a, s.data[0]);
  }

  geluTanh(x) {
    const out = new Float32Array(x.size);
    const c = Math.sqrt(2 / Math.PI);
    for (let i = 0; i < out.length; i++) {
      const v = x.data[i];
      out[i] = 0.5 * v * (1 + Math.tanh(c * (v + 0.044715 * v * v * v)));
    }
    return t(out, x.shape.slice());
  }

  // GeGLU: gelu(gate) * up. gate/up are [T, I].
  geglu(gate, up) {
    return this.mul(this.geluTanh(gate), up);
  }

  // cap * tanh(x / cap), elementwise (logit / attn soft-capping).
  softcap(x, cap) {
    const out = new Float32Array(x.size);
    for (let i = 0; i < out.length; i++) out[i] = cap * Math.tanh(x.data[i] / cap);
    return t(out, x.shape.slice());
  }

  // RoPE on q/k laid out [T, heads, headDim]. Rotates only the first
  // `rotaryDim` dims (partial rotary). Neighbour-pair (interleaved) convention
  // is avoided in favour of the half-split convention used by HF Gemma.
  rope(x, positions, theta, headDim, heads, rotaryDim) {
    const [T] = x.shape;
    const out = Float32Array.from(x.data);
    const half = rotaryDim >> 1;
    for (let i = 0; i < T; i++) {
      const pos = positions[i];
      for (let h = 0; h < heads; h++) {
        const base = (i * heads + h) * headDim;
        for (let j = 0; j < half; j++) {
          const freq = Math.pow(theta, -(2 * j) / rotaryDim);
          const ang = pos * freq;
          const cos = Math.cos(ang), sin = Math.sin(ang);
          const a = x.data[base + j];
          const b = x.data[base + j + half];
          out[base + j] = a * cos - b * sin;
          out[base + j + half] = b * cos + a * sin;
        }
      }
    }
    return t(out, x.shape.slice());
  }

  // Multi/grouped-query causal attention with optional sliding window.
  // q:[T,Hq,Dh] k:[S,Hkv,Dh] v:[S,Hkv,Dv]; positions length T and S (kv).
  // Returns [T, Hq*Dv].
  attention(q, k, v, { qPos, kPos, scale, slidingWindow, attnSoftcap }) {
    const [T, Hq, Dh] = q.shape;
    const S = k.shape[0];
    const Hkv = k.shape[1];
    const Dv = v.shape[2];
    const group = Hq / Hkv;
    const out = new Float32Array(T * Hq * Dv);
    const scores = new Float32Array(S);
    for (let i = 0; i < T; i++) {
      const qpos = qPos[i];
      for (let h = 0; h < Hq; h++) {
        const kvh = Math.floor(h / group);
        const qb = (i * Hq + h) * Dh;
        let maxv = -Infinity;
        for (let j = 0; j < S; j++) {
          const kpos = kPos[j];
          // causal + optional sliding-window mask
          if (kpos > qpos || (slidingWindow > 0 && qpos - kpos >= slidingWindow)) {
            scores[j] = -Infinity;
            continue;
          }
          const kb = (j * Hkv + kvh) * Dh;
          let dot = 0;
          for (let d = 0; d < Dh; d++) dot += q.data[qb + d] * k.data[kb + d];
          dot *= scale;
          if (attnSoftcap > 0) dot = attnSoftcap * Math.tanh(dot / attnSoftcap);
          scores[j] = dot;
          if (dot > maxv) maxv = dot;
        }
        // softmax over valid keys
        let sum = 0;
        for (let j = 0; j < S; j++) {
          if (scores[j] === -Infinity) {
            scores[j] = 0;
          } else {
            const e = Math.exp(scores[j] - maxv);
            scores[j] = e;
            sum += e;
          }
        }
        const inv = sum > 0 ? 1 / sum : 0;
        const ob = (i * Hq + h) * Dv;
        for (let j = 0; j < S; j++) {
          const w = scores[j] * inv;
          if (w === 0) continue;
          const vb = (j * Hkv + kvh) * Dv;
          for (let d = 0; d < Dv; d++) out[ob + d] += w * v.data[vb + d];
        }
      }
    }
    return t(out, [T, Hq * Dv]);
  }

  // Reshape view (no copy needed for CPU).
  reshape(x, shape) {
    return t(x.data, shape);
  }

  // Contiguous row slice: rows [start, start+count) of a [rows, dim] tensor.
  sliceRows(x, start, count) {
    const dim = x.shape[1];
    return t(x.data.slice(start * dim, (start + count) * dim), [count, dim]);
  }

  // Column slice: [:, offset:offset+width] of a [rows, stride] tensor.
  sliceCols(x, offset, width) {
    const [rows, stride] = x.shape;
    const out = new Float32Array(rows * width);
    for (let r = 0; r < rows; r++) {
      out.set(x.data.subarray(r * stride + offset, r * stride + offset + width), r * width);
    }
    return t(out, [rows, width]);
  }

  // Concatenate along dim 0 (sequence). Both must share trailing dims.
  concatRows(a, b) {
    if (!a) return t(Float32Array.from(b.data), b.shape.slice());
    const out = new Float32Array(a.size + b.size);
    out.set(a.data, 0);
    out.set(b.data, a.size);
    return t(out, [a.shape[0] + b.shape[0], ...a.shape.slice(1)]);
  }

  // KV-cache append (same contract as WebGpuBackend.kvAppend). The CPU
  // reference just concatenates; `dead` mirrors the GPU retirement signal but
  // is ignored because CpuBackend has no freeTensor.
  kvAppend(existing, add) {
    return { tensor: this.concatRows(existing, add), dead: existing };
  }
}
