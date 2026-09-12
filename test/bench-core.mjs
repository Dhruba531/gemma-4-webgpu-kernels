// Shared synthetic-weight benchmark core, driven by test/bench.mjs (Node/Dawn)
// and test/bench.html (browser). Real Gemma-4 E2B shapes, random data: kernel
// timings and memory traffic match the real model without the 2.46 GB download.

import { WebGpuBackend } from "../src/engine/backend-webgpu.js";
import { Gemma4Config } from "../src/model/config.js";
import { Gemma4Model } from "../src/model/gemma4.js";

// Time isolated kernel shapes (decode M=1) to attribute the per-token budget:
// each entry loops the op `reps` times back-to-back in one submit, then divides.
export async function runKernelBench({ device, log }) {
  const B = new WebGpuBackend(device);
  let seed = 12345;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 2 ** 32);
  const randBytes = (n) => {
    const u = new Uint8Array(n);
    const w = new Uint32Array(u.buffer, 0, n >> 2);
    let s = 987654321;
    for (let i = 0; i < w.length; i++) { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; w[i] = s >>> 0; }
    return u;
  };
  const scales = (n) => Float32Array.from({ length: n }, () => 0.004 + rnd() * 0.008);
  const randF32 = (n, sc) => Float32Array.from({ length: n }, () => (rnd() * 2 - 1) * sc);
  const qt = (N, K, bits, signedI8 = 0) => B.quantTensor({
    qbytes: randBytes(Math.ceil((N * K) / (signedI8 ? 1 : 8 / bits))),
    scales: scales(N), shape: [N, K], bits, signedI8, groupSize: K,
  });

  const cases = [
    { name: "lm_head 2b [262144,1536]", reps: 20, make: () => { const W = qt(262144, 1536, 2); const x = B.tensor(randF32(1536, 0.1), [1, 1536]); return () => B.linear(x, W); } },
    { name: "mlp gate 4b [6144,1536]", reps: 60, make: () => { const W = qt(6144, 1536, 4); const x = B.tensor(randF32(1536, 0.1), [1, 1536]); return () => B.linear(x, W); } },
    { name: "mlp down 2b [1536,6144]", reps: 60, make: () => { const W = qt(1536, 6144, 2); const x = B.tensor(randF32(6144, 0.1), [1, 6144]); return () => B.linear(x, W); } },
    { name: "q_proj 4b [2048,1536]", reps: 60, make: () => { const W = qt(2048, 1536, 4); const x = B.tensor(randF32(1536, 0.1), [1, 1536]); return () => B.linear(x, W); } },
    { name: "gemv f32 [8960,1536]", reps: 60, make: () => { const W = B.tensor(randF32(8960 * 1536, 0.02), [8960, 1536]); const x = B.tensor(randF32(1536, 0.1), [1, 1536]); return () => B.linear(x, W); } },
    { name: "attention Dh256 S512", reps: 60, make: () => { const q = B.tensor(randF32(8 * 256, 0.1), [1, 8, 256]); const k = B.tensor(randF32(512 * 256, 0.1), [512, 1, 256]); const v = B.tensor(randF32(512 * 256, 0.1), [512, 1, 256]); const kp = Array.from({ length: 512 }, (_, i) => i); return () => B.attention(q, k, v, { qPos: [511], kPos: kp, scale: 1 / 16, slidingWindow: 512 }); } },
    // chained (output feeds next input) so the GPU cannot overlap reps: this
    // measures dependent dispatch-to-dispatch latency, the floor for the ~900
    // small ops in a forward pass
    { name: "rmsNorm [1,1536] chained", reps: 400, make: () => { let x = B.tensor(randF32(1536, 0.1), [1, 1536]); const w = B.tensor(randF32(1536, 0.05), [1536]); return () => (x = B.rmsNorm(x, w, 1e-6)); } },
    { name: "add [1,1536] chained", reps: 400, make: () => { let a = B.tensor(randF32(1536, 0.1), [1, 1536]); const b = B.tensor(randF32(1536, 0.1), [1, 1536]); return () => (a = B.add(a, b)); } },
  ];
  for (const c of cases) {
    const run = c.make();
    // warmup + pipeline compile
    let out = run();
    await B.readback(out);
    const t0 = performance.now();
    for (let i = 0; i < c.reps; i++) out = run();
    await B.readback(out);
    const ms = (performance.now() - t0) / c.reps;
    log(`  ${c.name}: ${ms.toFixed(3)} ms/op`);
  }
}

export async function runBench({ device, configRaw, log, prefill = 64, decode = 24, vdiv = 1 }) {
  const raw = JSON.parse(JSON.stringify(configRaw));
  raw.text_config.vocab_size = Math.floor(raw.text_config.vocab_size / vdiv);
  raw.text_config.vocab_size_per_layer_input = raw.text_config.vocab_size;
  const C = new Gemma4Config(raw);
  const B = new WebGpuBackend(device);

  let seed = 0x9e3779b9;
  const rnd = () => ((seed = (seed * 1103515245 + 12345) >>> 0) / 2 ** 32);
  function randBytes(n) {
    const u = new Uint8Array(n);
    const w = new Uint32Array(u.buffer, 0, n >> 2);
    let s = (seed = (seed * 1103515245 + 12345) >>> 0) | 1;
    for (let i = 0; i < w.length; i++) { s ^= s << 13; s ^= s >>> 17; s ^= s << 5; w[i] = s >>> 0; }
    return u;
  }
  const randF32 = (n, sc) => Float32Array.from({ length: n }, () => (rnd() * 2 - 1) * sc);
  const scales = (n) => Float32Array.from({ length: n }, () => 0.004 + rnd() * 0.008);

  const H = C.hidden_size, I = C.intermediate_size, L = C.num_hidden_layers;
  const Dple = C.hidden_size_per_layer_input, V = C.vocab_size;
  const Hq = C.num_attention_heads, Hkv = C.num_key_value_heads;

  // name -> spec; spec = { shape, bits, signedI8?, groupSize? } | { shape, f32: scaleOfRandom, base? }
  const table = new Map();
  // norm weights store the full multiplier -> synthesize near 1
  const norm = (name, dim) => table.set(name, { shape: [dim], f32: 0.05, base: 1 });
  table.set("language_model.embed_tokens.weight", { shape: [V, H], bits: 2 });
  table.set("language_model.embed_tokens_per_layer.weight", { shape: [V, L * Dple], bits: 4, groupSize: Dple });
  table.set("language_model.per_layer_model_projection.weight", { shape: [L * Dple, H], f32: 0.02 });
  norm("language_model.per_layer_projection_norm.weight", Dple);
  norm("language_model.norm.weight", H);
  table.set("lm_head.weight", { shape: [V, H], bits: 2 });
  for (let i = 0; i < L; i++) {
    const p = `language_model.layers.${i}`, Dh = C.headDim(i);
    const mlpBits = C.bitsFor(`${p}.mlp.gate_proj`);
    table.set(`${p}.self_attn.q_proj.weight`, { shape: [Hq * Dh, H], bits: 4 });
    table.set(`${p}.self_attn.k_proj.weight`, { shape: [Hkv * Dh, H], bits: 4 });
    table.set(`${p}.self_attn.v_proj.weight`, { shape: [Hkv * Dh, H], bits: 4 });
    table.set(`${p}.self_attn.o_proj.weight`, { shape: [H, Hq * Dh], bits: 4 });
    norm(`${p}.self_attn.q_norm.weight`, Dh);
    norm(`${p}.self_attn.k_norm.weight`, Dh);
    table.set(`${p}.mlp.gate_proj.weight`, { shape: [I, H], bits: mlpBits });
    table.set(`${p}.mlp.up_proj.weight`, { shape: [I, H], bits: mlpBits });
    table.set(`${p}.mlp.down_proj.weight`, { shape: [H, I], bits: mlpBits });
    table.set(`${p}.per_layer_input_gate.weight`, { shape: [Dple, H], bits: 8, signedI8: 1 });
    table.set(`${p}.per_layer_projection.weight`, { shape: [H, Dple], bits: 8, signedI8: 1 });
    table.set(`${p}.layer_scalar`, { shape: [1], f32: 0.05, base: 1 });
    norm(`${p}.post_per_layer_input_norm.weight`, H);
    for (const n of ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"]) {
      norm(`${p}.${n}.weight`, H);
    }
  }

  let uploaded = 0;
  const weights = {
    _c: new Map(),
    get(name) {
      if (this._c.has(name)) return this._c.get(name);
      const spec = table.get(name);
      if (!spec) return null;
      let t;
      if (spec.f32 !== undefined) {
        const n = spec.shape.reduce((a, b) => a * b, 1);
        const data = randF32(n, spec.f32);
        if (spec.base) for (let i = 0; i < n; i++) data[i] += spec.base;
        t = B.tensor(data, spec.shape);
        uploaded += n * 4;
      } else {
        const [N, K] = spec.shape;
        const signedI8 = spec.signedI8 ?? 0;
        const perByte = signedI8 ? 1 : 8 / spec.bits;
        const groupSize = spec.groupSize ?? K;
        const qbytes = randBytes(Math.ceil((N * K) / perByte));
        t = B.quantTensor({
          qbytes, scales: scales(N * Math.ceil(K / groupSize)),
          shape: [N, K], bits: spec.bits, signedI8, groupSize,
        });
        uploaded += qbytes.byteLength;
      }
      this._c.set(name, t);
      return t;
    },
  };

  log(`config: ${L} layers, hidden ${H}, vocab ${V}, prefill ${prefill}, decode ${decode}`);
  const t0 = performance.now();
  const model = new Gemma4Model(C, B, weights);
  for (const name of table.keys()) weights.get(name); // exclude upload from timings
  log(`weights: ${(uploaded / 2 ** 30).toFixed(2)} GiB uploaded in ${((performance.now() - t0) / 1000).toFixed(1)}s`);

  const ids = Array.from({ length: prefill }, () => 7 + Math.floor(rnd() * (V - 8)));
  const finite = (a) => { for (let i = 0; i < a.length; i++) if (!Number.isFinite(a[i])) return false; return true; };
  const l2 = (a) => { let s = 0; for (let i = 0; i < 512; i++) s += a[i] * a[i]; return Math.sqrt(s); };

  const tPre = performance.now();
  let logits = await model.forward(ids, ids.map((_, i) => i));
  const prefillMs = performance.now() - tPre;
  log(`prefill: ${prefillMs.toFixed(0)} ms total, ${(prefillMs / prefill).toFixed(1)} ms/token  (finite: ${finite(logits)}, l2=${l2(logits).toFixed(3)})`);

  let pos = prefill;
  for (let s = 0; s < 2; s++) logits = await model.forward([ids[0]], [pos++]); // warmup

  const times = [];
  for (let s = 0; s < decode; s++) {
    const t = performance.now();
    logits = await model.forward([ids[(s * 13) % prefill]], [pos++]);
    times.push(performance.now() - t);
  }
  times.sort((a, b) => a - b);
  const med = times[times.length >> 1];
  log(`decode: median ${med.toFixed(1)} ms/token (${(1000 / med).toFixed(1)} tok/s), min ${times[0].toFixed(1)}, max ${times[times.length - 1].toFixed(1)}  (finite: ${finite(logits)}, l2=${l2(logits).toFixed(3)})`);
  return { prefillMs, prefillPerTok: prefillMs / prefill, decodeMedian: med, times, l2: l2(logits) };
}
