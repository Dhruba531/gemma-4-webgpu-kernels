// Streaming-loader test (CPU backend; deterministic). Verifies the stream
// assembler reconstructs tensors split across arbitrary chunk boundaries, skips
// non-text tensors, and that StreamedWeights builds usable tensors.
// Run: node test/stream.test.mjs
import assert from "node:assert";
import { CpuBackend } from "../src/engine/backend-cpu.js";
import { Gemma4Config } from "../src/model/config.js";
import { SafetensorsFile, buildSafetensors } from "../src/engine/safetensors.js";
import { assembleFromStream, StreamedWeights, isTextTensor } from "../src/engine/stream-loader.js";

let pass = 0;
const ok = (c, m) => { assert.ok(c, m); console.log("  ✓ " + m); pass++; };
const rng = (() => { let a = 3; return () => ((a = (a * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff); })();

const packU = (vals, bits) => { const pb = 8 / bits, o = new Uint8Array(Math.ceil(vals.length / pb)); vals.forEach((v, i) => o[Math.floor(i / pb)] |= (v & ((1 << bits) - 1)) << ((i % pb) * bits)); return o; };
const f32b = (a) => new Uint8Array(Float32Array.from(a).buffer.slice());
const bf16b = (a) => { const u = new Uint8Array(a.length * 2), dv = new DataView(u.buffer), fb = new DataView(new ArrayBuffer(4)); a.forEach((v, i) => { fb.setFloat32(0, v, true); dv.setUint16(i * 2, fb.getUint16(2, true), true); }); return u; };

const cfg = new Gemma4Config({
  quantization_config: { module_quant_configs: { "^lm_head$": { num_bits: 2 }, "language_model\\.layers\\.\\d+\\.self_attn\\.": { num_bits: 4 } } },
  text_config: { hidden_size: 16, num_hidden_layers: 1 },
});

const N = 8, K = 16;
const qvals = Array.from({ length: N * K }, () => Math.floor(rng() * 16));
const T = {
  // a 4-bit attn weight + scale (text)
  "model.language_model.layers.0.self_attn.q_proj.weight": { dtype: "U8", shape: [N, K / 2], bytes: packU(qvals, 4) },
  "model.language_model.layers.0.self_attn.q_proj.weight_scale": { dtype: "F32", shape: [N, 1], bytes: f32b(Array.from({ length: N }, () => 0.03)) },
  // a BF16 norm (text)
  "model.language_model.norm.weight": { dtype: "BF16", shape: [K], bytes: bf16b(Array.from({ length: K }, () => 0.1)) },
  // a vision tensor that must be skipped
  "model.vision_tower.patch.weight": { dtype: "F32", shape: [4], bytes: f32b([1, 2, 3, 4]) },
};
const blob = buildSafetensors(T);
const st = new SafetensorsFile(blob);

// feed the body in irregular chunks
async function* chunker(buf, size) {
  const u = new Uint8Array(buf);
  for (let i = 0; i < u.length; i += size) yield u.subarray(i, Math.min(i + size, u.length));
}

console.log("stream assembly");
const parts = await assembleFromStream({ chunks: chunker(blob, 7), header: st.header, dataStart: st.dataStart, backend: new CpuBackend() });
ok(parts.qbytes.has("model.language_model.layers.0.self_attn.q_proj.weight"), "captured quant weight");
ok(parts.scales.has("model.language_model.layers.0.self_attn.q_proj.weight_scale"), "captured scale");
ok(parts.floats.has("model.language_model.norm.weight"), "captured BF16 norm");
ok(!parts.qbytes.has("model.vision_tower.patch.weight") && !parts.floats.has("model.vision_tower.patch.weight"), "skipped vision tensor");

// reconstructed bytes match the original
const orig = st.rawBytes("model.language_model.layers.0.self_attn.q_proj.weight");
const got = parts.qbytes.get("model.language_model.layers.0.self_attn.q_proj.weight").bytes;
ok(orig.length === got.length && orig.every((b, i) => b === got[i]), "quant bytes reconstructed exactly across chunk boundaries");

console.log("StreamedWeights builds tensors");
const C = new CpuBackend();
const W = new StreamedWeights(parts, cfg, C);
const qp = W.get("language_model.layers.0.self_attn.q_proj.weight");
ok(qp && qp.quant && qp.quant.bits === 4, "quant tensor built with 4 bits");
const nrm = W.get("language_model.norm.weight");
ok(nrm && Math.abs(nrm.data[0] - 0.1) < 1e-2, "BF16 norm decoded");

// quant linear via StreamedWeights matches a direct reference dequant
const x = Float32Array.from({ length: K }, () => rng());
const y = C.linear(C.tensor(x, [1, K]), qp).data;
let ref = new Float32Array(N);
for (let n = 0; n < N; n++) { let acc = 0; for (let k = 0; k < K; k++) { const b = got[n * (K / 2) + (k >> 1)]; const raw = (b >> ((k & 1) * 4)) & 15; const q = raw - 8; acc += x[k] * q * 0.03; } ref[n] = acc; } // offset binary: q = raw - 2^(bits-1)
let max = 0; for (let n = 0; n < N; n++) max = Math.max(max, Math.abs(y[n] - ref[n]));
ok(max < 1e-4, `streamed quant linear matches reference (max diff ${max.toExponential(2)})`);

console.log("stream validation");
const qMeta = st.header["model.language_model.layers.0.self_attn.q_proj.weight"];
const truncatedAt = st.dataStart + qMeta.data_offsets[1] - 1;
await assert.rejects(
  () => assembleFromStream({
    chunks: chunker(blob.slice(0, truncatedAt), 7),
    header: st.header,
    dataStart: st.dataStart,
    backend: new CpuBackend(),
  }),
  /truncated weight stream/,
);
ok(true, "rejects a truncated checkpoint stream");

console.log(`\n${pass} checks passed`);
