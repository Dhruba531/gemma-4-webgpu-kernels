// End-to-end streaming path on GPU: build a real-format blob, feed it through
// assembleFromStream (GPU immediate-upload) + StreamedWeights, run the full
// model, and compare to a CPU non-streaming reference.
// (Headless Dawn occasionally segfaults on tiny I8 buffers — unrelated to logic;
//  the runner retries.) Run: node test/stream-gpu.test.mjs
import assert from "node:assert";
import { create, globals } from "webgpu";
Object.assign(globalThis, globals);
import { CpuBackend } from "../src/engine/backend-cpu.js";
import { WebGpuBackend } from "../src/engine/backend-webgpu.js";
import { Gemma4Config } from "../src/model/config.js";
import { Gemma4Model } from "../src/model/gemma4.js";
import { SafetensorsFile, buildSafetensors } from "../src/engine/safetensors.js";
import { ModelWeights } from "../src/engine/weights.js";
import { assembleFromStream, StreamedWeights } from "../src/engine/stream-loader.js";

const device = await (await create([]).requestAdapter()).requestDevice();
const G = new WebGpuBackend(device), C = new CpuBackend();
const rng = (() => { let a = 5; return () => ((a = (a * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff); })();
const packU = (v, b) => { const pb = 8 / b, o = new Uint8Array(Math.ceil(v.length / pb)); v.forEach((x, i) => o[Math.floor(i / pb)] |= (x & ((1 << b) - 1)) << ((i % pb) * b)); return o; };
const i8b = (v) => { const o = new Uint8Array(v.length); v.forEach((x, i) => o[i] = x & 0xff); return o; };
const f32b = (a) => new Uint8Array(Float32Array.from(a).buffer.slice());
const bf16b = (a) => { const u = new Uint8Array(a.length * 2), dv = new DataView(u.buffer), fb = new DataView(new ArrayBuffer(4)); a.forEach((v, i) => { fb.setFloat32(0, v, true); dv.setUint16(i * 2, fb.getUint16(2, true), true); }); return u; };
const rA = (n, s = 1) => Array.from({ length: n }, () => (rng() * 2 - 1) * s);
const rU = (n, b) => Array.from({ length: n }, () => Math.floor(rng() * (1 << b)));
const rI = (n) => Array.from({ length: n }, () => Math.floor(rng() * 256) - 128);

const quantMod = {
  "^lm_head$": { num_bits: 2 }, "language_model\\.embed_tokens$": { num_bits: 2 },
  "language_model\\.embed_tokens_per_layer$": { num_bits: 4 },
  "language_model\\.layers\\.\\d+\\.mlp\\.": { num_bits: 4 },
  "language_model\\.layers\\.\\d+\\.per_layer_input_gate$": { num_bits: 8 },
  "language_model\\.layers\\.\\d+\\.per_layer_projection$": { num_bits: 8 },
  "language_model\\.layers\\.\\d+\\.self_attn\\.": { num_bits: 4 },
};
const cfg = new Gemma4Config({
  quantization_config: { module_quant_configs: quantMod },
  text_config: {
    vocab_size: 40, hidden_size: 32, intermediate_size: 64, num_hidden_layers: 3,
    num_attention_heads: 4, num_key_value_heads: 1, head_dim: 8, global_head_dim: 16,
    hidden_size_per_layer_input: 12, num_kv_shared_layers: 1, sliding_window: 3,
    rms_norm_eps: 1e-6, final_logit_softcapping: 30,
    rope_parameters: { full_attention: { rope_theta: 1e6, partial_rotary_factor: 0.25 }, sliding_attention: { rope_theta: 1e4 } },
    layer_types: ["sliding_attention", "full_attention", "sliding_attention"],
  },
});
const H = cfg.hidden_size, I = cfg.intermediate_size, V = cfg.vocab_size, P = cfg.hidden_size_per_layer_input, L = cfg.num_hidden_layers, Hq = cfg.num_attention_heads, Hkv = cfg.num_key_value_heads;
const T = {};
const addQ = (b, sc, N, K, bits, signed, scaleCols) => {
  const cols = signed ? K : K / (8 / bits);
  T[b] = { dtype: signed ? "I8" : "U8", shape: [N, cols], bytes: signed ? i8b(rI(N * cols)) : packU(rU(N * K, bits), bits) };
  T[sc] = { dtype: "F32", shape: [N, scaleCols], bytes: f32b(Array.from({ length: N * scaleCols }, () => 0.02 + rng() * 0.04)) };
};
const addBF = (b, sh) => { T[b] = { dtype: "BF16", shape: sh, bytes: bf16b(rA(sh.reduce((a, b) => a * b, 1), 0.05)) }; };
addQ("model.language_model.embed_tokens.embedding_quantized", "model.language_model.embed_tokens.embedding_scale", V, H, 2, false, 1);
addQ("model.language_model.embed_tokens_per_layer.embedding_quantized", "model.language_model.embed_tokens_per_layer.embedding_scale", V, L * P, 4, false, L);
addBF("model.language_model.per_layer_model_projection.weight", [L * P, H]);
addBF("model.language_model.per_layer_projection_norm.weight", [P]);
addBF("model.language_model.norm.weight", [H]);
addQ("lm_head.weight", "lm_head.weight_scale", V, H, 2, false, 1);
T["model.vision_tower.skip.weight"] = { dtype: "F32", shape: [3], bytes: f32b([1, 2, 3]) }; // must be skipped
for (let i = 0; i < L; i++) {
  const p = `model.language_model.layers.${i}`, Dh = cfg.headDim(i);
  for (const nm of ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm", "post_per_layer_input_norm"]) addBF(`${p}.${nm}.weight`, [H]);
  addBF(`${p}.self_attn.q_norm.weight`, [Dh]); addBF(`${p}.self_attn.k_norm.weight`, [Dh]);
  addQ(`${p}.self_attn.q_proj.weight`, `${p}.self_attn.q_proj.weight_scale`, Hq * Dh, H, 4, false, 1);
  addQ(`${p}.self_attn.k_proj.weight`, `${p}.self_attn.k_proj.weight_scale`, Hkv * Dh, H, 4, false, 1);
  addQ(`${p}.self_attn.v_proj.weight`, `${p}.self_attn.v_proj.weight_scale`, Hkv * Dh, H, 4, false, 1);
  addQ(`${p}.self_attn.o_proj.weight`, `${p}.self_attn.o_proj.weight_scale`, H, Hq * Dh, 4, false, 1);
  addQ(`${p}.mlp.gate_proj.weight`, `${p}.mlp.gate_proj.weight_scale`, I, H, 4, false, 1);
  addQ(`${p}.mlp.up_proj.weight`, `${p}.mlp.up_proj.weight_scale`, I, H, 4, false, 1);
  addQ(`${p}.mlp.down_proj.weight`, `${p}.mlp.down_proj.weight_scale`, H, I, 4, false, 1);
  addQ(`${p}.per_layer_input_gate.weight`, `${p}.per_layer_input_gate.weight_scale`, P, H, 8, true, 1);
  addQ(`${p}.per_layer_projection.weight`, `${p}.per_layer_projection.weight_scale`, H, P, 8, true, 1);
}
const blob = buildSafetensors(T);
const st = new SafetensorsFile(blob);
async function* chunker(buf, size) { const u = new Uint8Array(buf); for (let i = 0; i < u.length; i += size) yield u.subarray(i, Math.min(i + size, u.length)); }

const ids = [2, 5, 11, 19, 30], pos = ids.map((_, i) => i);
// CPU reference via non-streaming loader
const cl = await new Gemma4Model(cfg, C, new ModelWeights(st, cfg, C)).forward(ids, pos);
// GPU via STREAMING loader (the real browser path)
const parts = await assembleFromStream({ chunks: chunker(blob, 4096), header: st.header, dataStart: st.dataStart, backend: G });
const gl = await new Gemma4Model(cfg, G, new StreamedWeights(parts, cfg, G)).forward(ids, pos);

assert.ok(gl.every(Number.isFinite), "streamed GPU logits finite");
let max = 0; for (let i = 0; i < V; i++) max = Math.max(max, Math.abs(cl[i] - gl[i]));
assert.ok(max < 5e-3, `streamed GPU ≡ CPU (max diff ${max.toExponential(2)})`);
console.log(`  ✓ streaming GPU path runs full model, GPU≡CPU (max diff ${max.toExponential(2)})`);
console.log("\nstreaming GPU integration passed");
process.exit(0);
