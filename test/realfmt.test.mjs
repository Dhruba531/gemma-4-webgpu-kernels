// Integration test for the REAL checkpoint format (without the 2.46 GB download).
// Builds a tiny safetensors blob using the actual on-disk naming + dtypes
// (U8/I8 packed weights + scales, BF16 norms, model.language_model.* prefix,
// .embedding_quantized), then runs the full model through ModelWeights on both
// backends and checks GPU≡CPU. Run: node test/realfmt.test.mjs

import assert from "node:assert";
import { create, globals } from "webgpu";
Object.assign(globalThis, globals);
import { CpuBackend } from "../src/engine/backend-cpu.js";
import { WebGpuBackend } from "../src/engine/backend-webgpu.js";
import { Gemma4Config } from "../src/model/config.js";
import { Gemma4Model } from "../src/model/gemma4.js";
import { SafetensorsFile, buildSafetensors } from "../src/engine/safetensors.js";
import { ModelWeights } from "../src/engine/weights.js";

const gpu = create([]);
const device = await (await gpu.requestAdapter()).requestDevice();
const G = new WebGpuBackend(device), C = new CpuBackend();
const rng = (() => { let a = 5; return () => ((a = (a * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff); })();

// --- encoders ---
const packU = (vals, bits) => { const pb = 8 / bits, o = new Uint8Array(Math.ceil(vals.length / pb)); vals.forEach((v, i) => o[Math.floor(i / pb)] |= (v & ((1 << bits) - 1)) << ((i % pb) * bits)); return o; };
const i8bytes = (vals) => { const o = new Uint8Array(vals.length); vals.forEach((v, i) => o[i] = v & 0xff); return o; };
const f32bytes = (arr) => new Uint8Array(Float32Array.from(arr).buffer.slice());
const bf16bytes = (arr) => { const u = new Uint8Array(arr.length * 2), dv = new DataView(u.buffer), fb = new DataView(new ArrayBuffer(4)); arr.forEach((v, i) => { fb.setFloat32(0, v, true); dv.setUint16(i * 2, fb.getUint16(2, true), true); }); return u; };
const randArr = (n, s = 1) => Array.from({ length: n }, () => (rng() * 2 - 1) * s);
const randU = (n, bits) => Array.from({ length: n }, () => Math.floor(rng() * (1 << bits)));
const randI8 = (n) => Array.from({ length: n }, () => Math.floor(rng() * 256) - 128);

const quantMod = {
  "^lm_head$": { num_bits: 2 },
  "language_model\\.embed_tokens$": { num_bits: 2 },
  "language_model\\.embed_tokens_per_layer$": { num_bits: 4 },
  "language_model\\.layers\\.(\\d|1[0-4])\\.mlp\\.": { num_bits: 4 },
  "language_model\\.layers\\.\\d+\\.mlp\\.": { num_bits: 2 },
  "language_model\\.layers\\.\\d+\\.per_layer_input_gate$": { num_bits: 8 },
  "language_model\\.layers\\.\\d+\\.per_layer_projection$": { num_bits: 8 },
  "language_model\\.layers\\.\\d+\\.self_attn\\.": { num_bits: 4 },
};

const cfg = new Gemma4Config({
  quantization_config: { module_quant_configs: quantMod },
  text_config: {
    vocab_size: 40, hidden_size: 32, intermediate_size: 64, num_hidden_layers: 6,
    num_attention_heads: 4, num_key_value_heads: 1, head_dim: 8, global_head_dim: 16,
    hidden_size_per_layer_input: 12, vocab_size_per_layer_input: 40, num_kv_shared_layers: 2,
    sliding_window: 3, rms_norm_eps: 1e-6, final_logit_softcapping: 30,
    rope_parameters: { full_attention: { rope_theta: 1e6, partial_rotary_factor: 0.25 }, sliding_attention: { rope_theta: 1e4 } },
    layer_types: ["sliding_attention", "full_attention", "sliding_attention", "sliding_attention", "sliding_attention", "sliding_attention"],
  },
});

const H = cfg.hidden_size, I = cfg.intermediate_size, V = cfg.vocab_size, Dple = cfg.hidden_size_per_layer_input, L = cfg.num_hidden_layers, Hq = cfg.num_attention_heads, Hkv = cfg.num_key_value_heads;
const T = {};
const addQ = (real, scaleReal, N, K, bits, signed, scaleCols) => {
  T[real] = { dtype: signed ? "I8" : "U8", shape: [N, signed ? K : K / (8 / bits)], bytes: signed ? i8bytes(randI8(N * K)) : packU(randU(N * (K / (8 / bits)) * (8 / bits), bits).slice(0, N * K), bits) };
  // ensure exact byte length
  const cols = signed ? K : K / (8 / bits);
  T[real].bytes = signed ? i8bytes(randI8(N * cols)) : packU(randU(N * K, bits), bits);
  T[real].shape = [N, cols];
  T[scaleReal] = { dtype: "F32", shape: [N, scaleCols], bytes: f32bytes(Array.from({ length: N * scaleCols }, () => 0.02 + rng() * 0.05)) };
};
const addBF = (real, shape) => { T[real] = { dtype: "BF16", shape, bytes: bf16bytes(randArr(shape.reduce((a, b) => a * b, 1), 0.05)) }; };

// embeddings + head + globals
addQ("model.language_model.embed_tokens.embedding_quantized", "model.language_model.embed_tokens.embedding_scale", V, H, 2, false, 1);
addQ("model.language_model.embed_tokens_per_layer.embedding_quantized", "model.language_model.embed_tokens_per_layer.embedding_scale", V, L * Dple, 4, false, L);
addBF("model.language_model.per_layer_model_projection.weight", [L * Dple, H]);
addBF("model.language_model.per_layer_projection_norm.weight", [Dple]);
addBF("model.language_model.norm.weight", [H]);
addQ("lm_head.weight", "lm_head.weight_scale", V, H, 2, false, 1);

for (let i = 0; i < L; i++) {
  const p = `model.language_model.layers.${i}`;
  const Dh = cfg.headDim(i);
  for (const nm of ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm", "post_per_layer_input_norm"]) addBF(`${p}.${nm}.weight`, [H]);
  addBF(`${p}.layer_scalar`, [1]);
  addBF(`${p}.self_attn.q_norm.weight`, [Dh]);
  addBF(`${p}.self_attn.k_norm.weight`, [Dh]);
  addQ(`${p}.self_attn.q_proj.weight`, `${p}.self_attn.q_proj.weight_scale`, Hq * Dh, H, 4, false, 1);
  addQ(`${p}.self_attn.k_proj.weight`, `${p}.self_attn.k_proj.weight_scale`, Hkv * Dh, H, 4, false, 1);
  addQ(`${p}.self_attn.v_proj.weight`, `${p}.self_attn.v_proj.weight_scale`, Hkv * Dh, H, 4, false, 1);
  addQ(`${p}.self_attn.o_proj.weight`, `${p}.self_attn.o_proj.weight_scale`, H, Hq * Dh, 4, false, 1);
  addQ(`${p}.mlp.gate_proj.weight`, `${p}.mlp.gate_proj.weight_scale`, I, H, 4, false, 1);
  addQ(`${p}.mlp.up_proj.weight`, `${p}.mlp.up_proj.weight_scale`, I, H, 4, false, 1);
  addQ(`${p}.mlp.down_proj.weight`, `${p}.mlp.down_proj.weight_scale`, H, I, 4, false, 1);
  addQ(`${p}.per_layer_input_gate.weight`, `${p}.per_layer_input_gate.weight_scale`, Dple, H, 8, true, 1);
  addQ(`${p}.per_layer_projection.weight`, `${p}.per_layer_projection.weight_scale`, H, Dple, 8, true, 1);
}

const blob = buildSafetensors(T);
const fileC = new SafetensorsFile(blob), fileG = new SafetensorsFile(blob.slice(0));
const wC = new ModelWeights(fileC, cfg, C);
const wG = new ModelWeights(fileG, cfg, G);

const ids = [2, 5, 11, 19, 30], pos = ids.map((_, i) => i);
console.log("real-format integration");
const cl = await new Gemma4Model(cfg, C, wC).forward(ids, pos);
const gl = await new Gemma4Model(cfg, G, wG).forward(ids, pos);
assert.ok(cl.length === V, "logits length");
assert.ok(cl.every(Number.isFinite), "CPU logits finite");
assert.ok(gl.every(Number.isFinite), "GPU logits finite");
let max = 0; for (let i = 0; i < V; i++) max = Math.max(max, Math.abs(cl[i] - gl[i]));
assert.ok(max < 5e-3, `GPU≡CPU on real format (max diff ${max.toExponential(2)})`);
console.log(`  ✓ loads real names/dtypes, runs both backends, GPU≡CPU (max diff ${max.toExponential(2)})`);
console.log("\nintegration passed");
process.exit(0);
