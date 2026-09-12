// Test helpers: deterministic RNG + a synthetic weight provider that fabricates
// correctly-shaped random tensors on demand for a given Gemma4Config.

import { CpuBackend } from "../src/engine/backend-cpu.js";

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function () {
    a |= 0;
    a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// Build the expected shape for a weight name under config C.
export function shapeOf(name, C) {
  const H = C.hidden_size;
  const I = C.intermediate_size;
  const V = C.vocab_size;
  const P = C.hidden_size_per_layer_input;
  const L = C.num_hidden_layers;
  const Hq = C.num_attention_heads;
  const Hkv = C.num_key_value_heads;

  if (name === "language_model.embed_tokens.weight") return [V, H];
  if (name === "language_model.embed_tokens_per_layer.weight") return [V, L * P];
  if (name === "language_model.per_layer_model_projection.weight") return [L * P, H];
  if (name === "language_model.per_layer_projection_norm.weight") return [P];
  if (name === "language_model.norm.weight") return [H];
  if (name === "lm_head.weight") return [V, H];

  const m = /^language_model\.layers\.(\d+)\.(.+)$/.exec(name);
  if (m) {
    const i = +m[1];
    const Dh = C.headDim(i);
    const sub = m[2];
    const table = {
      "input_layernorm.weight": [H],
      "post_attention_layernorm.weight": [H],
      "pre_feedforward_layernorm.weight": [H],
      "post_feedforward_layernorm.weight": [H],
      "post_per_layer_input_norm.weight": [H],
      "self_attn.q_proj.weight": [Hq * Dh, H],
      "self_attn.k_proj.weight": [Hkv * Dh, H],
      "self_attn.v_proj.weight": [Hkv * Dh, H],
      "self_attn.o_proj.weight": [H, Hq * Dh],
      "self_attn.q_norm.weight": [Dh],
      "self_attn.k_norm.weight": [Dh],
      "mlp.gate_proj.weight": [I, H],
      "mlp.up_proj.weight": [I, H],
      "mlp.down_proj.weight": [H, I],
      "per_layer_input_gate.weight": [P, H],
      "per_layer_projection.weight": [H, P],
      "layer_scalar": [1],
    };
    if (table[sub]) return table[sub];
  }
  throw new Error("unknown weight " + name);
}

export function syntheticWeights(C, seed = 1) {
  const rng = mulberry32(seed);
  const cache = new Map();
  return {
    get(name) {
      if (cache.has(name)) return cache.get(name);
      const shape = shapeOf(name, C);
      const n = shape.reduce((a, b) => a * b, 1);
      const data = new Float32Array(n);
      const isNorm = name.endsWith("norm.weight") || name.endsWith("layernorm.weight");
      const isScalar = name.endsWith("layer_scalar");
      // norms and layer_scalar store the FULL multiplier: keep near 1.
      // linear weights: small so activations stay in a sane range.
      const base = isNorm || isScalar ? 1 : 0;
      const scale = isNorm || isScalar ? 0.05 : 0.04;
      for (let i = 0; i < n; i++) data[i] = base + (rng() * 2 - 1) * scale;
      const t = new CpuBackend().tensor(data, shape);
      cache.set(name, t);
      return t;
    },
  };
}
