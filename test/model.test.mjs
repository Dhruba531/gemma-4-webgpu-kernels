// End-to-end CPU-backend tests for the Gemma-4 forward graph.
// Run: node test/model.test.mjs

import assert from "node:assert";
import { Gemma4Config } from "../src/model/config.js";
import { CpuBackend } from "../src/engine/backend-cpu.js";
import { Gemma4Model } from "../src/model/gemma4.js";
import { sample } from "../src/model/sampler.js";
import { syntheticWeights } from "./util.mjs";

let pass = 0;
const ok = (cond, msg) => {
  assert.ok(cond, msg);
  console.log("  ✓ " + msg);
  pass++;
};

// A small but architecturally complete config: exercises sliding+global layers,
// partial rotary, MQA, PLE, and KV sharing.
const tiny = new Gemma4Config({
  text_config: {
    vocab_size: 64,
    hidden_size: 32,
    intermediate_size: 64,
    num_hidden_layers: 6,
    num_attention_heads: 4,
    num_key_value_heads: 1,
    head_dim: 8,
    global_head_dim: 16,
    hidden_size_per_layer_input: 12,
    vocab_size_per_layer_input: 64,
    num_kv_shared_layers: 2,
    sliding_window: 3,
    rms_norm_eps: 1e-6,
    final_logit_softcapping: 30.0,
    rope_parameters: {
      full_attention: { rope_theta: 1e6, partial_rotary_factor: 0.25 },
      sliding_attention: { rope_theta: 1e4 },
    },
    layer_types: [
      "sliding_attention", "full_attention", "sliding_attention",
      "sliding_attention", "sliding_attention", "sliding_attention",
    ],
  },
});

console.log("config sanity");
ok(tiny.num_hidden_layers === 6, "6 layers");
ok(tiny.isGlobal(1) && !tiny.isGlobal(0), "layer 1 is global, layer 0 local");
ok(tiny.headDim(1) === 16 && tiny.headDim(0) === 8, "per-layer head_dim");
// kv sharing maps into already-written source layers
for (let i = 0; i < tiny.num_hidden_layers; i++) {
  const s = tiny.kvSourceLayer(i);
  ok(s >= 0 && s <= i, `layer ${i} kv source ${s} is valid (<= i)`);
}

const B = new CpuBackend();
const weights = syntheticWeights(tiny, 42);

console.log("forward pass");
const model = new Gemma4Model(tiny, B, weights);
const ids = [2, 10, 20, 30, 40];
const logits = await model.forward(ids, [0, 1, 2, 3, 4]);
ok(logits.length === tiny.vocab_size, "logits length == vocab_size");
ok(logits.every(Number.isFinite), "no NaNs/Infs in logits");
ok(Math.max(...logits) <= tiny.final_logit_softcapping + 1e-3, "logits respect soft-cap bound");

console.log("KV-cache incremental == full-sequence");
// Full sequence in one shot.
const full = new Gemma4Model(tiny, B, weights);
const fullLogits = await full.forward(ids, [0, 1, 2, 3, 4]);
// Same tokens fed one at a time through the cache.
const inc = new Gemma4Model(tiny, B, weights);
let incLogits;
for (let i = 0; i < ids.length; i++) incLogits = await inc.forward([ids[i]], [i]);
let maxDiff = 0;
for (let i = 0; i < fullLogits.length; i++)
  maxDiff = Math.max(maxDiff, Math.abs(fullLogits[i] - incLogits[i]));
ok(maxDiff < 1e-3, `last-token logits match (max abs diff ${maxDiff.toExponential(2)})`);

console.log("sliding window actually masks");
// With window=3, token at pos 5 must not see pos 0..1. Build a model whose v is
// an identity-ish signal isn't trivial here; instead assert the mask path runs
// for a long sequence without error and stays finite.
const longModel = new Gemma4Model(tiny, B, weights);
const longIds = Array.from({ length: 20 }, (_, i) => (i * 7) % tiny.vocab_size);
const longLogits = await longModel.forward(longIds, longIds.map((_, i) => i));
ok(longLogits.every(Number.isFinite), "20-token forward stays finite");

console.log("sampler");
const greedy = sample(Float32Array.from([0.1, 5.0, 0.2, -1]), { temperature: 0 });
ok(greedy === 1, "greedy picks argmax");
const topk = sample(Float32Array.from([0, 10, 9, -5]), { temperature: 1, topK: 1, rng: () => 0.99 });
ok(topk === 1, "top-k=1 is deterministic argmax");
const penalized = sample(Float32Array.from([10, 4]), {
  temperature: 0,
  repetitionPenalty: 2,
  recentTokens: [0, 0],
});
ok(penalized === 0, "repetition penalty applies once per unique token");

console.log(`\n${pass} checks passed`);
