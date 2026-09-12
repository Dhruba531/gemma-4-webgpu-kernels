// Gemma-4 (E2B, text) forward graph.
//
// Backend-agnostic: every tensor op goes through `this.B`, so the exact same
// code runs on the CPU reference backend (Node tests) and the WebGPU backend
// (browser). Weights are fetched lazily by name through `this.weights.get(name)`.
//
// Architecture (from config.json of google/gemma-4-E2B-it-qat-mobile-transformers):
//   - 35 decoder layers, hidden 1536
//   - MQA: 8 query heads, 1 KV head; head_dim 256 (local) / 512 (global)
//   - global layers (every 5th) use partial rotary 0.25 + theta 1e6;
//     local/sliding layers use full rotary + theta 1e4, window 512
//   - per-layer query/key RMSNorm (and an UNWEIGHTED RMSNorm on v); attention
//     scores are NOT scaled by 1/sqrt(head_dim) (scale = 1.0)
//   - GeGLU MLP (gelu_tanh), pre/post norms around attn & ffn (4 norms/layer)
//   - Per-Layer Embeddings injected each layer (gate -> mul -> project -> norm),
//     then the whole residual stream is multiplied by the layer's `layer_scalar`
//   - KV cache shared across the last `num_kv_shared_layers` layers (shared
//     layers have no k/v projections)
//   - config carries final_logit_softcapping=30 but the runtime never applies it
//
// These details were cross-checked against the original working implementation
// in reference/gemma-4-e2b.js and the real checkpoint header
// (reference/header.json); norm weights are stored as the full multiplier.

import { KVCache } from "./kvcache.js";

export class Gemma4Model {
  constructor(config, backend, weights) {
    this.config = config;
    this.B = backend;
    this.weights = weights; // { get(name) -> tensor|null (cached) }
    this.cache = new KVCache(config, backend);
  }

  W(name) {
    const w = this.weights.get(name);
    if (!w) throw new Error(`missing weight: ${name}`);
    return w;
  }
  Wopt(name) {
    return this.weights.get(name);
  }

  reset() {
    this.cache.reset();
  }

  /**
   * Run one forward step over `ids` (token id array) at absolute `positions`.
   * Returns the logits (Float32Array, length vocab) for the LAST token only,
   * which is all sampling needs.
   */
  async forward(ids, positions) {
    const C = this.config;
    const B = this.B;
    const T = ids.length;

    // GPU backends bracket the pass in a memory frame: every intermediate
    // buffer allocated below is freed in endFrame(), keeping only the live KV
    // cache. (No-ops on the CPU backend.)
    B.beginFrame?.();

    // ---- token embedding (scaled by sqrt(hidden)) ----
    let h = B.embedRows(this.W("language_model.embed_tokens.weight"), ids, Math.sqrt(C.hidden_size));

    // ---- per-layer input embeddings (Gemma-3n PLE) ----
    // Two sources, combined:
    //   emb  = embed_tokens_per_layer(ids) · sqrt(Dple)        [T, L*Dple]
    //   proj = RMSNorm( per_layer_model_projection(h0) )        [T, L*Dple]
    //   ple  = (proj + emb) · rsqrt(2)
    const Dple = C.hidden_size_per_layer_input;
    const L = C.num_hidden_layers;
    const emb = B.embedRows(this.W("language_model.embed_tokens_per_layer.weight"), ids, Math.sqrt(Dple));

    let ple = emb;
    const projW = this.Wopt("language_model.per_layer_model_projection.weight");
    const projNorm = this.Wopt("language_model.per_layer_projection_norm.weight");
    if (projW && projNorm) {
      let proj = B.linear(h, projW); // [T, L*Dple]
      // RMSNorm each (token, layer) block of size Dple
      proj = B.reshape(proj, [T * L, Dple]);
      proj = B.rmsNorm(proj, projNorm, C.rms_norm_eps);
      proj = B.reshape(proj, [T, L * Dple]);
      ple = B.scale(B.add(proj, emb), Math.SQRT1_2);
    }

    this.cache.pushPositions(positions);

    for (let i = 0; i < C.num_hidden_layers; i++) {
      h = await this.decoderLayer(i, h, positions, ple, Dple);
    }

    // ---- final norm + lm head ----
    h = B.rmsNorm(h, this.W("language_model.norm.weight"), C.rms_norm_eps);
    // only need the last row for next-token prediction
    const last = B.sliceRows(h, T - 1, 1);
    const logits = B.linear(last, this.W("lm_head.weight"));
    // NOTE: config.final_logit_softcapping (30.0) is deliberately NOT applied.
    // The reference runtime parses it but never uses it, and the Gemma-3/4
    // families dropped logit soft-capping in favor of q/k norms; applying it
    // would only distort sampling temperatures.
    const out = await B.readback(logits); // Float32Array length vocab

    // The awaited readback above is the safe point: all GPU work for this pass
    // has finished, so free everything except the KV cache, plus the k/v
    // buffers this step's concat superseded.
    B.endFrame?.(this.cache.liveTensors(), this.cache.takeDead());
    return out;
  }

  async decoderLayer(i, h, positions, ple, Dple) {
    const C = this.config;
    const B = this.B;
    const p = `language_model.layers.${i}`;

    // === attention block ===
    let residual = h;
    let x = B.rmsNorm(h, this.W(`${p}.input_layernorm.weight`), C.rms_norm_eps);
    let attn = this.attention(i, x, positions);
    attn = B.rmsNorm(attn, this.W(`${p}.post_attention_layernorm.weight`), C.rms_norm_eps);
    h = B.add(residual, attn);

    // === feed-forward block (GeGLU) ===
    residual = h;
    x = B.rmsNorm(h, this.W(`${p}.pre_feedforward_layernorm.weight`), C.rms_norm_eps);
    const gate = B.linear(x, this.W(`${p}.mlp.gate_proj.weight`));
    const up = B.linear(x, this.W(`${p}.mlp.up_proj.weight`));
    let ff = B.geglu(gate, up);
    ff = B.linear(ff, this.W(`${p}.mlp.down_proj.weight`));
    ff = B.rmsNorm(ff, this.W(`${p}.post_feedforward_layernorm.weight`), C.rms_norm_eps);
    h = B.add(residual, ff);

    // === Per-Layer Embedding injection ===
    // gate(h) -> gelu -> elementwise * per-layer-input slice -> project -> norm -> add
    const gateW = this.Wopt(`${p}.per_layer_input_gate.weight`);
    const projW = this.Wopt(`${p}.per_layer_projection.weight`);
    if (gateW && projW) {
      // slice the layer-i block out of ple [T, L*Dple]
      const pleI = B.sliceCols(ple, i * Dple, Dple); // [T, Dple]
      const g = B.linear(h, gateW); // [T, Dple]
      let inj = B.geglu(g, pleI); // gelu(g) * pleI, one dispatch
      inj = B.linear(inj, projW); // [T, hidden]
      const normW = this.Wopt(`${p}.post_per_layer_input_norm.weight`);
      if (normW) inj = B.rmsNorm(inj, normW, C.rms_norm_eps);
      h = B.add(h, inj);
    }

    // trailing learned per-layer scalar on the whole residual stream
    // (h = (h + norm(inj)) * layer_scalar in the reference runtime)
    const ls = this.Wopt(`${p}.layer_scalar`);
    if (ls) h = B.scaleByTensor(h, ls);

    return h;
  }

  attention(i, x, positions) {
    const C = this.config;
    const B = this.B;
    const p = `language_model.layers.${i}`;
    const Hq = C.num_attention_heads;
    const Hkv = C.num_key_value_heads;
    const Dh = C.headDim(i);
    const T = x.shape[0];

    const { theta, partial_rotary_factor } = C.ropeFor(i);
    const rotaryDim = Math.round(Dh * partial_rotary_factor);

    let q = B.linear(x, this.W(`${p}.self_attn.q_proj.weight`)); // [T, Hq*Dh]
    q = B.reshape(q, [T, Hq, Dh]);
    const qn = this.Wopt(`${p}.self_attn.q_norm.weight`);
    if (qn) q = this.headNorm(q, qn, Hq, Dh);
    q = B.rope(q, positions, theta, Dh, Hq, rotaryDim);

    // KV cache (with sharing): source layers compute+write K/V; shared layers
    // skip the k/v projections entirely and read the source layer's cache
    // (same attention type, so head_dim and RoPE base always match).
    const src = C.kvSourceLayer(i);
    let kFull, vFull;
    if (src === i) {
      let k = B.linear(x, this.W(`${p}.self_attn.k_proj.weight`)); // [T, Hkv*Dh]
      let v = B.linear(x, this.W(`${p}.self_attn.v_proj.weight`)); // [T, Hkv*Dh]
      k = B.reshape(k, [T, Hkv, Dh]);
      v = B.reshape(v, [T, Hkv, Dh]);
      const kn = this.Wopt(`${p}.self_attn.k_norm.weight`);
      if (kn) k = this.headNorm(k, kn, Hkv, Dh);
      // v gets an UNWEIGHTED per-head RMSNorm (reference runtime does this)
      v = this.headNorm(v, null, Hkv, Dh);
      k = B.rope(k, positions, theta, Dh, Hkv, rotaryDim);
      ({ k: kFull, v: vFull } = this.cache.appendAndRead(src, k, v));
    } else {
      const slot = this.cache.read(src);
      kFull = slot.k;
      vFull = slot.v;
    }
    const kPos = this.cache.positions;

    const out = B.attention(q, kFull, vFull, {
      qPos: positions,
      kPos,
      // scale is exactly 1.0: q is per-head RMS-normalized above, and the
      // reference runtime applies no 1/sqrt(head_dim)
      scale: 1.0,
      slidingWindow: C.isGlobal(i) ? 0 : C.sliding_window,
      attnSoftcap: 0, // Gemma-3/4 dropped attention soft-capping
    });

    return B.linear(out, this.W(`${p}.self_attn.o_proj.weight`)); // [T, hidden]
  }

  // RMSNorm applied independently per head over the head_dim axis.
  headNorm(x, weight, heads, Dh) {
    const B = this.B;
    const T = x.shape[0];
    const flat = B.reshape(x, [T * heads, Dh]);
    const normed = B.rmsNorm(flat, weight, this.config.rms_norm_eps);
    return B.reshape(normed, [T, heads, Dh]);
  }
}
