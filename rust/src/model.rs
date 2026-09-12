//! Gemma-4 (E2B, text) forward graph.
//!
//! Backend-agnostic: every tensor op goes through `B: Backend`, so the exact
//! same code runs on the CPU reference backend (tests) and the wgpu backend.
//! Weights are fetched lazily by name through the [`WeightSource`].
//!
//! Architecture (from config.json of google/gemma-4-E2B-it-qat-mobile-transformers):
//!   - 35 decoder layers, hidden 1536
//!   - MQA: 8 query heads, 1 KV head; head_dim 256 (local) / 512 (global)
//!   - global layers (every 5th) use partial rotary 0.25 + theta 1e6;
//!     local/sliding layers use full rotary + theta 1e4, window 512
//!   - per-layer query/key RMSNorm (and an UNWEIGHTED RMSNorm on v); attention
//!     scores are NOT scaled by 1/sqrt(head_dim) (scale = 1.0)
//!   - GeGLU MLP (gelu_tanh), pre/post norms around attn & ffn (4 norms/layer)
//!   - Per-Layer Embeddings injected each layer (gate -> mul -> project -> norm),
//!     then the whole residual stream is multiplied by the layer's `layer_scalar`
//!   - KV cache shared across the last `num_kv_shared_layers` layers (shared
//!     layers have no k/v projections)
//!   - config carries final_logit_softcapping=30 but the runtime never applies it
//!
//! These details were cross-checked against the original working implementation
//! and the real checkpoint header; norm weights are stored as the full multiplier.

use std::sync::Arc;

use crate::backend::{AttnOpts, Backend, TensorShape};
use crate::config::Gemma4Config;
use crate::kvcache::KVCache;
use crate::weights::WeightSource;

pub struct Gemma4Model<B: Backend, W: WeightSource<B>> {
    pub config: Arc<Gemma4Config>,
    pub backend: Arc<B>,
    pub weights: W,
    pub cache: KVCache<B>,
}

impl<B: Backend, W: WeightSource<B>> Gemma4Model<B, W> {
    pub fn new(config: Arc<Gemma4Config>, backend: Arc<B>, weights: W) -> Self {
        let cache = KVCache::new(config.num_hidden_layers);
        Self { config, backend, weights, cache }
    }

    fn w(&self, name: &str) -> B::Tensor {
        self.weights.get(name).unwrap_or_else(|| panic!("missing weight: {name}"))
    }
    fn w_opt(&self, name: &str) -> Option<B::Tensor> {
        self.weights.get(name)
    }

    pub fn reset(&mut self) {
        self.cache.reset(&self.backend);
    }

    /// Run one forward step over `ids` at absolute `positions`. Returns the
    /// logits (length vocab) for the LAST token only, which is all sampling needs.
    pub fn forward(&mut self, ids: &[u32], positions: &[u32]) -> Vec<f32> {
        assert_eq!(ids.len(), positions.len(), "ids/positions length mismatch");
        let c = self.config.clone();
        let b = self.backend.clone();
        let t = ids.len();

        // GPU backends bracket the pass in a memory frame: every intermediate
        // buffer allocated below is freed in end_frame(), keeping only the live
        // KV cache. (No-op on the CPU backend.)
        b.begin_frame();

        // ---- token embedding (scaled by sqrt(hidden)) ----
        let mut h = b.embed_rows(&self.w("language_model.embed_tokens.weight"), ids, (c.hidden_size as f32).sqrt());

        // ---- per-layer input embeddings (Gemma-3n PLE) ----
        // Two sources, combined:
        //   emb  = embed_tokens_per_layer(ids) · sqrt(Dple)        [T, L*Dple]
        //   proj = RMSNorm( per_layer_model_projection(h0) )        [T, L*Dple]
        //   ple  = (proj + emb) · rsqrt(2)
        let dple = c.hidden_size_per_layer_input;
        let l = c.num_hidden_layers;
        let emb = b.embed_rows(&self.w("language_model.embed_tokens_per_layer.weight"), ids, (dple as f32).sqrt());

        let mut ple = emb.clone();
        if let (Some(proj_w), Some(proj_norm)) = (
            self.w_opt("language_model.per_layer_model_projection.weight"),
            self.w_opt("language_model.per_layer_projection_norm.weight"),
        ) {
            let mut proj = b.linear(&h, &proj_w); // [T, L*Dple]
            // RMSNorm each (token, layer) block of size Dple
            proj = b.reshape(&proj, &[t * l, dple]);
            proj = b.rms_norm(&proj, Some(&proj_norm), c.rms_norm_eps);
            proj = b.reshape(&proj, &[t, l * dple]);
            ple = b.scale(&b.add(&proj, &emb), std::f32::consts::FRAC_1_SQRT_2);
        }

        self.cache.push_positions(positions);

        for i in 0..c.num_hidden_layers {
            h = self.decoder_layer(i, h, positions, &ple, dple);
        }

        // ---- final norm + lm head ----
        h = b.rms_norm(&h, Some(&self.w("language_model.norm.weight")), c.rms_norm_eps);
        // only need the last row for next-token prediction
        let last = b.slice_rows(&h, t - 1, 1);
        let logits = b.linear(&last, &self.w("lm_head.weight"));
        // NOTE: config.final_logit_softcapping (30.0) is deliberately NOT applied.
        // The reference runtime parses it but never uses it, and the Gemma-3/4
        // families dropped logit soft-capping in favor of q/k norms.
        let out = b.readback(&logits);

        // The blocking readback above is the safe point: all GPU work for this
        // pass has finished, so free everything except the KV cache, plus the
        // k/v buffers this step's growth superseded.
        let keep = self.cache.live_tensors();
        let dead = self.cache.take_dead();
        b.end_frame(&keep, dead);
        out
    }

    fn decoder_layer(&mut self, i: usize, h: B::Tensor, positions: &[u32], ple: &B::Tensor, dple: usize) -> B::Tensor {
        let c = self.config.clone();
        let b = self.backend.clone();
        let p = format!("language_model.layers.{i}");
        let eps = c.rms_norm_eps;

        // === attention block ===
        let residual = h;
        let x = b.rms_norm(&residual, Some(&self.w(&format!("{p}.input_layernorm.weight"))), eps);
        let mut attn = self.attention(i, &x, positions);
        attn = b.rms_norm(&attn, Some(&self.w(&format!("{p}.post_attention_layernorm.weight"))), eps);
        let mut h = b.add(&residual, &attn);

        // === feed-forward block (GeGLU) ===
        let residual = h;
        let x = b.rms_norm(&residual, Some(&self.w(&format!("{p}.pre_feedforward_layernorm.weight"))), eps);
        let gate = b.linear(&x, &self.w(&format!("{p}.mlp.gate_proj.weight")));
        let up = b.linear(&x, &self.w(&format!("{p}.mlp.up_proj.weight")));
        let mut ff = b.geglu(&gate, &up);
        ff = b.linear(&ff, &self.w(&format!("{p}.mlp.down_proj.weight")));
        ff = b.rms_norm(&ff, Some(&self.w(&format!("{p}.post_feedforward_layernorm.weight"))), eps);
        h = b.add(&residual, &ff);

        // === Per-Layer Embedding injection ===
        // gate(h) -> gelu -> elementwise * per-layer-input slice -> project -> norm -> add
        if let (Some(gate_w), Some(proj_w)) =
            (self.w_opt(&format!("{p}.per_layer_input_gate.weight")), self.w_opt(&format!("{p}.per_layer_projection.weight")))
        {
            let ple_i = b.slice_cols(ple, i * dple, dple); // [T, Dple]
            let g = b.linear(&h, &gate_w); // [T, Dple]
            let mut inj = b.geglu(&g, &ple_i); // gelu(g) * pleI, one dispatch
            inj = b.linear(&inj, &proj_w); // [T, hidden]
            if let Some(norm_w) = self.w_opt(&format!("{p}.post_per_layer_input_norm.weight")) {
                inj = b.rms_norm(&inj, Some(&norm_w), eps);
            }
            h = b.add(&h, &inj);
        }

        // trailing learned per-layer scalar on the whole residual stream
        if let Some(ls) = self.w_opt(&format!("{p}.layer_scalar")) {
            h = b.scale_by_tensor(&h, &ls);
        }
        h
    }

    fn attention(&mut self, i: usize, x: &B::Tensor, positions: &[u32]) -> B::Tensor {
        let c = self.config.clone();
        let b = self.backend.clone();
        let p = format!("language_model.layers.{i}");
        let hq = c.num_attention_heads;
        let hkv = c.num_key_value_heads;
        let dh = c.head_dim(i);
        let t = x.shape()[0];

        let rope = c.rope_for(i);
        let rotary_dim = (dh as f32 * rope.partial_rotary_factor).round() as usize;

        let mut q = b.linear(x, &self.w(&format!("{p}.self_attn.q_proj.weight"))); // [T, Hq*Dh]
        q = b.reshape(&q, &[t, hq, dh]);
        if let Some(qn) = self.w_opt(&format!("{p}.self_attn.q_norm.weight")) {
            q = self.head_norm(&q, Some(&qn), hq, dh);
        }
        q = b.rope(&q, positions, rope.theta, dh, hq, rotary_dim);

        // KV cache (with sharing): source layers compute+write K/V; shared layers
        // skip the k/v projections entirely and read the source layer's cache.
        let src = c.kv_source_layer(i);
        let (k_full, v_full) = if src == i {
            let mut k = b.linear(x, &self.w(&format!("{p}.self_attn.k_proj.weight"))); // [T, Hkv*Dh]
            let mut v = b.linear(x, &self.w(&format!("{p}.self_attn.v_proj.weight")));
            k = b.reshape(&k, &[t, hkv, dh]);
            v = b.reshape(&v, &[t, hkv, dh]);
            if let Some(kn) = self.w_opt(&format!("{p}.self_attn.k_norm.weight")) {
                k = self.head_norm(&k, Some(&kn), hkv, dh);
            }
            // v gets an UNWEIGHTED per-head RMSNorm (reference runtime does this)
            v = self.head_norm(&v, None, hkv, dh);
            k = b.rope(&k, positions, rope.theta, dh, hkv, rotary_dim);
            self.cache.append_and_read(&b, src, &k, &v)
        } else {
            let slot = self.cache.read(src).expect("shared KV source layer has no cache");
            (slot.k.clone(), slot.v.clone())
        };

        let out = b.attention(
            &q,
            &k_full,
            &v_full,
            &AttnOpts {
                q_pos: positions,
                k_pos: &self.cache.positions,
                // scale is exactly 1.0: q is per-head RMS-normalized above, and the
                // reference runtime applies no 1/sqrt(head_dim)
                scale: 1.0,
                sliding_window: if c.is_global(i) { 0 } else { c.sliding_window },
                attn_softcap: 0.0, // Gemma-3/4 dropped attention soft-capping
            },
        );

        b.linear(&out, &self.w(&format!("{p}.self_attn.o_proj.weight"))) // [T, hidden]
    }

    /// RMSNorm applied independently per head over the head_dim axis.
    fn head_norm(&self, x: &B::Tensor, weight: Option<&B::Tensor>, heads: usize, dh: usize) -> B::Tensor {
        let b = &self.backend;
        let t = x.shape()[0];
        let flat = b.reshape(x, &[t * heads, dh]);
        let normed = b.rms_norm(&flat, weight, self.config.rms_norm_eps);
        b.reshape(&normed, &[t, heads, dh])
    }
}
