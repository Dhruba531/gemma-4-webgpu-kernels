//! Gemma-4 (E2B, text) forward graph.
//!
//! Backend-agnostic: every tensor op goes through `B: Backend`, so the exact
//! same code runs on the CPU reference backend (tests) and the wgpu backend.
//! Weights are fetched lazily by name through the [`WeightSource`] the first
//! time a layer runs and kept as resolved handles from then on, so the
//! per-token path does no name formatting or map lookups.
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
//!
//! The graph is written against the backend's fused ops where Gemma-4's
//! structure allows it (`linear_geglu` for gate+up+GeGLU, `add_rms_norm` for
//! post-norm+residual(+layer_scalar), `linear_gelu_mul_cols` for the PLE gate);
//! their default implementations compose the primitives, so the CPU reference
//! is unchanged while the GPU records one dispatch each.

use std::sync::Arc;

use crate::backend::{AttnOpts, Backend, TensorShape};
use crate::config::Gemma4Config;
use crate::kvcache::KVCache;
use crate::weights::WeightSource;

/// Resolved weight handles for one decoder layer.
struct LayerWeights<T> {
    input_ln: T,
    q_proj: T,
    q_norm: Option<T>,
    k_proj: Option<T>,
    v_proj: Option<T>,
    k_norm: Option<T>,
    o_proj: T,
    post_attn_ln: T,
    pre_ffn_ln: T,
    gate_proj: T,
    up_proj: T,
    down_proj: T,
    post_ffn_ln: T,
    ple_gate: Option<T>,
    ple_proj: Option<T>,
    ple_norm: Option<T>,
    layer_scalar: Option<T>,
}

/// Resolved handles for the embedding / head side of the graph.
struct TopWeights<T> {
    embed: T,
    embed_per_layer: T,
    ple_proj: Option<T>,
    ple_proj_norm: Option<T>,
    final_norm: T,
    lm_head: T,
}

pub struct Gemma4Model<B: Backend, W: WeightSource<B>> {
    pub config: Arc<Gemma4Config>,
    pub backend: Arc<B>,
    pub weights: W,
    pub cache: KVCache<B>,
    layers: Vec<Option<LayerWeights<B::Tensor>>>,
    top: Option<TopWeights<B::Tensor>>,
}

impl<B: Backend, W: WeightSource<B>> Gemma4Model<B, W> {
    pub fn new(config: Arc<Gemma4Config>, backend: Arc<B>, weights: W) -> Self {
        let cache = KVCache::new(config.num_hidden_layers);
        let layers = (0..config.num_hidden_layers).map(|_| None).collect();
        Self { config, backend, weights, cache, layers, top: None }
    }

    fn w(&self, name: &str) -> B::Tensor {
        self.weights.get(name).unwrap_or_else(|| panic!("missing weight: {name}"))
    }
    fn w_opt(&self, name: &str) -> Option<B::Tensor> {
        self.weights.get(name)
    }

    fn resolve_top(&mut self) {
        if self.top.is_some() {
            return;
        }
        self.top = Some(TopWeights {
            embed: self.w("language_model.embed_tokens.weight"),
            embed_per_layer: self.w("language_model.embed_tokens_per_layer.weight"),
            ple_proj: self.w_opt("language_model.per_layer_model_projection.weight"),
            ple_proj_norm: self.w_opt("language_model.per_layer_projection_norm.weight"),
            final_norm: self.w("language_model.norm.weight"),
            lm_head: self.w("lm_head.weight"),
        });
    }

    fn resolve_layer(&mut self, i: usize) {
        if self.layers[i].is_some() {
            return;
        }
        let p = format!("language_model.layers.{i}");
        let src = self.config.kv_source_layer(i);
        let owns_kv = src == i;
        let lw = LayerWeights {
            input_ln: self.w(&format!("{p}.input_layernorm.weight")),
            q_proj: self.w(&format!("{p}.self_attn.q_proj.weight")),
            q_norm: self.w_opt(&format!("{p}.self_attn.q_norm.weight")),
            k_proj: owns_kv.then(|| self.w(&format!("{p}.self_attn.k_proj.weight"))),
            v_proj: owns_kv.then(|| self.w(&format!("{p}.self_attn.v_proj.weight"))),
            k_norm: if owns_kv { self.w_opt(&format!("{p}.self_attn.k_norm.weight")) } else { None },
            o_proj: self.w(&format!("{p}.self_attn.o_proj.weight")),
            post_attn_ln: self.w(&format!("{p}.post_attention_layernorm.weight")),
            pre_ffn_ln: self.w(&format!("{p}.pre_feedforward_layernorm.weight")),
            gate_proj: self.w(&format!("{p}.mlp.gate_proj.weight")),
            up_proj: self.w(&format!("{p}.mlp.up_proj.weight")),
            down_proj: self.w(&format!("{p}.mlp.down_proj.weight")),
            post_ffn_ln: self.w(&format!("{p}.post_feedforward_layernorm.weight")),
            ple_gate: self.w_opt(&format!("{p}.per_layer_input_gate.weight")),
            ple_proj: self.w_opt(&format!("{p}.per_layer_projection.weight")),
            ple_norm: self.w_opt(&format!("{p}.post_per_layer_input_norm.weight")),
            layer_scalar: self.w_opt(&format!("{p}.layer_scalar")),
        };
        self.layers[i] = Some(lw);
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
        self.resolve_top();
        for i in 0..c.num_hidden_layers {
            self.resolve_layer(i);
        }

        // GPU backends bracket the pass in a memory frame: every intermediate
        // buffer allocated below is freed in end_frame(), keeping only the live
        // KV cache. (No-op on the CPU backend.)
        b.begin_frame();

        let top = self.top.as_ref().unwrap();

        // ---- token embedding (scaled by sqrt(hidden)) ----
        let mut h = b.embed_rows(&top.embed, ids, (c.hidden_size as f32).sqrt());

        // ---- per-layer input embeddings (Gemma-3n PLE) ----
        // Two sources, combined:
        //   emb  = embed_tokens_per_layer(ids) · sqrt(Dple)        [T, L*Dple]
        //   proj = RMSNorm( per_layer_model_projection(h0) )        [T, L*Dple]
        //   ple  = (proj + emb) · rsqrt(2)
        let dple = c.hidden_size_per_layer_input;
        let l = c.num_hidden_layers;
        let emb = b.embed_rows(&top.embed_per_layer, ids, (dple as f32).sqrt());

        let mut ple = emb.clone();
        if let (Some(proj_w), Some(proj_norm)) = (&top.ple_proj, &top.ple_proj_norm) {
            let mut proj = b.linear(&h, proj_w); // [T, L*Dple]
            // RMSNorm each (token, layer) block of size Dple
            proj = b.reshape(&proj, &[t * l, dple]);
            proj = b.rms_norm(&proj, Some(proj_norm), c.rms_norm_eps);
            proj = b.reshape(&proj, &[t, l * dple]);
            ple = b.scale(&b.add(&proj, &emb), std::f32::consts::FRAC_1_SQRT_2);
        }

        self.cache.push_positions(positions);

        for i in 0..c.num_hidden_layers {
            h = self.decoder_layer(i, h, positions, &ple, dple);
        }

        // ---- final norm + lm head ----
        let top = self.top.as_ref().unwrap();
        h = b.rms_norm(&h, Some(&top.final_norm), c.rms_norm_eps);
        // only need the last row for next-token prediction
        let last = b.slice_rows(&h, t - 1, 1);
        let logits = b.linear(&last, &top.lm_head);
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
        let b = self.backend.clone();
        let eps = self.config.rms_norm_eps;

        // === attention block ===
        let residual = h;
        let attn = {
            let lw = self.layers[i].as_ref().unwrap();
            let x = b.rms_norm(&residual, Some(&lw.input_ln), eps);
            self.attention(i, &x, positions)
        };
        let lw = self.layers[i].as_ref().unwrap();
        // h = residual + post_attn_norm(attn)
        let mut h = b.add_rms_norm(&residual, &attn, &lw.post_attn_ln, eps, None);

        // === feed-forward block (GeGLU) ===
        let residual = h;
        let x = b.rms_norm(&residual, Some(&lw.pre_ffn_ln), eps);
        let ff = b.linear_geglu(&x, &lw.gate_proj, &lw.up_proj); // gelu(gate) * up
        let ff = b.linear(&ff, &lw.down_proj);
        h = b.add_rms_norm(&residual, &ff, &lw.post_ffn_ln, eps, None);

        // === Per-Layer Embedding injection ===
        // gate(h) -> gelu -> elementwise * per-layer-input slice -> project -> norm -> add,
        // then the trailing learned per-layer scalar on the whole residual stream.
        let ls = lw.layer_scalar.as_ref();
        if let (Some(gate_w), Some(proj_w)) = (&lw.ple_gate, &lw.ple_proj) {
            let inj = b.linear_gelu_mul_cols(&h, gate_w, ple, i * dple); // [T, Dple]
            let inj = b.linear(&inj, proj_w); // [T, hidden]
            match &lw.ple_norm {
                Some(norm_w) => return b.add_rms_norm(&h, &inj, norm_w, eps, ls),
                None => h = b.add(&h, &inj),
            }
        }
        match ls {
            Some(ls) => b.scale_by_tensor(&h, ls),
            None => h,
        }
    }

    fn attention(&mut self, i: usize, x: &B::Tensor, positions: &[u32]) -> B::Tensor {
        let c = self.config.clone();
        let b = self.backend.clone();
        let lw = self.layers[i].as_ref().unwrap();
        let hq = c.num_attention_heads;
        let hkv = c.num_key_value_heads;
        let dh = c.head_dim(i);
        let t = x.shape()[0];

        let rope = c.rope_for(i);
        let rotary_dim = (dh as f32 * rope.partial_rotary_factor).round() as usize;

        let mut q = b.linear(x, &lw.q_proj); // [T, Hq*Dh]
        q = b.reshape(&q, &[t, hq, dh]);
        // per-head q norm + RoPE (one fused dispatch on the GPU)
        q = match &lw.q_norm {
            Some(qn) => b.head_norm_rope(&q, qn, eps_of(&c), positions, rope.theta, dh, hq, rotary_dim),
            None => b.rope(&q, positions, rope.theta, dh, hq, rotary_dim),
        };

        // KV cache (with sharing): source layers compute+write K/V; shared layers
        // skip the k/v projections entirely and read the source layer's cache.
        let src = c.kv_source_layer(i);
        let (k_full, v_full) = if src == i {
            let mut k = b.linear(x, lw.k_proj.as_ref().expect("k_proj")); // [T, Hkv*Dh]
            let mut v = b.linear(x, lw.v_proj.as_ref().expect("v_proj"));
            k = b.reshape(&k, &[t, hkv, dh]);
            v = b.reshape(&v, &[t, hkv, dh]);
            k = match &lw.k_norm {
                Some(kn) => b.head_norm_rope(&k, kn, eps_of(&c), positions, rope.theta, dh, hkv, rotary_dim),
                None => b.rope(&k, positions, rope.theta, dh, hkv, rotary_dim),
            };
            // v gets an UNWEIGHTED per-head RMSNorm (reference runtime does this)
            v = Self::head_norm(&b, eps_of(&c), &v, None, hkv, dh);
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

        let lw = self.layers[i].as_ref().unwrap();
        b.linear(&out, &lw.o_proj) // [T, hidden]
    }

    /// RMSNorm applied independently per head over the head_dim axis.
    fn head_norm(b: &B, eps: f32, x: &B::Tensor, weight: Option<&B::Tensor>, heads: usize, dh: usize) -> B::Tensor {
        let t = x.shape()[0];
        let flat = b.reshape(x, &[t * heads, dh]);
        let normed = b.rms_norm(&flat, weight, eps);
        b.reshape(&normed, &[t, heads, dh])
    }
}

fn eps_of(c: &Gemma4Config) -> f32 {
    c.rms_norm_eps
}
