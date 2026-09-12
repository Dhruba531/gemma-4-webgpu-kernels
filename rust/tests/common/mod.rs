//! Test helpers: deterministic RNG + a synthetic weight provider that
//! fabricates correctly-shaped random tensors on demand for a Gemma4Config.
#![allow(dead_code)]

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use gemma_wgpu::backend::Backend;
use gemma_wgpu::backend_cpu::{CpuBackend, CpuTensor};
use gemma_wgpu::config::Gemma4Config;
use gemma_wgpu::weights::WeightSource;

/// mulberry32, as in test/util.mjs.
pub struct Mulberry(u32);
impl Mulberry {
    pub fn new(seed: u32) -> Self {
        Self(seed)
    }
    pub fn next(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x6d2b79f5);
        let mut t = self.0;
        t = (t ^ (t >> 15)).wrapping_mul(1 | t);
        t = (t.wrapping_add((t ^ (t >> 7)).wrapping_mul(61 | t))) ^ t;
        ((t ^ (t >> 14)) as f64) / 4294967296.0
    }
}

/// LCG used by the JS GPU tests.
pub struct Lcg(u32);
impl Lcg {
    pub fn new(seed: u32) -> Self {
        Self(seed)
    }
    pub fn next(&mut self) -> f32 {
        self.0 = (self.0.wrapping_mul(1103515245).wrapping_add(12345)) & 0x7fffffff;
        self.0 as f32 / 0x7fffffff as f32
    }
    pub fn rand(&mut self, n: usize, s: f32) -> Vec<f32> {
        (0..n).map(|_| (self.next() * 2.0 - 1.0) * s).collect()
    }
}

pub fn tiny_config_json(vocab: usize) -> serde_json::Value {
    serde_json::json!({ "text_config": {
        "vocab_size": vocab, "hidden_size": 32, "intermediate_size": 64, "num_hidden_layers": 6,
        "num_attention_heads": 4, "num_key_value_heads": 1, "head_dim": 8, "global_head_dim": 16,
        "hidden_size_per_layer_input": 12, "vocab_size_per_layer_input": vocab, "num_kv_shared_layers": 2,
        "sliding_window": 3, "rms_norm_eps": 1e-6, "final_logit_softcapping": 30.0,
        "rope_parameters": { "full_attention": { "rope_theta": 1e6, "partial_rotary_factor": 0.25 }, "sliding_attention": { "rope_theta": 1e4 } },
        "layer_types": ["sliding_attention","full_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention"],
    }})
}

pub fn tiny_config(vocab: usize) -> Arc<Gemma4Config> {
    Arc::new(Gemma4Config::from_json(&tiny_config_json(vocab)))
}

/// Expected shape for a logical weight name under config `c`.
pub fn shape_of(name: &str, c: &Gemma4Config) -> Vec<usize> {
    let h = c.hidden_size;
    let i_ = c.intermediate_size;
    let v = c.vocab_size;
    let p = c.hidden_size_per_layer_input;
    let l = c.num_hidden_layers;
    let hq = c.num_attention_heads;
    let hkv = c.num_key_value_heads;
    match name {
        "language_model.embed_tokens.weight" => return vec![v, h],
        "language_model.embed_tokens_per_layer.weight" => return vec![v, l * p],
        "language_model.per_layer_model_projection.weight" => return vec![l * p, h],
        "language_model.per_layer_projection_norm.weight" => return vec![p],
        "language_model.norm.weight" => return vec![h],
        "lm_head.weight" => return vec![v, h],
        _ => {}
    }
    let rest = name.strip_prefix("language_model.layers.").expect("unknown weight");
    let (idx, sub) = rest.split_once('.').unwrap();
    let dh = c.head_dim(idx.parse().unwrap());
    match sub {
        "input_layernorm.weight" | "post_attention_layernorm.weight" | "pre_feedforward_layernorm.weight" | "post_feedforward_layernorm.weight" | "post_per_layer_input_norm.weight" => vec![h],
        "self_attn.q_proj.weight" => vec![hq * dh, h],
        "self_attn.k_proj.weight" | "self_attn.v_proj.weight" => vec![hkv * dh, h],
        "self_attn.o_proj.weight" => vec![h, hq * dh],
        "self_attn.q_norm.weight" | "self_attn.k_norm.weight" => vec![dh],
        "mlp.gate_proj.weight" | "mlp.up_proj.weight" => vec![i_, h],
        "mlp.down_proj.weight" => vec![h, i_],
        "per_layer_input_gate.weight" => vec![p, h],
        "per_layer_projection.weight" => vec![h, p],
        "layer_scalar" => vec![1],
        _ => panic!("unknown weight {name}"),
    }
}

/// Synthetic CPU weights, deterministic per seed, generated lazily by name.
pub struct SyntheticWeights {
    config: Arc<Gemma4Config>,
    rng: RefCell<Mulberry>,
    cache: RefCell<HashMap<String, CpuTensor>>,
}

impl SyntheticWeights {
    pub fn new(config: Arc<Gemma4Config>, seed: u32) -> Self {
        Self { config, rng: RefCell::new(Mulberry::new(seed)), cache: RefCell::new(HashMap::new()) }
    }
}

impl WeightSource<CpuBackend> for SyntheticWeights {
    fn get(&self, name: &str) -> Option<CpuTensor> {
        if let Some(t) = self.cache.borrow().get(name) {
            return Some(t.clone());
        }
        let shape = shape_of(name, &self.config);
        let n: usize = shape.iter().product();
        let is_norm = name.ends_with("norm.weight") || name.ends_with("layernorm.weight");
        let is_scalar = name.ends_with("layer_scalar");
        // norms and layer_scalar store the FULL multiplier: keep near 1.
        let (base, scale) = if is_norm || is_scalar { (1.0, 0.05) } else { (0.0, 0.04) };
        let mut rng = self.rng.borrow_mut();
        let data: Vec<f32> = (0..n).map(|_| (base + (rng.next() * 2.0 - 1.0) * scale) as f32).collect();
        let t = CpuBackend::new().tensor(&data, &shape);
        self.cache.borrow_mut().insert(name.to_string(), t.clone());
        Some(t)
    }
}

/// Mirror a CPU weight source onto another backend, caching uploads.
pub struct Mirrored<B: Backend> {
    src: SyntheticWeights,
    backend: Arc<B>,
    cache: RefCell<HashMap<String, B::Tensor>>,
}

impl<B: Backend> Mirrored<B> {
    pub fn new(src: SyntheticWeights, backend: Arc<B>) -> Self {
        Self { src, backend, cache: RefCell::new(HashMap::new()) }
    }
}

impl<B: Backend> WeightSource<B> for Mirrored<B> {
    fn get(&self, name: &str) -> Option<B::Tensor> {
        if let Some(t) = self.cache.borrow().get(name) {
            return Some(t.clone());
        }
        let c = self.src.get(name)?;
        let t = self.backend.tensor(c.data(), &c.shape);
        self.cache.borrow_mut().insert(name.to_string(), t.clone());
        Some(t)
    }
}

pub fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len(), "length mismatch");
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
}

use gemma_wgpu::dequant::pack_unsigned;
use gemma_wgpu::safetensors::{f32_to_bf16, RawTensor};

/// The tiny test config plus the real checkpoint's mixed-bit quantization rules.
pub fn real_format_config_json(vocab: usize) -> serde_json::Value {
    let mut raw = tiny_config_json(vocab);
    raw["quantization_config"] = serde_json::json!({ "module_quant_configs": {
        "^lm_head$": { "num_bits": 2 },
        "language_model\\.embed_tokens$": { "num_bits": 2 },
        "language_model\\.embed_tokens_per_layer$": { "num_bits": 4 },
        "language_model\\.layers\\.(\\d|1[0-4])\\.mlp\\.": { "num_bits": 4 },
        "language_model\\.layers\\.\\d+\\.mlp\\.": { "num_bits": 2 },
        "language_model\\.layers\\.\\d+\\.per_layer_input_gate$": { "num_bits": 8 },
        "language_model\\.layers\\.\\d+\\.per_layer_projection$": { "num_bits": 8 },
        "language_model\\.layers\\.\\d+\\.self_attn\\.": { "num_bits": 4 },
    }});
    raw
}

/// A tiny checkpoint in the REAL on-disk format: U8/I8 packed weights + f32
/// scales, BF16 norms, `model.language_model.*` names, `.embedding_quantized`.
pub fn real_format_tensors(cfg: &Gemma4Config, seed: u32) -> Vec<(String, RawTensor)> {
    let mut r = Lcg::new(seed);
    let (h, i_, v, dple, l, hq, hkv) = (cfg.hidden_size, cfg.intermediate_size, cfg.vocab_size, cfg.hidden_size_per_layer_input, cfg.num_hidden_layers, cfg.num_attention_heads, cfg.num_key_value_heads);

    let mut tensors: Vec<(String, RawTensor)> = Vec::new();
    fn add_q(t: &mut Vec<(String, RawTensor)>, r: &mut Lcg, real: &str, n: usize, k: usize, bits: u32, signed: bool, scale_cols: usize) {
        let bytes = if signed {
            (0..n * k).map(|_| ((r.next() * 256.0).floor() as i32 - 128) as i8 as u8).collect()
        } else {
            pack_unsigned(&(0..n * k).map(|_| (r.next() * (1u32 << bits) as f32).floor() as u32 % (1 << bits)).collect::<Vec<_>>(), bits)
        };
        let cols = if signed { k } else { k / (8 / bits as usize) };
        t.push((real.to_string(), RawTensor { dtype: (if signed { "I8" } else { "U8" }).into(), shape: vec![n, cols], bytes }));
        let sc: Vec<u8> = (0..n * scale_cols).flat_map(|_| (0.02 + r.next() * 0.05).to_le_bytes()).collect();
        t.push((format!("{real}_scale"), RawTensor { dtype: "F32".into(), shape: vec![n, scale_cols], bytes: sc }));
    }
    fn add_bf(t: &mut Vec<(String, RawTensor)>, r: &mut Lcg, real: &str, shape: &[usize]) {
        let n: usize = shape.iter().product();
        let bytes: Vec<u8> = (0..n).flat_map(|_| f32_to_bf16((r.next() * 2.0 - 1.0) * 0.05).to_le_bytes()).collect();
        t.push((real.to_string(), RawTensor { dtype: "BF16".into(), shape: shape.to_vec(), bytes }));
    }
    // embeddings use `.embedding_quantized` / `.embedding_scale` on disk
    fn add_embed(t: &mut Vec<(String, RawTensor)>, r: &mut Lcg, base: &str, n: usize, k: usize, bits: u32, scale_cols: usize) {
        let cols = k / (8 / bits as usize);
        let vals: Vec<u32> = (0..n * k).map(|_| (r.next() * (1u32 << bits) as f32).floor() as u32 % (1 << bits)).collect();
        t.push((format!("{base}.embedding_quantized"), RawTensor { dtype: "U8".into(), shape: vec![n, cols], bytes: pack_unsigned(&vals, bits) }));
        t.push((format!("{base}.embedding_scale"), RawTensor { dtype: "F32".into(), shape: vec![n, scale_cols], bytes: (0..n * scale_cols).flat_map(|_| (0.02 + r.next() * 0.05).to_le_bytes()).collect() }));
    }
    add_embed(&mut tensors, &mut r, "model.language_model.embed_tokens", v, h, 2, 1);
    add_embed(&mut tensors, &mut r, "model.language_model.embed_tokens_per_layer", v, l * dple, 4, l);
    add_bf(&mut tensors, &mut r, "model.language_model.per_layer_model_projection.weight", &[l * dple, h]);
    add_bf(&mut tensors, &mut r, "model.language_model.per_layer_projection_norm.weight", &[dple]);
    add_bf(&mut tensors, &mut r, "model.language_model.norm.weight", &[h]);
    add_q(&mut tensors, &mut r, "lm_head.weight", v, h, 2, false, 1);
    for i in 0..l {
        let p = format!("model.language_model.layers.{i}");
        let dh = cfg.head_dim(i);
        for nm in ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm", "post_per_layer_input_norm"] {
            add_bf(&mut tensors, &mut r, &format!("{p}.{nm}.weight"), &[h]);
        }
        add_bf(&mut tensors, &mut r, &format!("{p}.layer_scalar"), &[1]);
        add_bf(&mut tensors, &mut r, &format!("{p}.self_attn.q_norm.weight"), &[dh]);
        add_bf(&mut tensors, &mut r, &format!("{p}.self_attn.k_norm.weight"), &[dh]);
        add_q(&mut tensors, &mut r, &format!("{p}.self_attn.q_proj.weight"), hq * dh, h, 4, false, 1);
        add_q(&mut tensors, &mut r, &format!("{p}.self_attn.k_proj.weight"), hkv * dh, h, 4, false, 1);
        add_q(&mut tensors, &mut r, &format!("{p}.self_attn.v_proj.weight"), hkv * dh, h, 4, false, 1);
        add_q(&mut tensors, &mut r, &format!("{p}.self_attn.o_proj.weight"), h, hq * dh, 4, false, 1);
        add_q(&mut tensors, &mut r, &format!("{p}.mlp.gate_proj.weight"), i_, h, 4, false, 1);
        add_q(&mut tensors, &mut r, &format!("{p}.mlp.up_proj.weight"), i_, h, 4, false, 1);
        add_q(&mut tensors, &mut r, &format!("{p}.mlp.down_proj.weight"), h, i_, 4, false, 1);
        add_q(&mut tensors, &mut r, &format!("{p}.per_layer_input_gate.weight"), dple, h, 8, true, 1);
        add_q(&mut tensors, &mut r, &format!("{p}.per_layer_projection.weight"), h, dple, 8, true, 1);
    }
    // a vision tensor that the text model must ignore
    tensors.push(("model.vision_tower.patch.weight".into(), RawTensor { dtype: "F32".into(), shape: vec![4], bytes: vec![0u8; 16] }));

    tensors
}
