//! Weight provider for the real Gemma-4 QAT checkpoint.
//!
//! The model asks for logical names like
//! `language_model.layers.0.self_attn.q_proj.weight`; this resolves them to the
//! on-disk tensors and returns either an f32 tensor (norms / projections stored
//! as BF16/F32) or a *quantized* tensor kept packed on the GPU (U8 2/4-bit or I8
//! 8-bit) plus its f32 scale — dequantized on the fly inside the kernels.
//!
//! On-disk layout (from the checkpoint header):
//!   model.language_model.<...>.weight            U8/I8  + .weight_scale
//!   model.language_model.embed_tokens.embedding_quantized + .embedding_scale
//!   lm_head.weight (U8) + lm_head.weight_scale
//!   norms / per_layer_model_projection           BF16
//! Bit widths come from config.quantization_config (`Gemma4Config::bits_for`).

use std::cell::RefCell;
use std::collections::HashMap;
use std::sync::Arc;

use crate::backend::{Backend, QuantWeight};
use crate::config::Gemma4Config;
use crate::safetensors::SafetensorsFile;

/// Source of model weights by logical name. Implementations cache.
pub trait WeightSource<B: Backend> {
    fn get(&self, name: &str) -> Option<B::Tensor>;
}

/// On-disk names for a logical weight name.
pub struct ResolvedNames {
    pub weight: String,
    pub scale: String,
    /// Tensors stored without a `.weight` suffix (e.g. `layer_scalar`).
    pub bare: Option<String>,
}

pub fn resolve_tensor_names(logical: &str) -> ResolvedNames {
    let base = logical.strip_suffix(".weight").unwrap_or(logical);
    // lm_head has no "model." prefix on disk
    if base == "lm_head" {
        return ResolvedNames { weight: "lm_head.weight".into(), scale: "lm_head.weight_scale".into(), bare: None };
    }
    let real = format!("model.{base}");
    if base.ends_with("embed_tokens") || base.ends_with("embed_tokens_per_layer") {
        return ResolvedNames { weight: format!("{real}.embedding_quantized"), scale: format!("{real}.embedding_scale"), bare: None };
    }
    ResolvedNames { weight: format!("{real}.weight"), scale: format!("{real}.weight_scale"), bare: Some(real) }
}

/// Which on-disk tensors the text model needs (skips vision_tower/audio_tower/…).
pub fn is_text_tensor(name: &str) -> bool {
    name.starts_with("model.language_model.") || name.starts_with("lm_head")
}

/// Weights read from one or more safetensors files, uploaded lazily on first use.
pub struct ModelWeights<B: Backend> {
    files: Vec<SafetensorsFile>,
    index: HashMap<String, usize>,
    config: Arc<Gemma4Config>,
    backend: Arc<B>,
    cache: RefCell<HashMap<String, Option<B::Tensor>>>,
}

impl<B: Backend> ModelWeights<B> {
    pub fn new(files: Vec<SafetensorsFile>, config: Arc<Gemma4Config>, backend: Arc<B>) -> Self {
        let mut index = HashMap::new();
        for (fi, f) in files.iter().enumerate() {
            for n in f.names() {
                index.insert(n.to_string(), fi);
            }
        }
        Self { files, index, config, backend, cache: RefCell::new(HashMap::new()) }
    }

    fn file(&self, name: &str) -> Option<&SafetensorsFile> {
        self.index.get(name).map(|&i| &self.files[i])
    }

    /// Names of the text-model tensors, for eager preloading.
    pub fn text_tensor_names(&self) -> Vec<String> {
        let mut v: Vec<String> = self.index.keys().filter(|n| is_text_tensor(n)).cloned().collect();
        v.sort();
        v
    }

    fn load(&self, logical: &str) -> Option<B::Tensor> {
        let ResolvedNames { mut weight, scale, bare } = resolve_tensor_names(logical);
        let mut file = self.file(&weight);
        if file.is_none() {
            if let Some(b) = bare {
                file = self.file(&b);
                if file.is_some() {
                    weight = b;
                }
            }
        }
        let file = file?;
        let meta = file.meta(&weight).ok()?;
        let base = logical.strip_suffix(".weight").unwrap_or(logical);

        if meta.dtype == "U8" || meta.dtype == "I8" {
            let signed_i8 = meta.dtype == "I8";
            let bits = if signed_i8 { 8 } else { self.config.bits_for(base) };
            let per_byte = if signed_i8 { 1 } else { (8 / bits) as usize };
            let n = meta.shape[0];
            let stored_cols = meta.shape[1];
            let k = stored_cols * per_byte; // logical in-features
            let scale_meta = file.meta(&scale).unwrap_or_else(|_| panic!("quantized tensor {weight} is missing its scale tensor {scale}"));
            let scale_cols = scale_meta.shape.get(1).copied().unwrap_or(1);
            let group_size = k / scale_cols;
            let scales = file.tensor(&scale).expect("decode scale").data;
            let qbytes = file.raw_bytes(&weight).expect("raw bytes").to_vec();
            return Some(self.backend.quant_tensor(QuantWeight { qbytes, scales, shape: [n, k], bits, signed_i8, group_size }));
        }
        // BF16 / F16 / F32 -> f32
        let t = file.tensor(&weight).expect("decode float tensor");
        Some(self.backend.tensor(&t.data, &t.shape))
    }
}

impl<B: Backend> WeightSource<B> for ModelWeights<B> {
    fn get(&self, name: &str) -> Option<B::Tensor> {
        if let Some(t) = self.cache.borrow().get(name) {
            return t.clone();
        }
        let t = self.load(name);
        self.cache.borrow_mut().insert(name.to_string(), t.clone());
        t
    }
}

/// Every logical weight name the text model may request, in load order.
pub fn logical_weight_names(config: &Gemma4Config) -> Vec<String> {
    let mut v = vec![
        "language_model.embed_tokens.weight".to_string(),
        "language_model.embed_tokens_per_layer.weight".to_string(),
        "language_model.per_layer_model_projection.weight".to_string(),
        "language_model.per_layer_projection_norm.weight".to_string(),
    ];
    for i in 0..config.num_hidden_layers {
        let p = format!("language_model.layers.{i}");
        for s in [
            "input_layernorm.weight",
            "self_attn.q_proj.weight",
            "self_attn.q_norm.weight",
            "self_attn.k_proj.weight",
            "self_attn.k_norm.weight",
            "self_attn.v_proj.weight",
            "self_attn.o_proj.weight",
            "post_attention_layernorm.weight",
            "pre_feedforward_layernorm.weight",
            "mlp.gate_proj.weight",
            "mlp.up_proj.weight",
            "mlp.down_proj.weight",
            "post_feedforward_layernorm.weight",
            "per_layer_input_gate.weight",
            "per_layer_projection.weight",
            "post_per_layer_input_norm.weight",
            "layer_scalar",
        ] {
            v.push(format!("{p}.{s}"));
        }
    }
    v.push("language_model.norm.weight".into());
    v.push("lm_head.weight".into());
    v
}
