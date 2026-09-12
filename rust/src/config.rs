//! Gemma-4 configuration parsing.
//!
//! The HF `config.json` for google/gemma-4-E2B-it-qat-mobile-transformers is a
//! multimodal config (text + vision + audio). This engine implements the *text*
//! path only, so we read `text_config` and normalise it into the flat shape the
//! rest of the runtime consumes.

use serde_json::Value;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LayerType {
    Sliding,
    Full,
}

#[derive(Clone, Copy, Debug)]
pub struct RopeParams {
    pub theta: f32,
    /// < 1 means only a prefix of head_dim is rotated.
    pub partial_rotary_factor: f32,
}

#[derive(Debug)]
pub struct QuantRule {
    pub re: fancy_regex::Regex,
    pub bits: u32,
}

#[derive(Debug)]
pub struct Gemma4Config {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    pub head_dim: usize,
    pub global_head_dim: usize,
    pub rms_norm_eps: f32,
    pub sliding_window: u32,
    pub final_logit_softcapping: f32,
    pub hidden_size_per_layer_input: usize,
    pub vocab_size_per_layer_input: usize,
    pub num_kv_shared_layers: usize,
    pub bos_token_id: u32,
    pub eos_token_ids: Vec<u32>,
    pub pad_token_id: u32,
    pub layer_types: Vec<LayerType>,
    pub rope_full: RopeParams,
    pub rope_sliding: RopeParams,
    pub quant_rules: Vec<QuantRule>,
}

fn get_usize(t: &Value, key: &str, default: usize) -> usize {
    t.get(key).and_then(Value::as_u64).map(|v| v as usize).unwrap_or(default)
}
fn get_f32(t: &Value, key: &str, default: f32) -> f32 {
    t.get(key).and_then(Value::as_f64).map(|v| v as f32).unwrap_or(default)
}

impl Gemma4Config {
    /// Accept either a full multimodal config or a bare text_config.
    pub fn from_json(raw: &Value) -> Self {
        let t = raw.get("text_config").unwrap_or(raw);
        let num_hidden_layers = get_usize(t, "num_hidden_layers", 35);

        // eos can be a scalar or list; always expose an array.
        let eos_token_ids: Vec<u32> = match t.get("eos_token_id") {
            Some(Value::Array(a)) => a.iter().filter_map(Value::as_u64).map(|v| v as u32).collect(),
            Some(v) => v.as_u64().map(|v| vec![v as u32]).unwrap_or_else(|| vec![1, 106]),
            None => vec![1, 106],
        };

        // Per-layer attention type. If the model omits layer_types, fall back to
        // "every 5th layer is full attention" which is the Gemma-3 pattern.
        let layer_types: Vec<LayerType> = match t.get("layer_types").and_then(Value::as_array) {
            Some(a) if a.len() == num_hidden_layers => a
                .iter()
                .map(|v| if v.as_str() == Some("full_attention") { LayerType::Full } else { LayerType::Sliding })
                .collect(),
            _ => (0..num_hidden_layers).map(|i| if (i + 1) % 5 == 0 { LayerType::Full } else { LayerType::Sliding }).collect(),
        };

        let rp = t.get("rope_parameters").cloned().unwrap_or(Value::Null);
        let rope = |key: &str, theta: f32| RopeParams {
            theta: rp.get(key).map(|v| get_f32(v, "rope_theta", theta)).unwrap_or(theta),
            partial_rotary_factor: rp.get(key).map(|v| get_f32(v, "partial_rotary_factor", 1.0)).unwrap_or(1.0),
        };

        // Mixed-precision QAT: map a weight name -> bit width via the regex table.
        let mut quant_rules = Vec::new();
        if let Some(qc) = raw.get("quantization_config").and_then(|q| q.get("module_quant_configs")).and_then(Value::as_object) {
            for (pattern, cfg) in qc {
                if let (Ok(re), Some(bits)) = (fancy_regex::Regex::new(pattern), cfg.get("num_bits").and_then(Value::as_u64)) {
                    quant_rules.push(QuantRule { re, bits: bits as u32 });
                }
            }
        }

        Self {
            vocab_size: get_usize(t, "vocab_size", 262144),
            hidden_size: get_usize(t, "hidden_size", 1536),
            intermediate_size: get_usize(t, "intermediate_size", 6144),
            num_hidden_layers,
            num_attention_heads: get_usize(t, "num_attention_heads", 8),
            num_key_value_heads: get_usize(t, "num_key_value_heads", 1),
            head_dim: get_usize(t, "head_dim", 256),
            global_head_dim: get_usize(t, "global_head_dim", 512),
            rms_norm_eps: get_f32(t, "rms_norm_eps", 1e-6),
            sliding_window: get_usize(t, "sliding_window", 512) as u32,
            final_logit_softcapping: get_f32(t, "final_logit_softcapping", 30.0),
            hidden_size_per_layer_input: get_usize(t, "hidden_size_per_layer_input", 256),
            vocab_size_per_layer_input: get_usize(t, "vocab_size_per_layer_input", 262144),
            num_kv_shared_layers: get_usize(t, "num_kv_shared_layers", 20),
            bos_token_id: get_usize(t, "bos_token_id", 2) as u32,
            eos_token_ids,
            pad_token_id: get_usize(t, "pad_token_id", 0) as u32,
            layer_types,
            rope_full: rope("full_attention", 1_000_000.0),
            rope_sliding: rope("sliding_attention", 10_000.0),
            quant_rules,
        }
    }

    pub fn from_json_str(s: &str) -> Result<Self, serde_json::Error> {
        Ok(Self::from_json(&serde_json::from_str(s)?))
    }

    pub fn layer_type(&self, i: usize) -> LayerType {
        self.layer_types[i]
    }

    pub fn is_global(&self, i: usize) -> bool {
        self.layer_types[i] == LayerType::Full
    }

    /// head_dim depends on whether the layer is global or local.
    pub fn head_dim(&self, i: usize) -> usize {
        if self.is_global(i) { self.global_head_dim } else { self.head_dim }
    }

    pub fn rope_for(&self, i: usize) -> RopeParams {
        if self.is_global(i) { self.rope_full } else { self.rope_sliding }
    }

    /// Which physical layer owns the KV cache that layer `i` reads from.
    ///
    /// The last `num_kv_shared_layers` layers don't compute their own KV; they
    /// reuse the KV of the most recent earlier layer of the SAME attention type
    /// that lives in the non-shared region. Matching the type guarantees the
    /// cached K/V have the same head_dim and RoPE base as the query.
    pub fn kv_source_layer(&self, i: usize) -> usize {
        let first_shared = self.num_hidden_layers.saturating_sub(self.num_kv_shared_layers);
        if i < first_shared {
            return i;
        }
        let ty = self.layer_types[i];
        (0..first_shared).rev().find(|&j| self.layer_types[j] == ty).unwrap_or(i)
    }

    /// Quantized bit width for a weight name, or 16 for unquantized (bf16/f32).
    pub fn bits_for(&self, name: &str) -> u32 {
        for rule in &self.quant_rules {
            if rule.re.is_match(name).unwrap_or(false) {
                return rule.bits;
            }
        }
        16
    }
}
