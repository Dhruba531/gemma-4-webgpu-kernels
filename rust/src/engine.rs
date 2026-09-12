//! Gemma4Mobile — top-level engine: load the model from a local directory and
//! stream chat.
//!
//! ```ignore
//! let mut engine = Gemma4Mobile::load(dir, Arc::new(WgpuBackend::create()?), &mut |p| {})?;
//! for step in engine.generate(&messages, GenerateOpts::default()) {
//!     print!("{}", step.delta);
//! }
//! ```

use std::path::Path;
use std::sync::Arc;

use serde_json::Value;

use crate::backend::Backend;
use crate::config::Gemma4Config;
use crate::model::Gemma4Model;
use crate::safetensors::SafetensorsFile;
use crate::sampler::{sample, Rng, SampleOpts};
use crate::template::{apply_chat_template, fold_system_prompt, Message};
use crate::tokenizer::{DecoderState, Tokenizer};
use crate::weights::{logical_weight_names, ModelWeights, WeightSource};

/// Load-progress events.
#[derive(Clone, Debug)]
pub enum LoadStage {
    Config,
    Tokenizer,
    /// Uploading weights: (tensors done, tensors total, bytes on disk).
    Weights { done: usize, total: usize },
    Ready,
}

#[derive(Clone, Debug)]
pub struct GenerateOpts {
    pub max_new_tokens: usize,
    pub temperature: f32,
    pub top_k: usize,
    pub top_p: f32,
    pub repetition_penalty: f32,
    pub seed: Option<u64>,
}

impl Default for GenerateOpts {
    fn default() -> Self {
        Self { max_new_tokens: 512, temperature: 1.0, top_k: 64, top_p: 0.95, repetition_penalty: 1.0, seed: None }
    }
}

/// One generated token.
#[derive(Clone, Debug)]
pub struct Step {
    pub token_id: u32,
    /// The raw piece text for this token (specials included).
    pub token: String,
    /// Text appended by this token (specials skipped).
    pub delta: String,
    /// Accumulated text so far (specials skipped).
    pub text: String,
}

pub struct Gemma4Mobile<B: Backend> {
    pub config: Arc<Gemma4Config>,
    pub tokenizer: Arc<Tokenizer>,
    pub model: Gemma4Model<B, ModelWeights<B>>,
    pub backend: Arc<B>,
}

fn read_json(path: &Path) -> Result<Value, String> {
    let s = std::fs::read_to_string(path).map_err(|e| format!("read {}: {e}", path.display()))?;
    serde_json::from_str(&s).map_err(|e| format!("parse {}: {e}", path.display()))
}

fn read_optional_json(path: &Path) -> Option<Value> {
    path.exists().then(|| read_json(path).ok()).flatten()
}

impl<B: Backend> Gemma4Mobile<B> {
    /// Load config, tokenizer and every `*.safetensors` in `dir`, uploading all
    /// text-model weights to the backend up front.
    pub fn load(dir: &Path, backend: Arc<B>, on_progress: &mut dyn FnMut(LoadStage)) -> Result<Self, String> {
        on_progress(LoadStage::Config);
        let raw_config = read_json(&dir.join("config.json"))?;
        let mut config = Gemma4Config::from_json(&raw_config);
        let gen_cfg = read_optional_json(&dir.join("generation_config.json"));
        // EOS: union of config.json and generation_config.json ids. The generation
        // config carries extra stop ids the model config omits (this checkpoint
        // stops on [1, 106, 50] — id 50 is the one actually emitted at turn end).
        if let Some(g) = gen_cfg.as_ref().and_then(|g| g.get("eos_token_id")) {
            let extra: Vec<u32> = match g {
                Value::Array(a) => a.iter().filter_map(Value::as_u64).map(|v| v as u32).collect(),
                v => v.as_u64().map(|v| vec![v as u32]).unwrap_or_default(),
            };
            for e in extra {
                if !config.eos_token_ids.contains(&e) {
                    config.eos_token_ids.push(e);
                }
            }
        }
        let config = Arc::new(config);

        on_progress(LoadStage::Tokenizer);
        let tok_json = read_json(&dir.join("tokenizer.json"))?;
        let tok_cfg = read_optional_json(&dir.join("tokenizer_config.json"));
        let bos = Some(config.bos_token_id).or_else(|| tok_cfg.as_ref().and_then(|c| c.get("bos_token_id")).and_then(Value::as_u64).map(|v| v as u32));
        let tokenizer = Arc::new(Tokenizer::from_json(&tok_json, bos, Some(&config.eos_token_ids))?);

        let mut files = Vec::new();
        let mut entries: Vec<_> = std::fs::read_dir(dir)
            .map_err(|e| format!("read dir {}: {e}", dir.display()))?
            .filter_map(|e| e.ok().map(|e| e.path()))
            .filter(|p| p.extension().is_some_and(|x| x == "safetensors"))
            .collect();
        entries.sort();
        if entries.is_empty() {
            return Err(format!("no *.safetensors in {}", dir.display()));
        }
        for p in entries {
            files.push(SafetensorsFile::open(&p).map_err(|e| format!("{}: {e}", p.display()))?);
        }
        let weights = ModelWeights::new(files, config.clone(), backend.clone());

        // Eagerly upload every text tensor so the first token isn't a 2.4 GB stall.
        let names = logical_weight_names(&config);
        let total = names.len();
        for (i, n) in names.iter().enumerate() {
            let _ = weights.get(n);
            on_progress(LoadStage::Weights { done: i + 1, total });
        }
        backend.flush();

        let model = Gemma4Model::new(config.clone(), backend.clone(), weights);
        on_progress(LoadStage::Ready);
        Ok(Self { config, tokenizer, model, backend })
    }

    /// Tokenize a chat into input ids, applying the Gemma template.
    pub fn encode_chat(&self, messages: &[Message]) -> Vec<u32> {
        let folded = fold_system_prompt(messages);
        let text = apply_chat_template(&folded, true);
        self.tokenizer.encode(&text, true)
    }

    /// Stream generation for a chat. The returned iterator yields one [`Step`]
    /// per generated token and stops on EOS or `max_new_tokens`.
    pub fn generate(&mut self, messages: &[Message], opts: GenerateOpts) -> Generation<'_, B> {
        self.model.reset();
        let ids = self.encode_chat(messages);
        self.generate_ids(ids, opts)
    }

    /// Stream generation from raw prompt token ids (already includes BOS etc).
    pub fn generate_ids(&mut self, ids: Vec<u32>, opts: GenerateOpts) -> Generation<'_, B> {
        let positions: Vec<u32> = (0..ids.len() as u32).collect();
        let t0 = std::time::Instant::now();
        let logits = self.model.forward(&ids, &positions);
        let prefill_secs = t0.elapsed().as_secs_f64();
        let rng = opts.seed.map(Rng::new).unwrap_or_else(Rng::from_entropy);
        Generation {
            model: &mut self.model,
            tokenizer: self.tokenizer.clone(),
            eos: self.config.eos_token_ids.clone(),
            opts,
            rng,
            logits: Some(logits),
            pos: ids.len() as u32,
            step: 0,
            text_dec: DecoderState::new(true),
            token_dec: DecoderState::new(false),
            recent: Vec::new(),
            prompt_tokens: ids.len(),
            prefill_secs,
            decode_secs: 0.0,
        }
    }
}

/// Iterator over generated tokens (see [`Gemma4Mobile::generate`]).
pub struct Generation<'a, B: Backend> {
    model: &'a mut Gemma4Model<B, ModelWeights<B>>,
    tokenizer: Arc<Tokenizer>,
    eos: Vec<u32>,
    opts: GenerateOpts,
    rng: Rng,
    logits: Option<Vec<f32>>,
    pos: u32,
    step: usize,
    text_dec: DecoderState,
    token_dec: DecoderState,
    recent: Vec<u32>,
    pub prompt_tokens: usize,
    /// Wall time of the prompt prefill.
    pub prefill_secs: f64,
    /// Accumulated wall time of decode forwards.
    pub decode_secs: f64,
}

impl<B: Backend> Iterator for Generation<'_, B> {
    type Item = Step;

    fn next(&mut self) -> Option<Step> {
        if self.step >= self.opts.max_new_tokens {
            return None;
        }
        let logits = self.logits.take()?;
        let use_penalty = self.opts.repetition_penalty != 1.0;
        let next = {
            let sopts = SampleOpts {
                temperature: self.opts.temperature,
                top_k: self.opts.top_k,
                top_p: self.opts.top_p,
                repetition_penalty: self.opts.repetition_penalty,
                recent_tokens: use_penalty.then_some(self.recent.as_slice()),
            };
            let rng = &mut self.rng;
            sample(&logits, &sopts, &mut || rng.next_f64())
        };
        if self.eos.contains(&next) {
            return None;
        }
        if use_penalty {
            self.recent.push(next);
            if self.recent.len() > 64 {
                self.recent.remove(0);
            }
        }
        let token = self.token_dec.push(&self.tokenizer, next);
        let delta = self.text_dec.push(&self.tokenizer, next);
        let step = Step { token_id: next, token, delta, text: self.text_dec.text.clone() };

        let t0 = std::time::Instant::now();
        self.logits = Some(self.model.forward(&[next], &[self.pos]));
        self.decode_secs += t0.elapsed().as_secs_f64();
        self.pos += 1;
        self.step += 1;
        Some(step)
    }
}
