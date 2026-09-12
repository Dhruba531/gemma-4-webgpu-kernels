//! End-to-end engine test: write a tiny real-format checkpoint + tokenizer to
//! a temp directory, load it through `Gemma4Mobile::load`, and stream a reply.
mod common;

use std::sync::Arc;

use common::*;
use gemma_wgpu::backend_cpu::CpuBackend;
use gemma_wgpu::config::Gemma4Config;
use gemma_wgpu::engine::{GenerateOpts, Gemma4Mobile, LoadStage};
use gemma_wgpu::safetensors::build_safetensors;
use gemma_wgpu::template::Message;

fn write_model_dir() -> std::path::PathBuf {
    let dir = std::env::temp_dir().join(format!("gemma-wgpu-test-{}-{}", std::process::id(), std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).unwrap().as_nanos()));
    std::fs::create_dir_all(&dir).unwrap();
    let raw = real_format_config_json(40);
    std::fs::write(dir.join("config.json"), serde_json::to_string(&raw).unwrap()).unwrap();
    std::fs::write(dir.join("generation_config.json"), r#"{"eos_token_id":[1,3,39]}"#).unwrap();
    let cfg = Gemma4Config::from_json(&raw);
    std::fs::write(dir.join("model.safetensors"), build_safetensors(&real_format_tensors(&cfg, 11))).unwrap();
    // a 40-entry Unigram vocab: 4 specials, byte-fallback bytes for 'a'..'z' style pieces
    let mut vocab: Vec<serde_json::Value> = vec![];
    for (p, s) in [("<pad>", 0.0), ("<eos>", 0.0), ("<bos>", 0.0), ("<turn|>", 0.0), ("<|turn>", 0.0), ("<unk>", 0.0), ("▁", -3.0), ("▁hi", -1.0), ("user", -2.0), ("model", -2.0), ("\n", -2.0)] {
        vocab.push(serde_json::json!([p, s]));
    }
    let mut i = 0;
    while vocab.len() < 40 {
        vocab.push(serde_json::json!([format!("<0x{:02X}>", 0x61 + i), -6.0]));
        i += 1;
    }
    let tok = serde_json::json!({
        "added_tokens": [ {"id":0,"content":"<pad>"}, {"id":1,"content":"<eos>"}, {"id":2,"content":"<bos>"}, {"id":3,"content":"<turn|>"}, {"id":4,"content":"<|turn>"} ],
        "model": { "type": "Unigram", "unk_id": 5, "byte_fallback": true, "vocab": vocab }
    });
    std::fs::write(dir.join("tokenizer.json"), serde_json::to_string(&tok).unwrap()).unwrap();
    dir
}

#[test]
fn load_and_generate_cpu() {
    let dir = write_model_dir();
    let mut stages = Vec::new();
    let mut engine = Gemma4Mobile::load(&dir, Arc::new(CpuBackend::new()), &mut |s| stages.push(format!("{s:?}"))).expect("load");
    assert!(stages.last().unwrap().contains("Ready"));
    assert!(stages.iter().any(|s| s.contains("Weights")));
    // eos union of config.json ([1,106] default) and generation_config.json ([1,3,39])
    assert!(engine.config.eos_token_ids.contains(&3) && engine.config.eos_token_ids.contains(&39));

    let ids = engine.encode_chat(&[Message::user("hi")]);
    assert_eq!(ids[0], 2, "bos first");
    assert!(ids.contains(&4) && ids.contains(&3), "turn delimiters are single tokens");

    let opts = GenerateOpts { max_new_tokens: 6, temperature: 0.0, seed: Some(1), ..Default::default() };
    let steps: Vec<_> = engine.generate(&[Message::user("hi")], opts.clone()).collect();
    assert!(!steps.is_empty() && steps.len() <= 6);
    for s in &steps {
        assert!(!engine.config.eos_token_ids.contains(&s.token_id), "eos never yielded");
        assert!((s.token_id as usize) < engine.config.vocab_size);
    }
    let last = steps.last().unwrap();
    assert_eq!(last.text, steps.iter().map(|s| s.delta.as_str()).collect::<String>(), "text accumulates deltas");

    // greedy decoding is deterministic across calls (cache reset between runs)
    let again: Vec<u32> = engine.generate(&[Message::user("hi")], opts).map(|s| s.token_id).collect();
    assert_eq!(again, steps.iter().map(|s| s.token_id).collect::<Vec<_>>());

    let _ = std::fs::remove_dir_all(&dir);
    let _ = LoadStage::Ready;
}
