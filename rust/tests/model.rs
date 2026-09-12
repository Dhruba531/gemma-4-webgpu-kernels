//! End-to-end CPU-backend tests for the Gemma-4 forward graph.
mod common;

use std::sync::Arc;

use common::*;
use gemma_wgpu::backend_cpu::CpuBackend;
use gemma_wgpu::model::Gemma4Model;
use gemma_wgpu::sampler::{sample, SampleOpts};

const IDS: [u32; 5] = [2, 10, 20, 30, 40];

#[test]
fn config_sanity() {
    let c = tiny_config(64);
    assert_eq!(c.num_hidden_layers, 6);
    assert!(c.is_global(1) && !c.is_global(0), "layer 1 is global, layer 0 local");
    assert_eq!((c.head_dim(1), c.head_dim(0)), (16, 8), "per-layer head_dim");
    for i in 0..c.num_hidden_layers {
        let s = c.kv_source_layer(i);
        assert!(s <= i, "layer {i} kv source {s} is valid (<= i)");
    }
    // shared layers (4, 5) reuse the most recent same-type non-shared layer
    assert_eq!(c.kv_source_layer(4), 3);
    assert_eq!(c.kv_source_layer(5), 3);
}

#[test]
fn forward_pass_is_finite() {
    let c = tiny_config(64);
    let b = Arc::new(CpuBackend::new());
    let mut m = Gemma4Model::new(c.clone(), b, SyntheticWeights::new(c.clone(), 42));
    let logits = m.forward(&IDS, &[0, 1, 2, 3, 4]);
    assert_eq!(logits.len(), c.vocab_size, "logits length == vocab_size");
    assert!(logits.iter().all(|x| x.is_finite()), "no NaNs/Infs in logits");
}

#[test]
fn incremental_decode_matches_full_sequence() {
    let c = tiny_config(64);
    let b = Arc::new(CpuBackend::new());
    let mut full = Gemma4Model::new(c.clone(), b.clone(), SyntheticWeights::new(c.clone(), 42));
    let full_logits = full.forward(&IDS, &[0, 1, 2, 3, 4]);
    let mut inc = Gemma4Model::new(c.clone(), b, SyntheticWeights::new(c.clone(), 42));
    let mut inc_logits = Vec::new();
    for (i, &id) in IDS.iter().enumerate() {
        inc_logits = inc.forward(&[id], &[i as u32]);
    }
    let d = max_abs_diff(&full_logits, &inc_logits);
    assert!(d < 1e-3, "last-token logits match (max abs diff {d:e})");
}

#[test]
fn long_sliding_window_sequence_stays_finite() {
    let c = tiny_config(64);
    let b = Arc::new(CpuBackend::new());
    let mut m = Gemma4Model::new(c.clone(), b, SyntheticWeights::new(c.clone(), 42));
    let ids: Vec<u32> = (0..20).map(|i| (i * 7) % 64).collect();
    let pos: Vec<u32> = (0..20).collect();
    let logits = m.forward(&ids, &pos);
    assert!(logits.iter().all(|x| x.is_finite()), "20-token forward stays finite");
}

#[test]
fn reset_clears_cache_and_reproduces() {
    let c = tiny_config(64);
    let b = Arc::new(CpuBackend::new());
    let mut m = Gemma4Model::new(c.clone(), b, SyntheticWeights::new(c.clone(), 7));
    let a = m.forward(&IDS, &[0, 1, 2, 3, 4]);
    m.reset();
    assert!(m.cache.is_empty());
    let b2 = m.forward(&IDS, &[0, 1, 2, 3, 4]);
    assert_eq!(a, b2);
}

#[test]
fn sampler() {
    let mut rng = || 0.99f64;
    let greedy = sample(&[0.1, 5.0, 0.2, -1.0], &SampleOpts { temperature: 0.0, ..Default::default() }, &mut rng);
    assert_eq!(greedy, 1, "greedy picks argmax");
    let topk = sample(&[0.0, 10.0, 9.0, -5.0], &SampleOpts { temperature: 1.0, top_k: 1, ..Default::default() }, &mut rng);
    assert_eq!(topk, 1, "top-k=1 is deterministic argmax");
    let penalized = sample(
        &[10.0, 4.0],
        &SampleOpts { temperature: 0.0, repetition_penalty: 2.0, recent_tokens: Some(&[0, 0]), ..Default::default() },
        &mut rng,
    );
    assert_eq!(penalized, 0, "repetition penalty applies once per unique token");
    // top-p keeps only the head of the distribution
    let mut small = || 0.0f64;
    let topp = sample(&[0.0, 10.0, 9.9, -5.0], &SampleOpts { temperature: 1.0, top_p: 0.5, ..Default::default() }, &mut small);
    assert_eq!(topp, 1);
    // full-vocab temperature sampling stays in range and is seed-deterministic
    let mut r1 = gemma_wgpu::sampler::Rng::new(1);
    let mut r2 = gemma_wgpu::sampler::Rng::new(1);
    let l: Vec<f32> = (0..100).map(|i| (i as f32 * 0.37).sin()).collect();
    let a = sample(&l, &SampleOpts::default(), &mut || r1.next_f64());
    let b = sample(&l, &SampleOpts::default(), &mut || r2.next_f64());
    assert_eq!(a, b);
    assert!((a as usize) < l.len());
}
