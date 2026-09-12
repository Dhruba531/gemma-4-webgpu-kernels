//! Gemma-4 E2B inference on wgpu, every kernel hand-written WGSL.
//!
//! A Rust port of the browser engine in the parent directory. The model graph
//! ([`model::Gemma4Model`]) is backend-agnostic: it runs on the plain-Rust
//! reference ([`backend_cpu::CpuBackend`]) or on wgpu
//! ([`backend_wgpu::WgpuBackend`]) with identical results, which is what the
//! test-suite asserts.

pub mod backend;
pub mod backend_cpu;
pub mod backend_wgpu;
pub mod config;
pub mod dequant;
pub mod device;
pub mod engine;
pub mod hub;
pub mod kernels;
pub mod kvcache;
pub mod model;
pub mod safetensors;
pub mod sampler;
pub mod template;
pub mod tokenizer;
pub mod weights;

pub use backend::{AttnOpts, Backend, QuantWeight, TensorShape};
pub use backend_cpu::CpuBackend;
pub use backend_wgpu::WgpuBackend;
pub use config::Gemma4Config;
pub use engine::{GenerateOpts, Gemma4Mobile, LoadStage, Step};
pub use model::Gemma4Model;
pub use template::Message;
pub use tokenizer::Tokenizer;

/// The real Gemma-4 E2B `config.json`, embedded for benchmarks that run at
/// production shapes without the checkpoint.
pub const GEMMA4_E2B_CONFIG_JSON: &str = include_str!("../assets/config.json");
