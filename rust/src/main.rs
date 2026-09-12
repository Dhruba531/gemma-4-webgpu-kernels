//! `gemma-wgpu` CLI: download the checkpoint, chat, one-shot generate,
//! benchmark at real shapes, or print adapter info.

use std::io::{BufRead, Write};
use std::path::PathBuf;
use std::sync::Arc;
use std::time::Instant;

use clap::{Parser, Subcommand, ValueEnum};

use gemma_wgpu::backend::{Backend, QuantWeight};
use gemma_wgpu::backend_cpu::CpuBackend;
use gemma_wgpu::backend_wgpu::WgpuBackend;
use gemma_wgpu::config::Gemma4Config;
use gemma_wgpu::engine::{GenerateOpts, Gemma4Mobile, LoadStage};
use gemma_wgpu::hub;
use gemma_wgpu::model::Gemma4Model;
use gemma_wgpu::template::Message;
use gemma_wgpu::weights::WeightSource;

#[derive(Parser)]
#[command(name = "gemma-wgpu", version, about = "Gemma-4 E2B on wgpu — every kernel hand-written WGSL")]
struct Cli {
    #[command(subcommand)]
    cmd: Cmd,
}

#[derive(Clone, Copy, ValueEnum, PartialEq, Eq)]
enum BackendKind {
    Wgpu,
    Cpu,
}

#[derive(clap::Args, Clone)]
struct ModelArgs {
    /// Directory holding config.json, tokenizer.json and model.safetensors
    #[arg(long, default_value = "models/gemma-4-e2b")]
    dir: PathBuf,
    #[arg(long, value_enum, default_value_t = BackendKind::Wgpu)]
    backend: BackendKind,
    /// Compile kernels without runtime bounds checks (faster; kernels are parity-tested)
    #[arg(long)]
    trusted_shaders: bool,
}

#[derive(clap::Args, Clone)]
struct SamplingArgs {
    #[arg(long, default_value_t = 512)]
    max_new_tokens: usize,
    #[arg(long, default_value_t = 1.0)]
    temperature: f32,
    #[arg(long, default_value_t = 64)]
    top_k: usize,
    #[arg(long, default_value_t = 0.95)]
    top_p: f32,
    #[arg(long, default_value_t = 1.0)]
    repetition_penalty: f32,
    /// RNG seed for reproducible sampling
    #[arg(long)]
    seed: Option<u64>,
    /// Optional system prompt
    #[arg(long)]
    system: Option<String>,
}

impl SamplingArgs {
    fn opts(&self) -> GenerateOpts {
        GenerateOpts {
            max_new_tokens: self.max_new_tokens,
            temperature: self.temperature,
            top_k: self.top_k,
            top_p: self.top_p,
            repetition_penalty: self.repetition_penalty,
            seed: self.seed,
        }
    }
}

#[derive(Subcommand)]
enum Cmd {
    /// Download the checkpoint files from the Hugging Face Hub (needs HF_TOKEN for the gated repo)
    Download {
        #[arg(long, default_value = hub::DEFAULT_REPO)]
        repo: String,
        #[arg(long, default_value = "main")]
        revision: String,
        #[arg(long, default_value = "models/gemma-4-e2b")]
        dir: PathBuf,
    },
    /// Interactive chat (REPL on stdin)
    Chat {
        #[command(flatten)]
        model: ModelArgs,
        #[command(flatten)]
        sampling: SamplingArgs,
    },
    /// Generate a single reply for a prompt and print it (streamed)
    Generate {
        #[command(flatten)]
        model: ModelArgs,
        #[command(flatten)]
        sampling: SamplingArgs,
        #[arg(long)]
        prompt: String,
    },
    /// Benchmark decode/prefill at real Gemma-4 E2B shapes with synthetic weights (no download)
    Bench {
        #[arg(long, default_value_t = 64)]
        prefill: usize,
        #[arg(long, default_value_t = 24)]
        decode: usize,
        /// Divide the vocab (262144) by this to shrink the lm_head/embedding tables
        #[arg(long, default_value_t = 1)]
        vdiv: usize,
        /// Also time isolated kernel shapes
        #[arg(long)]
        kernels: bool,
        /// Compile kernels without runtime bounds checks (faster; kernels are parity-tested)
        #[arg(long)]
        trusted_shaders: bool,
    },
    /// Print the wgpu adapter this machine would use
    Info,
}

fn main() {
    let cli = Cli::parse();
    let res = match cli.cmd {
        Cmd::Download { repo, revision, dir } => download(&repo, &revision, &dir),
        Cmd::Chat { model, sampling } => with_backend(model, |b, m| chat(b, m, &sampling)),
        Cmd::Generate { model, sampling, prompt } => with_backend(model, |b, m| generate(b, m, &sampling, &prompt)),
        Cmd::Bench { prefill, decode, vdiv, kernels, trusted_shaders } => bench(prefill, decode, vdiv, kernels, trusted_shaders),
        Cmd::Info => info(),
    };
    if let Err(e) = res {
        eprintln!("error: {e}");
        std::process::exit(1);
    }
}

fn info() -> Result<(), String> {
    let g = WgpuBackend::create()?;
    let i = &g.ctx.info;
    println!("adapter:      {}", i.name);
    println!("backend:      {}", i.backend);
    println!("device type:  {}", i.device_type);
    println!("max buffer:   {:.2} GiB", i.max_buffer_size as f64 / (1u64 << 30) as f64);
    println!("max binding:  {:.2} GiB", i.max_storage_buffer_binding_size as f64 / (1u64 << 30) as f64);
    Ok(())
}

fn download(repo: &str, revision: &str, dir: &PathBuf) -> Result<(), String> {
    for file in hub::MODEL_FILES {
        let mut last = Instant::now();
        let mut progress = |f: &str, done: u64, total: Option<u64>| {
            if last.elapsed().as_millis() < 200 && total.is_some_and(|t| done < t) {
                return;
            }
            last = Instant::now();
            match total {
                Some(t) => eprint!("\r{f}: {:.1} / {:.1} MB ({:.0}%)   ", done as f64 / 1e6, t as f64 / 1e6, 100.0 * done as f64 / t as f64),
                None => eprint!("\r{f}: {:.1} MB   ", done as f64 / 1e6),
            }
        };
        match hub::download_file(repo, file, revision, dir, &mut progress) {
            Ok(p) => eprintln!("\r{file}: saved to {}                              ", p.display()),
            Err(e) if *file == "tokenizer_config.json" || *file == "generation_config.json" => eprintln!("\r{file}: skipped ({e})"),
            Err(e) => return Err(e),
        }
    }
    Ok(())
}

/// Load the model on the requested backend and hand it to `f`.
fn with_backend(args: ModelArgs, f: impl FnOnce(BackendKind, &mut dyn ChatEngine) -> Result<(), String>) -> Result<(), String> {
    let mut last = Instant::now();
    let t0 = Instant::now();
    let mut progress = |s: LoadStage| match s {
        LoadStage::Config => eprintln!("loading config…"),
        LoadStage::Tokenizer => eprintln!("loading tokenizer…"),
        LoadStage::Weights { done, total } => {
            if done == total || last.elapsed().as_millis() > 200 {
                last = Instant::now();
                eprint!("\ruploading weights: {done}/{total}   ");
            }
        }
        LoadStage::Ready => eprintln!("\nready in {:.1}s", t0.elapsed().as_secs_f64()),
    };
    match args.backend {
        BackendKind::Wgpu => {
            let b = Arc::new(WgpuBackend::create()?.with_trusted_shaders(args.trusted_shaders));
            eprintln!("gpu: {} ({}){}", b.ctx.info.name, b.ctx.info.backend, if b.trusted_shaders() { ", trusted shaders" } else { "" });
            let mut e = Gemma4Mobile::load(&args.dir, b, &mut progress)?;
            f(BackendKind::Wgpu, &mut e)
        }
        BackendKind::Cpu => {
            eprintln!("warning: the CPU backend is a correctness reference; expect seconds per token");
            let mut e = Gemma4Mobile::load(&args.dir, Arc::new(CpuBackend::new()), &mut progress)?;
            f(BackendKind::Cpu, &mut e)
        }
    }
}

/// Object-safe view over `Gemma4Mobile<B>` so the CLI can be backend-generic.
trait ChatEngine {
    fn run(&mut self, messages: &[Message], opts: GenerateOpts, out: &mut dyn FnMut(&str)) -> RunStats;
}

struct RunStats {
    text: String,
    prompt_tokens: usize,
    new_tokens: usize,
    prefill_secs: f64,
    decode_secs: f64,
}

impl<B: Backend> ChatEngine for Gemma4Mobile<B> {
    fn run(&mut self, messages: &[Message], opts: GenerateOpts, out: &mut dyn FnMut(&str)) -> RunStats {
        let mut gen = self.generate(messages, opts);
        let mut text = String::new();
        let mut n = 0;
        for step in &mut gen {
            out(&step.delta);
            text = step.text;
            n += 1;
        }
        RunStats { text, prompt_tokens: gen.prompt_tokens, new_tokens: n, prefill_secs: gen.prefill_secs, decode_secs: gen.decode_secs }
    }
}

fn report(s: &RunStats) {
    let tps = if s.decode_secs > 0.0 { s.new_tokens as f64 / s.decode_secs } else { 0.0 };
    eprintln!(
        "\n[{} prompt tok, prefill {:.0} ms ({:.1} ms/tok) · {} new tok, {:.1} tok/s]",
        s.prompt_tokens,
        s.prefill_secs * 1e3,
        s.prefill_secs * 1e3 / s.prompt_tokens.max(1) as f64,
        s.new_tokens,
        tps
    );
}

fn generate(_b: BackendKind, engine: &mut dyn ChatEngine, sampling: &SamplingArgs, prompt: &str) -> Result<(), String> {
    let mut messages = Vec::new();
    if let Some(s) = &sampling.system {
        messages.push(Message::system(s));
    }
    messages.push(Message::user(prompt));
    let stdout = std::io::stdout();
    let stats = engine.run(&messages, sampling.opts(), &mut |d| {
        let mut o = stdout.lock();
        let _ = o.write_all(d.as_bytes());
        let _ = o.flush();
    });
    println!();
    report(&stats);
    Ok(())
}

fn chat(_b: BackendKind, engine: &mut dyn ChatEngine, sampling: &SamplingArgs) -> Result<(), String> {
    let mut history: Vec<Message> = Vec::new();
    if let Some(s) = &sampling.system {
        history.push(Message::system(s));
    }
    eprintln!("chat ready — type a message, `/reset` to clear history, `/quit` to exit");
    let stdin = std::io::stdin();
    loop {
        eprint!("\n> ");
        let mut line = String::new();
        if stdin.lock().read_line(&mut line).map_err(|e| e.to_string())? == 0 {
            break;
        }
        let line = line.trim();
        if line.is_empty() {
            continue;
        }
        if line == "/quit" || line == "/exit" {
            break;
        }
        if line == "/reset" {
            history.retain(|m| m.role == "system");
            eprintln!("(history cleared)");
            continue;
        }
        history.push(Message::user(line));
        let stdout = std::io::stdout();
        let stats = engine.run(&history, sampling.opts(), &mut |d| {
            let mut o = stdout.lock();
            let _ = o.write_all(d.as_bytes());
            let _ = o.flush();
        });
        println!();
        report(&stats);
        history.push(Message::assistant(&stats.text));
    }
    Ok(())
}

// --- benchmark: real shapes, synthetic weights ---------------------------

struct Xorshift(u32);
impl Xorshift {
    fn next(&mut self) -> u32 {
        let mut s = self.0;
        s ^= s << 13;
        s ^= s >> 17;
        s ^= s << 5;
        self.0 = s;
        s
    }
    fn f(&mut self) -> f32 {
        self.next() as f32 / u32::MAX as f32
    }
    fn bytes(&mut self, n: usize) -> Vec<u8> {
        let mut out = vec![0u8; n];
        for c in out.chunks_mut(4) {
            let w = self.next().to_le_bytes();
            c.copy_from_slice(&w[..c.len()]);
        }
        out
    }
    fn f32s(&mut self, n: usize, sc: f32, base: f32) -> Vec<f32> {
        (0..n).map(|_| base + (self.f() * 2.0 - 1.0) * sc).collect()
    }
    fn scales(&mut self, n: usize) -> Vec<f32> {
        (0..n).map(|_| 0.004 + self.f() * 0.008).collect()
    }
}

enum Spec {
    Q { shape: [usize; 2], bits: u32, signed: bool, group: Option<usize> },
    F { shape: Vec<usize>, sc: f32, base: f32 },
}

/// Synthetic weights at the real E2B shapes, generated on demand and uploaded once.
struct BenchWeights {
    table: std::collections::HashMap<String, Spec>,
    backend: Arc<WgpuBackend>,
    rng: std::cell::RefCell<Xorshift>,
    cache: std::cell::RefCell<std::collections::HashMap<String, <WgpuBackend as Backend>::Tensor>>,
    uploaded: std::cell::Cell<usize>,
}

impl WeightSource<WgpuBackend> for BenchWeights {
    fn get(&self, name: &str) -> Option<<WgpuBackend as Backend>::Tensor> {
        if let Some(t) = self.cache.borrow().get(name) {
            return Some(t.clone());
        }
        let spec = self.table.get(name)?;
        let mut rng = self.rng.borrow_mut();
        let t = match spec {
            Spec::F { shape, sc, base } => {
                let n: usize = shape.iter().product();
                self.uploaded.set(self.uploaded.get() + n * 4);
                self.backend.tensor(&rng.f32s(n, *sc, *base), shape)
            }
            Spec::Q { shape, bits, signed, group } => {
                let [n, k] = *shape;
                let per_byte = if *signed { 1 } else { (8 / bits) as usize };
                let group_size = group.unwrap_or(k);
                let qbytes = rng.bytes((n * k).div_ceil(per_byte));
                self.uploaded.set(self.uploaded.get() + qbytes.len());
                let scales = rng.scales(n * k.div_ceil(group_size));
                self.backend.quant_tensor(QuantWeight { qbytes, scales, shape: [n, k], bits: *bits, signed_i8: *signed, group_size })
            }
        };
        self.cache.borrow_mut().insert(name.to_string(), t.clone());
        Some(t)
    }
}

fn bench(prefill: usize, decode: usize, vdiv: usize, kernels: bool, trusted: bool) -> Result<(), String> {
    let g = Arc::new(WgpuBackend::create()?.with_trusted_shaders(trusted));
    println!("gpu: {} ({}){}", g.ctx.info.name, g.ctx.info.backend, if g.trusted_shaders() { ", trusted shaders" } else { "" });
    if kernels {
        kernel_bench(&g);
    }
    let mut raw: serde_json::Value = serde_json::from_str(gemma_wgpu::GEMMA4_E2B_CONFIG_JSON).unwrap();
    let v = raw["text_config"]["vocab_size"].as_u64().unwrap() as usize / vdiv.max(1);
    raw["text_config"]["vocab_size"] = v.into();
    raw["text_config"]["vocab_size_per_layer_input"] = v.into();
    let c = Arc::new(Gemma4Config::from_json(&raw));
    let (h, i_, l, dple, hq, hkv) = (c.hidden_size, c.intermediate_size, c.num_hidden_layers, c.hidden_size_per_layer_input, c.num_attention_heads, c.num_key_value_heads);

    let mut table = std::collections::HashMap::new();
    let mut norm = |name: String, dim: usize| table.insert(name, Spec::F { shape: vec![dim], sc: 0.05, base: 1.0 });
    norm("language_model.per_layer_projection_norm.weight".into(), dple);
    norm("language_model.norm.weight".into(), h);
    for i in 0..l {
        let p = format!("language_model.layers.{i}");
        let dh = c.head_dim(i);
        norm(format!("{p}.self_attn.q_norm.weight"), dh);
        norm(format!("{p}.self_attn.k_norm.weight"), dh);
        norm(format!("{p}.post_per_layer_input_norm.weight"), h);
        for n in ["input_layernorm", "post_attention_layernorm", "pre_feedforward_layernorm", "post_feedforward_layernorm"] {
            norm(format!("{p}.{n}.weight"), h);
        }
    }
    table.insert("language_model.embed_tokens.weight".into(), Spec::Q { shape: [v, h], bits: 2, signed: false, group: None });
    table.insert("language_model.embed_tokens_per_layer.weight".into(), Spec::Q { shape: [v, l * dple], bits: 4, signed: false, group: Some(dple) });
    table.insert("language_model.per_layer_model_projection.weight".into(), Spec::F { shape: vec![l * dple, h], sc: 0.02, base: 0.0 });
    table.insert("lm_head.weight".into(), Spec::Q { shape: [v, h], bits: 2, signed: false, group: None });
    for i in 0..l {
        let p = format!("language_model.layers.{i}");
        let dh = c.head_dim(i);
        let mlp_bits = c.bits_for(&format!("{p}.mlp.gate_proj"));
        // the 2-bit MLP layers (15-34) double their intermediate width
        let inter = if mlp_bits == 2 { i_ * 2 } else { i_ };
        table.insert(format!("{p}.self_attn.q_proj.weight"), Spec::Q { shape: [hq * dh, h], bits: 4, signed: false, group: None });
        table.insert(format!("{p}.self_attn.k_proj.weight"), Spec::Q { shape: [hkv * dh, h], bits: 4, signed: false, group: None });
        table.insert(format!("{p}.self_attn.v_proj.weight"), Spec::Q { shape: [hkv * dh, h], bits: 4, signed: false, group: None });
        table.insert(format!("{p}.self_attn.o_proj.weight"), Spec::Q { shape: [h, hq * dh], bits: 4, signed: false, group: None });
        table.insert(format!("{p}.mlp.gate_proj.weight"), Spec::Q { shape: [inter, h], bits: mlp_bits, signed: false, group: None });
        table.insert(format!("{p}.mlp.up_proj.weight"), Spec::Q { shape: [inter, h], bits: mlp_bits, signed: false, group: None });
        table.insert(format!("{p}.mlp.down_proj.weight"), Spec::Q { shape: [h, inter], bits: mlp_bits, signed: false, group: None });
        table.insert(format!("{p}.per_layer_input_gate.weight"), Spec::Q { shape: [dple, h], bits: 8, signed: true, group: None });
        table.insert(format!("{p}.per_layer_projection.weight"), Spec::Q { shape: [h, dple], bits: 8, signed: true, group: None });
        table.insert(format!("{p}.layer_scalar"), Spec::F { shape: vec![1], sc: 0.05, base: 1.0 });
    }
    let names: Vec<String> = table.keys().cloned().collect();
    let weights = BenchWeights { table, backend: g.clone(), rng: std::cell::RefCell::new(Xorshift(0x9e3779b9)), cache: Default::default(), uploaded: Default::default() };

    println!("config: {l} layers, hidden {h}, vocab {v}, prefill {prefill}, decode {decode}");
    let t0 = Instant::now();
    for n in &names {
        weights.get(n); // exclude upload from timings
    }
    g.wait_idle();
    println!("weights: {:.2} GiB uploaded in {:.1}s", weights.uploaded.get() as f64 / (1u64 << 30) as f64, t0.elapsed().as_secs_f64());
    let mut model = Gemma4Model::new(c.clone(), g.clone(), weights);

    let mut rng = Xorshift(12345);
    let ids: Vec<u32> = (0..prefill).map(|_| 7 + (rng.f() * (v - 8) as f32) as u32).collect();
    let pos: Vec<u32> = (0..prefill as u32).collect();
    let finite = |a: &[f32]| a.iter().all(|x| x.is_finite());
    let l2 = |a: &[f32]| a.iter().take(512).map(|x| x * x).sum::<f32>().sqrt();

    // warm up pipelines with a tiny pass, then time prefill
    model.forward(&[ids[0]], &[0]);
    model.reset();
    let t = Instant::now();
    let mut logits = model.forward(&ids, &pos);
    let prefill_ms = t.elapsed().as_secs_f64() * 1e3;
    println!("prefill: {prefill_ms:.0} ms total, {:.1} ms/token  (finite: {}, l2={:.3})", prefill_ms / prefill as f64, finite(&logits), l2(&logits));

    let mut p = prefill as u32;
    for _ in 0..2 {
        logits = model.forward(&[ids[0]], &[p]);
        p += 1;
    }
    let mut times = Vec::new();
    for s in 0..decode {
        let t = Instant::now();
        logits = model.forward(&[ids[(s * 13) % prefill]], &[p]);
        p += 1;
        times.push(t.elapsed().as_secs_f64() * 1e3);
    }
    times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let med = times[times.len() / 2];
    println!(
        "decode: median {med:.1} ms/token ({:.1} tok/s), min {:.1}, max {:.1}  (finite: {}, l2={:.3})",
        1000.0 / med,
        times[0],
        times[times.len() - 1],
        finite(&logits),
        l2(&logits)
    );
    Ok(())
}

/// Time isolated kernel shapes (decode M=1) to attribute the per-token budget.
fn kernel_bench(g: &Arc<WgpuBackend>) {
    let mut rng = Xorshift(987654321);
    let qt = |rng: &mut Xorshift, n: usize, k: usize, bits: u32, signed: bool| {
        let per_byte = if signed { 1 } else { (8 / bits) as usize };
        g.quant_tensor(QuantWeight { qbytes: rng.bytes((n * k).div_ceil(per_byte)), scales: rng.scales(n), shape: [n, k], bits, signed_i8: signed, group_size: k })
    };
    type Op = Box<dyn FnMut() -> <WgpuBackend as Backend>::Tensor>;
    let mut cases: Vec<(&str, usize, Op)> = Vec::new();
    {
        let w = qt(&mut rng, 262144, 1536, 2, false);
        let x = g.tensor(&rng.f32s(1536, 0.1, 0.0), &[1, 1536]);
        let g2 = g.clone();
        cases.push(("lm_head 2b [262144,1536]", 20, Box::new(move || g2.linear(&x, &w))));
    }
    {
        let w = qt(&mut rng, 6144, 1536, 4, false);
        let x = g.tensor(&rng.f32s(1536, 0.1, 0.0), &[1, 1536]);
        let g2 = g.clone();
        cases.push(("mlp gate 4b [6144,1536]", 60, Box::new(move || g2.linear(&x, &w))));
    }
    {
        let w = qt(&mut rng, 1536, 6144, 2, false);
        let x = g.tensor(&rng.f32s(6144, 0.1, 0.0), &[1, 6144]);
        let g2 = g.clone();
        cases.push(("mlp down 2b [1536,6144]", 60, Box::new(move || g2.linear(&x, &w))));
    }
    {
        let w = qt(&mut rng, 2048, 1536, 4, false);
        let x = g.tensor(&rng.f32s(1536, 0.1, 0.0), &[1, 1536]);
        let g2 = g.clone();
        cases.push(("q_proj 4b [2048,1536]", 60, Box::new(move || g2.linear(&x, &w))));
    }
    {
        let w = g.tensor(&rng.f32s(8960 * 1536, 0.02, 0.0), &[8960, 1536]);
        let x = g.tensor(&rng.f32s(1536, 0.1, 0.0), &[1, 1536]);
        let g2 = g.clone();
        cases.push(("gemv f32 [8960,1536]", 60, Box::new(move || g2.linear(&x, &w))));
    }
    {
        let q = g.tensor(&rng.f32s(8 * 256, 0.1, 0.0), &[1, 8, 256]);
        let k = g.tensor(&rng.f32s(512 * 256, 0.1, 0.0), &[512, 1, 256]);
        let v = g.tensor(&rng.f32s(512 * 256, 0.1, 0.0), &[512, 1, 256]);
        let kp: Vec<u32> = (0..512).collect();
        let g2 = g.clone();
        cases.push(("attention Dh256 S512", 60, Box::new(move || g2.attention(&q, &k, &v, &gemma_wgpu::backend::AttnOpts { q_pos: &[511], k_pos: &kp, scale: 1.0 / 16.0, sliding_window: 512, attn_softcap: 0.0 }))));
    }
    {
        // chained: measures dependent dispatch-to-dispatch latency
        let mut x = g.tensor(&rng.f32s(1536, 0.1, 0.0), &[1, 1536]);
        let w = g.tensor(&rng.f32s(1536, 0.05, 0.0), &[1536]);
        let g2 = g.clone();
        cases.push(("rmsNorm [1,1536] chained", 400, Box::new(move || { x = g2.rms_norm(&x, Some(&w), 1e-6); x.clone() })));
    }
    {
        let mut a = g.tensor(&rng.f32s(1536, 0.1, 0.0), &[1, 1536]);
        let b = g.tensor(&rng.f32s(1536, 0.1, 0.0), &[1, 1536]);
        let g2 = g.clone();
        cases.push(("add [1,1536] chained", 400, Box::new(move || { a = g2.add(&a, &b); a.clone() })));
    }
    println!("kernel timings (ms/op):");
    for (name, reps, run) in cases.iter_mut() {
        let out = run();
        g.readback(&out); // warmup + pipeline compile
        let t0 = Instant::now();
        let mut out = run();
        for _ in 1..*reps {
            out = run();
        }
        g.readback(&out);
        println!("  {name}: {:.3} ms/op", t0.elapsed().as_secs_f64() * 1e3 / *reps as f64);
    }
}
