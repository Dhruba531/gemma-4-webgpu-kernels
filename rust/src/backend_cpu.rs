//! CPU reference backend.
//!
//! Every op the model needs is implemented here in plain Rust. It is the
//! ground truth the wgpu backend is checked against, and it lets the whole
//! transformer run (slowly) with no GPU — which is how the test-suite
//! validates the math.

use std::sync::Arc;

use crate::backend::{AttnOpts, Backend, KvAppend, QuantWeight, TensorShape};

/// A CPU tensor: f32 data + shape, or a packed quantized weight.
#[derive(Clone, Debug)]
pub struct CpuTensor {
    pub data: Arc<Vec<f32>>,
    pub shape: Vec<usize>,
    pub quant: Option<Arc<QuantWeight>>,
}

impl CpuTensor {
    pub fn new(data: Vec<f32>, shape: &[usize]) -> Self {
        Self { data: Arc::new(data), shape: shape.to_vec(), quant: None }
    }
    pub fn data(&self) -> &[f32] {
        &self.data
    }
}

impl TensorShape for CpuTensor {
    fn shape(&self) -> &[usize] {
        &self.shape
    }
    fn size(&self) -> usize {
        if self.quant.is_some() { self.shape.iter().product() } else { self.data.len() }
    }
}

fn t(data: Vec<f32>, shape: &[usize]) -> CpuTensor {
    CpuTensor::new(data, shape)
}

pub fn gelu_tanh_scalar(v: f32) -> f32 {
    let c = (2.0f32 / std::f32::consts::PI).sqrt();
    0.5 * v * (1.0 + (c * (v + 0.044715 * v * v * v)).tanh())
}

#[derive(Default)]
pub struct CpuBackend;

impl CpuBackend {
    pub fn new() -> Self {
        Self
    }

    fn ew(&self, a: &CpuTensor, f: impl Fn(f32) -> f32) -> CpuTensor {
        t(a.data.iter().map(|&v| f(v)).collect(), &a.shape)
    }
}

impl Backend for CpuBackend {
    type Tensor = CpuTensor;

    fn name(&self) -> &str {
        "cpu"
    }

    fn tensor(&self, data: &[f32], shape: &[usize]) -> CpuTensor {
        t(data.to_vec(), shape)
    }

    fn quant_tensor(&self, q: QuantWeight) -> CpuTensor {
        let shape = vec![q.shape[0], q.shape[1]];
        CpuTensor { data: Arc::new(Vec::new()), shape, quant: Some(Arc::new(q)) }
    }

    fn zeros(&self, shape: &[usize]) -> CpuTensor {
        t(vec![0.0; shape.iter().product()], shape)
    }

    fn reshape(&self, x: &CpuTensor, shape: &[usize]) -> CpuTensor {
        CpuTensor { data: x.data.clone(), shape: shape.to_vec(), quant: x.quant.clone() }
    }

    fn embed_rows(&self, table: &CpuTensor, ids: &[u32], scale: f32) -> CpuTensor {
        let d = table.shape[1];
        let mut out = vec![0.0f32; ids.len() * d];
        if let Some(q) = &table.quant {
            for (i, &id) in ids.iter().enumerate() {
                for dd in 0..d {
                    out[i * d + dd] = q.deq(id as usize, dd) * scale;
                }
            }
            return t(out, &[ids.len(), d]);
        }
        for (i, &id) in ids.iter().enumerate() {
            let src = id as usize * d;
            for dd in 0..d {
                out[i * d + dd] = table.data[src + dd] * scale;
            }
        }
        t(out, &[ids.len(), d])
    }

    // RMSNorm: y = x/rms(x) * weight. The qat-mobile checkpoint stores norm
    // weights as the FULL multiplier (plain `w`, not HF-Gemma's `1 + w`).
    fn rms_norm(&self, x: &CpuTensor, w: Option<&CpuTensor>, eps: f32) -> CpuTensor {
        let d = *x.shape.last().unwrap();
        let rows = x.size() / d;
        let mut out = vec![0.0f32; rows * d];
        for i in 0..rows {
            let base = i * d;
            let mut ss = 0.0f32;
            for dd in 0..d {
                let v = x.data[base + dd];
                ss += v * v;
            }
            let inv = 1.0 / (ss / d as f32 + eps).sqrt();
            for dd in 0..d {
                let wv = w.map_or(1.0, |w| w.data[dd]);
                out[base + dd] = x.data[base + dd] * inv * wv;
            }
        }
        t(out, &x.shape)
    }

    fn linear(&self, x: &CpuTensor, w: &CpuTensor) -> CpuTensor {
        let (tt, k) = (x.shape[0], x.shape[1]);
        let n = w.shape[0];
        let mut out = vec![0.0f32; tt * n];
        if let Some(q) = &w.quant {
            // Dequantize each weight row once, then dot it against every input
            // row: same per-element products and summation order as the naive
            // loop, just without re-unpacking the row T times.
            let mut row = vec![0.0f32; k];
            for nn in 0..n {
                for kk in 0..k {
                    row[kk] = q.deq(nn, kk);
                }
                for i in 0..tt {
                    let xb = &x.data[i * k..(i + 1) * k];
                    let mut acc = 0.0f32;
                    for kk in 0..k {
                        acc += xb[kk] * row[kk];
                    }
                    out[i * n + nn] = acc;
                }
            }
            return t(out, &[tt, n]);
        }
        for i in 0..tt {
            let xb = &x.data[i * k..(i + 1) * k];
            for nn in 0..n {
                let wb = &w.data[nn * k..(nn + 1) * k];
                let mut acc = 0.0f32;
                for kk in 0..k {
                    acc += xb[kk] * wb[kk];
                }
                out[i * n + nn] = acc;
            }
        }
        t(out, &[tt, n])
    }

    fn add(&self, a: &CpuTensor, b: &CpuTensor) -> CpuTensor {
        t(a.data.iter().zip(b.data.iter()).map(|(x, y)| x + y).collect(), &a.shape)
    }
    fn mul(&self, a: &CpuTensor, b: &CpuTensor) -> CpuTensor {
        t(a.data.iter().zip(b.data.iter()).map(|(x, y)| x * y).collect(), &a.shape)
    }
    fn scale(&self, a: &CpuTensor, s: f32) -> CpuTensor {
        self.ew(a, |v| v * s)
    }
    fn scale_by_tensor(&self, a: &CpuTensor, s: &CpuTensor) -> CpuTensor {
        self.scale(a, s.data[0])
    }
    fn gelu_tanh(&self, a: &CpuTensor) -> CpuTensor {
        self.ew(a, gelu_tanh_scalar)
    }
    fn geglu(&self, gate: &CpuTensor, up: &CpuTensor) -> CpuTensor {
        self.mul(&self.gelu_tanh(gate), up)
    }
    fn softcap(&self, a: &CpuTensor, cap: f32) -> CpuTensor {
        self.ew(a, |v| cap * (v / cap).tanh())
    }

    // RoPE on q/k laid out [T, heads, headDim]. Half-split convention (HF Gemma),
    // rotating only the first `rotary_dim` dims (partial rotary).
    fn rope(&self, x: &CpuTensor, positions: &[u32], theta: f32, head_dim: usize, heads: usize, rotary_dim: usize) -> CpuTensor {
        let tt = x.shape[0];
        let mut out = x.data.as_ref().clone();
        let half = rotary_dim / 2;
        for i in 0..tt {
            let pos = positions[i] as f32;
            for h in 0..heads {
                let base = (i * heads + h) * head_dim;
                for j in 0..half {
                    let freq = theta.powf(-(2.0 * j as f32) / rotary_dim as f32);
                    let ang = pos * freq;
                    let (sin, cos) = ang.sin_cos();
                    let a = x.data[base + j];
                    let b = x.data[base + j + half];
                    out[base + j] = a * cos - b * sin;
                    out[base + j + half] = b * cos + a * sin;
                }
            }
        }
        t(out, &x.shape)
    }

    // Multi/grouped-query causal attention with optional sliding window.
    fn attention(&self, q: &CpuTensor, k: &CpuTensor, v: &CpuTensor, o: &AttnOpts) -> CpuTensor {
        let (tt, hq, dh) = (q.shape[0], q.shape[1], q.shape[2]);
        let s = k.shape[0];
        let hkv = k.shape[1];
        let dv = v.shape[2];
        let group = hq / hkv;
        let mut out = vec![0.0f32; tt * hq * dv];
        let mut scores = vec![0.0f32; s];
        for i in 0..tt {
            let qpos = o.q_pos[i];
            for h in 0..hq {
                let kvh = h / group;
                let qb = (i * hq + h) * dh;
                let mut maxv = f32::NEG_INFINITY;
                for j in 0..s {
                    let kpos = o.k_pos[j];
                    // causal + optional sliding-window mask
                    if kpos > qpos || (o.sliding_window > 0 && qpos - kpos >= o.sliding_window) {
                        scores[j] = f32::NEG_INFINITY;
                        continue;
                    }
                    let kb = (j * hkv + kvh) * dh;
                    let mut dot = 0.0f32;
                    for d in 0..dh {
                        dot += q.data[qb + d] * k.data[kb + d];
                    }
                    dot *= o.scale;
                    if o.attn_softcap > 0.0 {
                        dot = o.attn_softcap * (dot / o.attn_softcap).tanh();
                    }
                    scores[j] = dot;
                    if dot > maxv {
                        maxv = dot;
                    }
                }
                let mut sum = 0.0f32;
                for j in 0..s {
                    if scores[j] == f32::NEG_INFINITY {
                        scores[j] = 0.0;
                    } else {
                        let e = (scores[j] - maxv).exp();
                        scores[j] = e;
                        sum += e;
                    }
                }
                let inv = if sum > 0.0 { 1.0 / sum } else { 0.0 };
                let ob = (i * hq + h) * dv;
                for j in 0..s {
                    let w = scores[j] * inv;
                    if w == 0.0 {
                        continue;
                    }
                    let vb = (j * hkv + kvh) * dv;
                    for d in 0..dv {
                        out[ob + d] += w * v.data[vb + d];
                    }
                }
            }
        }
        t(out, &[tt, hq * dv])
    }

    fn slice_rows(&self, x: &CpuTensor, start: usize, count: usize) -> CpuTensor {
        let dim = x.shape[1];
        t(x.data[start * dim..(start + count) * dim].to_vec(), &[count, dim])
    }

    fn slice_cols(&self, x: &CpuTensor, offset: usize, width: usize) -> CpuTensor {
        let (rows, stride) = (x.shape[0], x.shape[1]);
        let mut out = Vec::with_capacity(rows * width);
        for r in 0..rows {
            out.extend_from_slice(&x.data[r * stride + offset..r * stride + offset + width]);
        }
        t(out, &[rows, width])
    }

    fn concat_rows(&self, a: Option<&CpuTensor>, b: &CpuTensor) -> CpuTensor {
        let Some(a) = a else { return t(b.data.as_ref().clone(), &b.shape) };
        let mut out = Vec::with_capacity(a.data.len() + b.data.len());
        out.extend_from_slice(&a.data);
        out.extend_from_slice(&b.data);
        let mut shape = a.shape.clone();
        shape[0] += b.shape[0];
        t(out, &shape)
    }

    fn kv_append(&self, existing: Option<&CpuTensor>, add: &CpuTensor) -> KvAppend<CpuTensor> {
        KvAppend { tensor: self.concat_rows(existing, add), dead: existing.cloned() }
    }

    fn readback(&self, x: &CpuTensor) -> Vec<f32> {
        x.data.as_ref().clone()
    }
}
