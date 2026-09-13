//! wgpu-vs-CPU parity tests. Validates every WGSL kernel against the CPU
//! reference, then the whole Gemma-4 forward pass, then the real checkpoint
//! format through the safetensors loader. Skips (with a message) when no GPU
//! adapter is available.
mod common;

use std::sync::Arc;

use common::*;
use gemma_wgpu::backend::{AttnOpts, Backend, QuantWeight, TensorShape};
use gemma_wgpu::backend_cpu::CpuBackend;
use gemma_wgpu::backend_wgpu::WgpuBackend;
use gemma_wgpu::config::Gemma4Config;
use gemma_wgpu::dequant::pack_unsigned;

use gemma_wgpu::model::Gemma4Model;
use gemma_wgpu::safetensors::{build_safetensors, SafetensorsFile};
use gemma_wgpu::weights::ModelWeights;

fn gpu() -> Option<Arc<WgpuBackend>> {
    match WgpuBackend::create() {
        Ok(b) => Some(Arc::new(b)),
        Err(e) => {
            eprintln!("skipping GPU test: {e}");
            None
        }
    }
}

fn close(label: &str, g: &WgpuBackend, gt: &<WgpuBackend as Backend>::Tensor, ct: &[f32], tol: f32) {
    let gv = g.readback(gt);
    let d = max_abs_diff(&gv, ct);
    assert!(d < tol, "{label}: max diff {d:e} >= {tol:e}");
    eprintln!("  ✓ {label} (max diff {d:e})");
}

#[test]
fn kernel_parity() {
    let Some(g) = gpu() else { return };
    let c = CpuBackend::new();
    let mut r = Lcg::new(7);

    // matmul / linear (tiled f32, M > 1) and gemv (M == 1)
    {
        let (m, k, n) = (17, 40, 23);
        let x = r.rand(m * k, 1.0);
        let w = r.rand(n * k, 1.0);
        close("linear (tiled)", &g, &g.linear(&g.tensor(&x, &[m, k]), &g.tensor(&w, &[n, k])), c.linear(&c.tensor(&x, &[m, k]), &c.tensor(&w, &[n, k])).data(), 2e-3);
        let x1 = r.rand(k, 1.0);
        close("linear (gemv)", &g, &g.linear(&g.tensor(&x1, &[1, k]), &g.tensor(&w, &[n, k])), c.linear(&c.tensor(&x1, &[1, k]), &c.tensor(&w, &[n, k])).data(), 2e-3);
    }
    // rmsNorm (weighted + unweighted)
    {
        let (t, d) = (5, 48);
        let x = r.rand(t * d, 1.0);
        let w = r.rand(d, 0.1);
        close("rmsNorm", &g, &g.rms_norm(&g.tensor(&x, &[t, d]), Some(&g.tensor(&w, &[d])), 1e-6), c.rms_norm(&c.tensor(&x, &[t, d]), Some(&c.tensor(&w, &[d])), 1e-6).data(), 2e-3);
        close("rmsNorm (unweighted)", &g, &g.rms_norm(&g.tensor(&x, &[t, d]), None, 1e-6), c.rms_norm(&c.tensor(&x, &[t, d]), None, 1e-6).data(), 2e-3);
    }
    // elementwise
    {
        let n = 100;
        let a = r.rand(n, 1.0);
        let b = r.rand(n, 1.0);
        let (ga, gb) = (g.tensor(&a, &[n]), g.tensor(&b, &[n]));
        let (ca, cb) = (c.tensor(&a, &[n]), c.tensor(&b, &[n]));
        close("add", &g, &g.add(&ga, &gb), c.add(&ca, &cb).data(), 2e-3);
        close("mul", &g, &g.mul(&ga, &gb), c.mul(&ca, &cb).data(), 2e-3);
        close("scale", &g, &g.scale(&ga, 3.5), c.scale(&ca, 3.5).data(), 2e-3);
        close("geluTanh", &g, &g.gelu_tanh(&ga), c.gelu_tanh(&ca).data(), 2e-3);
        close("geglu", &g, &g.geglu(&ga, &gb), c.geglu(&ca, &cb).data(), 2e-3);
        close("softcap", &g, &g.softcap(&g.scale(&ga, 50.0), 30.0), c.softcap(&c.scale(&ca, 50.0), 30.0).data(), 2e-3);
        let s = [0.37f32];
        close("scaleByTensor", &g, &g.scale_by_tensor(&ga, &g.tensor(&s, &[1])), c.scale_by_tensor(&ca, &c.tensor(&s, &[1])).data(), 2e-3);
        // gelu must stay finite for huge inputs (tanh overflow guard)
        let big = [1000.0f32, -1000.0, 50.0];
        let gv = g.readback(&g.gelu_tanh(&g.tensor(&big, &[3])));
        assert!(gv.iter().all(|v| v.is_finite()), "gelu finite on large inputs: {gv:?}");
        // zeros
        let z = g.readback(&g.zeros(&[37]));
        assert!(z.iter().all(|&v| v == 0.0));
    }
    // embed
    {
        let (v, d) = (30, 16);
        let table = r.rand(v * d, 1.0);
        let ids = [3u32, 7, 0, 29, 12];
        close("embedRows", &g, &g.embed_rows(&g.tensor(&table, &[v, d]), &ids, 4.0), c.embed_rows(&c.tensor(&table, &[v, d]), &ids, 4.0).data(), 2e-3);
    }
    // rope (partial)
    {
        let (t, heads, dh, rot) = (4, 2, 16, 8);
        let pos = [0u32, 1, 5, 9];
        let x = r.rand(t * heads * dh, 1.0);
        close("rope(partial)", &g, &g.rope(&g.tensor(&x, &[t, heads, dh]), &pos, 1e4, dh, heads, rot), c.rope(&c.tensor(&x, &[t, heads, dh]), &pos, 1e4, dh, heads, rot).data(), 2e-3);
    }
    // sliceCols / sliceRows / concatRows
    {
        let (rows, stride) = (4, 20);
        let x = r.rand(rows * stride, 1.0);
        close("sliceCols", &g, &g.slice_cols(&g.tensor(&x, &[rows, stride]), 5, 7), c.slice_cols(&c.tensor(&x, &[rows, stride]), 5, 7).data(), 0.0 + 1e-9);
        close("sliceRows", &g, &g.slice_rows(&g.tensor(&x, &[rows, stride]), 2, 2), c.slice_rows(&c.tensor(&x, &[rows, stride]), 2, 2).data(), 1e-9);
        let a = r.rand(2 * stride, 1.0);
        let b = r.rand(3 * stride, 1.0);
        close("concatRows", &g, &g.concat_rows(Some(&g.tensor(&a, &[2, stride])), &g.tensor(&b, &[3, stride])), c.concat_rows(Some(&c.tensor(&a, &[2, stride])), &c.tensor(&b, &[3, stride])).data(), 1e-9);
        close("concatRows (None)", &g, &g.concat_rows(None, &g.tensor(&b, &[3, stride])), &b, 1e-9);
    }
    // kvAppend: three appends of 200 rows force at least one capacity reallocation (initial cap 256)
    {
        let dim = 8;
        let mut gt = None;
        let mut ct = None;
        for _ in 0..3 {
            let chunk = r.rand(200 * dim, 1.0);
            gt = Some(g.kv_append(gt.as_ref(), &g.tensor(&chunk, &[200, dim])).tensor);
            ct = Some(c.kv_append(ct.as_ref(), &c.tensor(&chunk, &[200, dim])).tensor);
        }
        let (gt, ct) = (gt.unwrap(), ct.unwrap());
        assert_eq!(gt.shape(), &[600, dim]);
        close("kvAppend (growth)", &g, &gt, ct.data(), 1e-9);
    }
    // attention (GQA + sliding window), including a multi-tile (S > 128) case
    {
        let (t, s, hq, hkv, dh) = (6, 6, 4, 1, 16);
        let q = r.rand(t * hq * dh, 1.0);
        let k = r.rand(s * hkv * dh, 1.0);
        let v = r.rand(s * hkv * dh, 1.0);
        let pos: Vec<u32> = (0..6).collect();
        let o = AttnOpts { q_pos: &pos, k_pos: &pos, scale: 1.0 / 4.0, sliding_window: 3, attn_softcap: 0.0 };
        close("attention(GQA,sliding)", &g, &g.attention(&g.tensor(&q, &[t, hq, dh]), &g.tensor(&k, &[s, hkv, dh]), &g.tensor(&v, &[s, hkv, dh]), &o), c.attention(&c.tensor(&q, &[t, hq, dh]), &c.tensor(&k, &[s, hkv, dh]), &c.tensor(&v, &[s, hkv, dh]), &o).data(), 2e-3);

        let (t, s, hq, hkv, dh) = (3, 300, 2, 1, 32);
        let q = r.rand(t * hq * dh, 0.5);
        let k = r.rand(s * hkv * dh, 0.5);
        let v = r.rand(s * hkv * dh, 0.5);
        let kpos: Vec<u32> = (0..300).collect();
        let qpos = [297u32, 298, 299];
        for window in [0u32, 150] {
            let o = AttnOpts { q_pos: &qpos, k_pos: &kpos, scale: 1.0, sliding_window: window, attn_softcap: 0.0 };
            close(&format!("attention(S=300, window={window})"), &g, &g.attention(&g.tensor(&q, &[t, hq, dh]), &g.tensor(&k, &[s, hkv, dh]), &g.tensor(&v, &[s, hkv, dh]), &o), c.attention(&c.tensor(&q, &[t, hq, dh]), &c.tensor(&k, &[s, hkv, dh]), &c.tensor(&v, &[s, hkv, dh]), &o).data(), 2e-3);
        }
    }
}

#[test]
fn quantized_kernel_parity() {
    let Some(g) = gpu() else { return };
    let c = CpuBackend::new();
    let mut r = Lcg::new(9);
    let scales = |r: &mut Lcg, n: usize| -> Vec<f32> { (0..n).map(|_| 0.01 + r.next() * 0.05).collect() };
    let randu = |r: &mut Lcg, n: usize, bits: u32| -> Vec<u32> { (0..n).map(|_| (r.next() * (1u32 << bits) as f32).floor() as u32 % (1 << bits)).collect() };

    // fast path: 4-bit and 2-bit, M > 1 (MROWS=4) and M == 1 (MROWS=1)
    for bits in [4u32, 2] {
        let (n, k) = (40, 64);
        let vals = randu(&mut r, n * k, bits);
        let q = QuantWeight { qbytes: pack_unsigned(&vals, bits), scales: scales(&mut r, n), shape: [n, k], bits, signed_i8: false, group_size: k };
        for m in [5usize, 1] {
            let x = r.rand(m * k, 1.0);
            close(&format!("linear q{bits} M={m}"), &g, &g.linear(&g.tensor(&x, &[m, k]), &g.quant_tensor(q.clone())), c.linear(&c.tensor(&x, &[m, k]), &c.quant_tensor(q.clone())).data(), 2e-3);
        }
    }
    // odd K: not divisible by the fast path's word width -> scalar fallback
    {
        let (bits, n, k, m) = (4u32, 21, 52, 3);
        let vals = randu(&mut r, n * k, bits);
        let q = QuantWeight { qbytes: pack_unsigned(&vals, bits), scales: scales(&mut r, n), shape: [n, k], bits, signed_i8: false, group_size: k };
        let x = r.rand(m * k, 1.0);
        close("linear q4 oddK (scalar)", &g, &g.linear(&g.tensor(&x, &[m, k]), &g.quant_tensor(q.clone())), c.linear(&c.tensor(&x, &[m, k]), &c.quant_tensor(q)).data(), 2e-3);
    }
    // signed int8
    {
        let (n, k, m) = (30, 48, 4);
        let qbytes: Vec<u8> = (0..n * k).map(|_| (r.next() * 256.0).floor() as u32 as u8).collect();
        let q = QuantWeight { qbytes, scales: (0..n).map(|_| 0.01 + r.next() * 0.03).collect(), shape: [n, k], bits: 8, signed_i8: true, group_size: k };
        let x = r.rand(m * k, 1.0);
        close("linear i8", &g, &g.linear(&g.tensor(&x, &[m, k]), &g.quant_tensor(q.clone())), c.linear(&c.tensor(&x, &[m, k]), &c.quant_tensor(q)).data(), 2e-3);
    }
    // quantized embedding with grouped scales (2 groups per row)
    {
        let (v, d, bits, group) = (50, 32, 4u32, 16);
        let vals = randu(&mut r, v * d, bits);
        let s_stride = d / group;
        let q = QuantWeight { qbytes: pack_unsigned(&vals, bits), scales: (0..v * s_stride).map(|_| 0.02 + r.next() * 0.04).collect(), shape: [v, d], bits, signed_i8: false, group_size: group };
        let ids = [3u32, 7, 49, 0, 21];
        close("embed q4 grouped", &g, &g.embed_rows(&g.quant_tensor(q.clone()), &ids, (d as f32).sqrt()), c.embed_rows(&c.quant_tensor(q), &ids, (d as f32).sqrt()).data(), 2e-3);
    }
}

#[test]
fn full_model_parity() {
    let Some(g) = gpu() else { return };
    let c = Arc::new(CpuBackend::new());
    let cfg = tiny_config(48);
    let ids = [2u32, 5, 11, 19, 30];
    let pos: Vec<u32> = (0..5).collect();

    let mut cm = Gemma4Model::new(cfg.clone(), c.clone(), SyntheticWeights::new(cfg.clone(), 123));
    let mut gm = Gemma4Model::new(cfg.clone(), g.clone(), Mirrored::new(SyntheticWeights::new(cfg.clone(), 123), g.clone()));
    let cl = cm.forward(&ids, &pos);
    let gl = gm.forward(&ids, &pos);
    let d = max_abs_diff(&cl, &gl);
    assert!(d < 5e-3, "full forward parity max diff {d:e}");
    eprintln!("  ✓ full prefill forward parity (max diff {d:e})");

    // incremental decode on GPU matches CPU (exercises in-place KV growth + frame recycling)
    let mut ic = Gemma4Model::new(cfg.clone(), c, SyntheticWeights::new(cfg.clone(), 123));
    let mut ig = Gemma4Model::new(cfg.clone(), g.clone(), Mirrored::new(SyntheticWeights::new(cfg.clone(), 123), g.clone()));
    let (mut lc, mut lg) = (Vec::new(), Vec::new());
    for (i, &id) in ids.iter().enumerate() {
        lc = ic.forward(&[id], &[i as u32]);
        lg = ig.forward(&[id], &[i as u32]);
    }
    let d2 = max_abs_diff(&lc, &lg);
    assert!(d2 < 5e-3, "incremental parity max diff {d2:e}");
    eprintln!("  ✓ incremental decode parity (max diff {d2:e})");

    // a long decode forces the KV capacity buffer (256 rows) to grow
    let mut ig2 = Gemma4Model::new(cfg.clone(), g.clone(), Mirrored::new(SyntheticWeights::new(cfg.clone(), 123), g.clone()));
    let mut last = Vec::new();
    for i in 0..300u32 {
        last = ig2.forward(&[(i * 7) % 48], &[i]);
    }
    assert!(last.iter().all(|v| v.is_finite()));
    assert_eq!(ig2.cache.len(), 300);
    eprintln!("  ✓ 300-step decode with KV growth stays finite");
}

/// Integration test for the REAL checkpoint format (without the 2.46 GB
/// download): a tiny safetensors blob using the actual on-disk naming + dtypes
/// (U8/I8 packed weights + scales, BF16 norms, `model.language_model.*`
/// prefix, `.embedding_quantized`), run through `ModelWeights` on both
/// backends.
#[test]
fn real_format_parity() {
    let Some(g) = gpu() else { return };
    let c = Arc::new(CpuBackend::new());

    let cfg = Arc::new(Gemma4Config::from_json(&real_format_config_json(40)));
    let v = cfg.vocab_size;
    let tensors = real_format_tensors(&cfg, 5);
    let blob = build_safetensors(&tensors);
    let wc = ModelWeights::new(vec![SafetensorsFile::from_vec(blob.clone()).unwrap()], cfg.clone(), c.clone());
    let wg = ModelWeights::new(vec![SafetensorsFile::from_vec(blob).unwrap()], cfg.clone(), g.clone());
    let ids = [2u32, 5, 11, 19, 30];
    let pos: Vec<u32> = (0..5).collect();
    let cl = Gemma4Model::new(cfg.clone(), c, wc).forward(&ids, &pos);
    let gl = Gemma4Model::new(cfg.clone(), g.clone(), wg).forward(&ids, &pos);
    assert_eq!(cl.len(), v);
    assert!(cl.iter().all(|x| x.is_finite()), "CPU logits finite");
    assert!(gl.iter().all(|x| x.is_finite()), "GPU logits finite");
    let d = max_abs_diff(&cl, &gl);
    assert!(d < 5e-3, "GPU≡CPU on real format (max diff {d:e})");
    eprintln!("  ✓ loads real names/dtypes, runs both backends, GPU≡CPU (max diff {d:e})");
}

/// The grouped split-K attention kernel (one workgroup per query × kv head ×
/// key split, serving the whole GQA group) only runs at real head shapes; the
/// generic tests above fall back to the per-head kernel. Cover it explicitly:
/// prefill (T > 1, causal), decode with many splits, sliding-window trimming
/// of the active key range, and both head_dims.
#[test]
fn grouped_attention_parity() {
    let Some(g) = gpu() else { return };
    let c = CpuBackend::new();
    let mut r = Lcg::new(21);
    // (t, s, hq, hkv, dh, window)
    let cases = [
        (3usize, 300usize, 8usize, 1usize, 256usize, 0u32),
        (3, 300, 8, 1, 256, 150),
        (1, 700, 8, 1, 512, 0),
        (1, 900, 8, 1, 256, 512),
        (5, 5, 8, 1, 256, 0),
        (6, 70, 8, 1, 256, 3),
        (2, 200, 4, 1, 256, 0),
        (1, 64, 8, 2, 512, 0),
    ];
    for (t, s, hq, hkv, dh, window) in cases {
        let q = r.rand(t * hq * dh, 0.5);
        let k = r.rand(s * hkv * dh, 0.5);
        let v = r.rand(s * hkv * dh, 0.5);
        let kpos: Vec<u32> = (0..s as u32).collect();
        let qpos: Vec<u32> = ((s - t) as u32..s as u32).collect();
        let o = AttnOpts { q_pos: &qpos, k_pos: &kpos, scale: 1.0 / (dh as f32).sqrt(), sliding_window: window, attn_softcap: 0.0 };
        close(
            &format!("grouped attention (T={t}, S={s}, Hq={hq}, Hkv={hkv}, Dh={dh}, window={window})"),
            &g,
            &g.attention(&g.tensor(&q, &[t, hq, dh]), &g.tensor(&k, &[s, hkv, dh]), &g.tensor(&v, &[s, hkv, dh]), &o),
            c.attention(&c.tensor(&q, &[t, hq, dh]), &c.tensor(&k, &[s, hkv, dh]), &c.tensor(&v, &[s, hkv, dh]), &o).data(),
            2e-3,
        );
    }
}

/// Fused ops (single dispatch on the GPU) against their composed reference,
/// including the 8-row prefill tile (M = 9 covers a partial tile) and every
/// MATMUL_Q epilogue mode.
#[test]
fn fused_op_parity() {
    let Some(g) = gpu() else { return };
    let c = CpuBackend::new();
    let mut r = Lcg::new(33);
    let scales = |r: &mut Lcg, n: usize| -> Vec<f32> { (0..n).map(|_| 0.01 + r.next() * 0.05).collect() };
    let randu = |r: &mut Lcg, n: usize, bits: u32| -> Vec<u32> { (0..n).map(|_| (r.next() * (1u32 << bits) as f32).floor() as u32 % (1 << bits)).collect() };
    let qw = |r: &mut Lcg, n: usize, k: usize, bits: u32| -> QuantWeight {
        let vals = randu(r, n * k, bits);
        QuantWeight { qbytes: pack_unsigned(&vals, bits), scales: scales(r, n), shape: [n, k], bits, signed_i8: false, group_size: k }
    };

    // add_rms_norm: residual + norm(x)·w, with and without the layer scalar
    {
        let (t, d) = (5, 48);
        let x = r.rand(t * d, 1.0);
        let res = r.rand(t * d, 1.0);
        let w = r.rand(d, 0.1);
        let s = [1.37f32];
        let (gx, gr, gw, gs) = (g.tensor(&x, &[t, d]), g.tensor(&res, &[t, d]), g.tensor(&w, &[d]), g.tensor(&s, &[1]));
        let (cx, cr, cw, cs) = (c.tensor(&x, &[t, d]), c.tensor(&res, &[t, d]), c.tensor(&w, &[d]), c.tensor(&s, &[1]));
        close("add_rms_norm", &g, &g.add_rms_norm(&gr, &gx, &gw, 1e-6, None), c.add_rms_norm(&cr, &cx, &cw, 1e-6, None).data(), 2e-3);
        close("add_rms_norm (·layer_scalar)", &g, &g.add_rms_norm(&gr, &gx, &gw, 1e-6, Some(&gs)), c.add_rms_norm(&cr, &cx, &cw, 1e-6, Some(&cs)).data(), 2e-3);
    }
    // head_norm_rope: per-head norm + partial rope in one dispatch (Dh 256 / 512 paths and a tiny head)
    for (t, heads, dh, rot) in [(3usize, 4usize, 16usize, 8usize), (2, 8, 256, 256), (2, 1, 512, 128)] {
        let x = r.rand(t * heads * dh, 1.0);
        let w = r.rand(dh, 0.1);
        let pos: Vec<u32> = (0..t as u32).map(|i| i * 7 + 3).collect();
        close(
            &format!("head_norm_rope (heads={heads}, Dh={dh}, rot={rot})"),
            &g,
            &g.head_norm_rope(&g.tensor(&x, &[t, heads, dh]), &g.tensor(&w, &[dh]), 1e-6, &pos, 1e4, dh, heads, rot),
            c.head_norm_rope(&c.tensor(&x, &[t, heads, dh]), &c.tensor(&w, &[dh]), 1e-6, &pos, 1e4, dh, heads, rot).data(),
            2e-3,
        );
    }
    // linear at M = 8 and 9 (MROWS = 8 row tiles) and M = 70 / 64 (shared-memory
    // tiled GEMM, partial and full 64-row tiles, N not a multiple of 64) for every bit width
    for bits in [2u32, 4, 8] {
        let (n, k) = (70, 96);
        let q = if bits == 8 {
            let qbytes: Vec<u8> = (0..n * k).map(|_| (r.next() * 256.0).floor() as u32 as u8).collect();
            QuantWeight { qbytes, scales: scales(&mut r, n), shape: [n, k], bits: 8, signed_i8: true, group_size: k }
        } else {
            qw(&mut r, n, k, bits)
        };
        for m in [8usize, 9, 64, 70] {
            let x = r.rand(m * k, 1.0);
            close(&format!("linear q{bits} M={m}"), &g, &g.linear(&g.tensor(&x, &[m, k]), &g.quant_tensor(q.clone())), c.linear(&c.tensor(&x, &[m, k]), &c.quant_tensor(q.clone())).data(), 4e-3);
        }
    }
    // linear_geglu: gelu(x·Wgᵀ)·(x·Wuᵀ) at decode (M=1), M=4 and prefill (M=9)
    for bits in [2u32, 4] {
        let (n, k) = (40, 64);
        let wg = qw(&mut r, n, k, bits);
        let wu = qw(&mut r, n, k, bits);
        for m in [1usize, 4, 9, 40] {
            let x = r.rand(m * k, 1.0);
            close(
                &format!("linear_geglu q{bits} M={m}"),
                &g,
                &g.linear_geglu(&g.tensor(&x, &[m, k]), &g.quant_tensor(wg.clone()), &g.quant_tensor(wu.clone())),
                c.linear_geglu(&c.tensor(&x, &[m, k]), &c.quant_tensor(wg.clone()), &c.quant_tensor(wu.clone())).data(),
                2e-3,
            );
        }
    }
    // linear_gelu_mul_cols: gelu(x·Wᵀ) · b[:, off:off+N] (the PLE gate)
    {
        let (n, k, stride, off) = (24, 32, 96, 48);
        let qbytes: Vec<u8> = (0..n * k).map(|_| (r.next() * 256.0).floor() as u32 as u8).collect();
        let w = QuantWeight { qbytes, scales: scales(&mut r, n), shape: [n, k], bits: 8, signed_i8: true, group_size: k };
        for m in [1usize, 6, 33] {
            let x = r.rand(m * k, 1.0);
            let b = r.rand(m * stride, 1.0);
            close(
                &format!("linear_gelu_mul_cols i8 M={m}"),
                &g,
                &g.linear_gelu_mul_cols(&g.tensor(&x, &[m, k]), &g.quant_tensor(w.clone()), &g.tensor(&b, &[m, stride]), off),
                c.linear_gelu_mul_cols(&c.tensor(&x, &[m, k]), &c.quant_tensor(w.clone()), &c.tensor(&b, &[m, stride]), off).data(),
                2e-3,
            );
        }
    }
}
