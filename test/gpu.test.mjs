// WebGPU-vs-CPU parity tests, run on a headless Dawn device.
// Run: node test/gpu.test.mjs
//
// Validates every WGSL kernel against the CPU reference, then runs the entire
// Gemma-4 forward pass on the GPU and checks it matches the CPU forward.

import assert from "node:assert";
import { create, globals } from "webgpu";
Object.assign(globalThis, globals);

import { CpuBackend } from "../src/engine/backend-cpu.js";
import { WebGpuBackend } from "../src/engine/backend-webgpu.js";
import { Gemma4Config } from "../src/model/config.js";
import { Gemma4Model } from "../src/model/gemma4.js";
import { syntheticWeights } from "./util.mjs";

const gpu = create([]);
const adapter = await gpu.requestAdapter();
const device = await adapter.requestDevice();
const G = new WebGpuBackend(device);
const C = new CpuBackend();

let pass = 0;
const rng = (() => { let a = 7; return () => ((a = (a * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff); })();
const rand = (n, s = 1) => Float32Array.from({ length: n }, () => (rng() * 2 - 1) * s);

async function close(label, gpuT, cpuT, tol = 2e-3) {
  const g = await G.readback(gpuT);
  const c = cpuT.data ?? cpuT;
  let max = 0;
  for (let i = 0; i < c.length; i++) max = Math.max(max, Math.abs(g[i] - c[i]));
  assert.ok(max < tol, `${label}: max diff ${max.toExponential(2)} >= ${tol}`);
  console.log(`  ✓ ${label} (max diff ${max.toExponential(2)})`);
  pass++;
}

console.log("kernel parity");

// matmul / linear
{
  const M = 17, Kd = 40, N = 23;
  const x = rand(M * Kd), w = rand(N * Kd);
  await close("linear", G.linear(G.tensor(x, [M, Kd]), G.tensor(w, [N, Kd])),
                        C.linear(C.tensor(x, [M, Kd]), C.tensor(w, [N, Kd])));
}
// rmsNorm
{
  const T = 5, D = 48;
  const x = rand(T * D), w = rand(D, 0.1);
  await close("rmsNorm", G.rmsNorm(G.tensor(x, [T, D]), G.tensor(w, [D]), 1e-6),
                         C.rmsNorm(C.tensor(x, [T, D]), C.tensor(w, [D]), 1e-6));
}
// elementwise
{
  const n = 100; const a = rand(n), b = rand(n);
  await close("add", G.add(G.tensor(a, [n]), G.tensor(b, [n])), C.add(C.tensor(a, [n]), C.tensor(b, [n])));
  await close("mul", G.mul(G.tensor(a, [n]), G.tensor(b, [n])), C.mul(C.tensor(a, [n]), C.tensor(b, [n])));
  await close("geluTanh", G.geluTanh(G.tensor(a, [n])), C.geluTanh(C.tensor(a, [n])));
  await close("geglu", G.geglu(G.tensor(a, [n]), G.tensor(b, [n])), C.geglu(C.tensor(a, [n]), C.tensor(b, [n])));
  await close("softcap", G.softcap(G.scale(G.tensor(a, [n]), 50), 30), C.softcap(C.scale(C.tensor(a, [n]), 50), 30));
}
// embed
{
  const V = 30, D = 16; const table = rand(V * D); const ids = [3, 7, 0, 29, 12];
  await close("embedRows", G.embedRows(G.tensor(table, [V, D]), ids, Math.sqrt(D)),
                           C.embedRows(C.tensor(table, [V, D]), ids, Math.sqrt(D)));
}
// rope (partial)
{
  const T = 4, heads = 2, Dh = 16, rot = 8; const pos = [0, 1, 5, 9];
  const x = rand(T * heads * Dh);
  await close("rope(partial)", G.rope(G.tensor(x, [T, heads, Dh]), pos, 1e4, Dh, heads, rot),
                               C.rope(C.tensor(x, [T, heads, Dh]), pos, 1e4, Dh, heads, rot));
}
// sliceCols / sliceRows / concatRows
{
  const rows = 4, stride = 20; const x = rand(rows * stride);
  await close("sliceCols", G.sliceCols(G.tensor(x, [rows, stride]), 5, 7), C.sliceCols(C.tensor(x, [rows, stride]), 5, 7));
  await close("sliceRows", G.sliceRows(G.tensor(x, [rows, stride]), 2, 2), C.sliceRows(C.tensor(x, [rows, stride]), 2, 2));
  const a = rand(2 * stride), b = rand(3 * stride);
  await close("concatRows", G.concatRows(G.tensor(a, [2, stride]), G.tensor(b, [3, stride])),
                            C.concatRows(C.tensor(a, [2, stride]), C.tensor(b, [3, stride])));
}
// kvAppend: in-place growth cache — three appends of 200 rows force at least
// one capacity reallocation (initial cap is 256)
{
  const dim = 8;
  let g = null, c = null;
  for (let i = 0; i < 3; i++) {
    const chunk = rand(200 * dim);
    g = G.kvAppend(g, G.tensor(chunk, [200, dim])).tensor;
    c = C.kvAppend(c, C.tensor(chunk, [200, dim])).tensor;
  }
  assert.deepEqual(g.shape, [600, dim]);
  await close("kvAppend (growth)", g, c);
}
// attention (GQA + sliding window)
{
  const T = 6, S = 6, Hq = 4, Hkv = 1, Dh = 16;
  const q = rand(T * Hq * Dh), k = rand(S * Hkv * Dh), v = rand(S * Hkv * Dh);
  const qPos = [0, 1, 2, 3, 4, 5], kPos = [0, 1, 2, 3, 4, 5];
  const opts = { qPos, kPos, scale: 1 / Math.sqrt(Dh), slidingWindow: 3, attnSoftcap: 0 };
  await close("attention(GQA,sliding)",
    G.attention(G.tensor(q, [T, Hq, Dh]), G.tensor(k, [S, Hkv, Dh]), G.tensor(v, [S, Hkv, Dh]), opts),
    C.attention(C.tensor(q, [T, Hq, Dh]), C.tensor(k, [S, Hkv, Dh]), C.tensor(v, [S, Hkv, Dh]), opts));
}

console.log("\nfull model parity (GPU vs CPU)");
{
  const cfg = new Gemma4Config({ text_config: {
    vocab_size: 48, hidden_size: 32, intermediate_size: 64, num_hidden_layers: 6,
    num_attention_heads: 4, num_key_value_heads: 1, head_dim: 8, global_head_dim: 16,
    hidden_size_per_layer_input: 12, vocab_size_per_layer_input: 48, num_kv_shared_layers: 2,
    sliding_window: 3, rms_norm_eps: 1e-6, final_logit_softcapping: 30.0,
    rope_parameters: { full_attention: { rope_theta: 1e6, partial_rotary_factor: 0.25 }, sliding_attention: { rope_theta: 1e4 } },
    layer_types: ["sliding_attention","full_attention","sliding_attention","sliding_attention","sliding_attention","sliding_attention"],
  }});
  const wC = syntheticWeights(cfg, 123);
  // mirror the same weights onto the GPU
  const wG = { _c: new Map(), get(n){ if(this._c.has(n)) return this._c.get(n); const t=wC.get(n); const g=G.tensor(t.data, t.shape); this._c.set(n,g); return g; } };

  const ids = [2, 5, 11, 19, 30];
  const posList = ids.map((_, i) => i);
  const cpuLogits = await new Gemma4Model(cfg, C, wC).forward(ids, posList);
  const gpuLogits = await new Gemma4Model(cfg, G, wG).forward(ids, posList);
  let max = 0;
  for (let i = 0; i < cpuLogits.length; i++) max = Math.max(max, Math.abs(cpuLogits[i] - gpuLogits[i]));
  assert.ok(max < 5e-3, `full forward parity max diff ${max.toExponential(2)}`);
  console.log(`  ✓ full prefill forward parity (max diff ${max.toExponential(2)})`);
  pass++;

  // incremental decode on GPU matches CPU
  const incG = new Gemma4Model(cfg, G, wG), incC = new Gemma4Model(cfg, C, wC);
  let lg, lc;
  for (let i = 0; i < ids.length; i++) { lg = await incG.forward([ids[i]], [i]); lc = await incC.forward([ids[i]], [i]); }
  let max2 = 0; for (let i = 0; i < lc.length; i++) max2 = Math.max(max2, Math.abs(lc[i] - lg[i]));
  assert.ok(max2 < 5e-3, `incremental parity max diff ${max2.toExponential(2)}`);
  console.log(`  ✓ incremental decode parity (max diff ${max2.toExponential(2)})`);
  pass++;
}

console.log(`\n${pass} GPU checks passed`);
process.exit(0);
