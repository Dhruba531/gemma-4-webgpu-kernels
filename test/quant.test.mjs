// Quantized-kernel parity: GPU vs CPU dequant-matmul / dequant-embed.
// Run: node test/quant.test.mjs
import assert from "node:assert";
import { create, globals } from "webgpu";
Object.assign(globalThis, globals);
import { CpuBackend } from "../src/engine/backend-cpu.js";
import { WebGpuBackend } from "../src/engine/backend-webgpu.js";

const gpu = create([]);
const device = await (await gpu.requestAdapter()).requestDevice();
const G = new WebGpuBackend(device), C = new CpuBackend();
let pass = 0;
const rng = (() => { let a = 9; return () => ((a = (a * 1103515245 + 12345) & 0x7fffffff) / 0x7fffffff); })();
const rand = (n, s = 1) => Float32Array.from({ length: n }, () => (rng() * 2 - 1) * s);

async function close(label, gt, ct, tol = 2e-3) {
  const g = await G.readback(gt), c = ct.data;
  let max = 0; for (let i = 0; i < c.length; i++) max = Math.max(max, Math.abs(g[i] - c[i]));
  assert.ok(max < tol, `${label}: max diff ${max.toExponential(2)}`);
  console.log(`  ✓ ${label} (max diff ${max.toExponential(2)})`); pass++;
}

// pack unsigned sub-byte values LSB-first
function packU(vals, bits) {
  const perByte = 8 / bits, out = new Uint8Array(Math.ceil(vals.length / perByte));
  vals.forEach((v, i) => { out[Math.floor(i / perByte)] |= (v & ((1 << bits) - 1)) << ((i % perByte) * bits); });
  return out;
}

console.log("quantized linear");
for (const bits of [4, 2]) {
  const N = 40, K = 64, M = 5;
  const vals = Array.from({ length: N * K }, () => Math.floor(rng() * (1 << bits)));
  const q = { qbytes: packU(vals, bits), scales: rand(N, 0.1).map(Math.abs) ?? rand(N), shape: [N, K], bits, signedI8: 0, zeroPoint: 1 << (bits - 1), groupSize: K };
  q.scales = Float32Array.from({ length: N }, () => 0.01 + rng() * 0.05);
  const x = rand(M * K);
  await close(`linear q${bits}`, G.linear(G.tensor(x, [M, K]), G.quantTensor(q)), C.linear(C.tensor(x, [M, K]), C.quantTensor(q)));
}
{ // odd K: not divisible by the fast path's word width -> scalar fallback
  const bits = 4, N = 21, K = 52, M = 3;
  const vals = Array.from({ length: N * K }, () => Math.floor(rng() * (1 << bits)));
  const q = { qbytes: packU(vals, bits), scales: Float32Array.from({ length: N }, () => 0.01 + rng() * 0.05), shape: [N, K], bits, signedI8: 0, zeroPoint: 1 << (bits - 1), groupSize: K };
  const x = rand(M * K);
  await close("linear q4 oddK (scalar)", G.linear(G.tensor(x, [M, K]), G.quantTensor(q)), C.linear(C.tensor(x, [M, K]), C.quantTensor(q)));
}
{ // signed int8
  const N = 30, K = 48, M = 4;
  const qbytes = new Uint8Array(N * K);
  for (let i = 0; i < qbytes.length; i++) qbytes[i] = (Math.floor(rng() * 256)) & 0xff; // any int8
  const q = { qbytes, scales: Float32Array.from({ length: N }, () => 0.01 + rng() * 0.03), shape: [N, K], bits: 8, signedI8: 1, zeroPoint: 0, groupSize: K };
  const x = rand(M * K);
  await close("linear i8", G.linear(G.tensor(x, [M, K]), G.quantTensor(q)), C.linear(C.tensor(x, [M, K]), C.quantTensor(q)));
}

console.log("quantized embedding");
{
  const V = 50, D = 32, bits = 4, group = 16; // 2 groups per row
  const vals = Array.from({ length: V * D }, () => Math.floor(rng() * (1 << bits)));
  const sStride = D / group;
  const q = { qbytes: packU(vals, bits), scales: Float32Array.from({ length: V * sStride }, () => 0.02 + rng() * 0.04), shape: [V, D], bits, signedI8: 0, zeroPoint: 1 << (bits - 1), groupSize: group };
  const ids = [3, 7, 49, 0, 21];
  await close("embed q4 grouped", G.embedRows(G.quantTensor(q), ids, Math.sqrt(D)), C.embedRows(C.quantTensor(q), ids, Math.sqrt(D)));
}

console.log(`\n${pass} quant checks passed`);
process.exit(0);
