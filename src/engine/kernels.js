// WGSL compute kernels for the Gemma-4 forward pass.
//
// Each exported string is a complete shader module. They are written to mirror,
// op-for-op, the CPU reference in backend-cpu.js, so GPU and CPU outputs agree.
// Everything is f32 for clarity and portability (no shader-f16 feature needed);
// switching weights/activations to f16 is the obvious next optimization.
//
// Binding convention per dispatch: storage buffers first (read, then
// read_write), then a single uniform "params" buffer last.

// ---------------------------------------------------------------------------
// Tiled matmul:  C[M,N] = A[M,K] · Bᵀ   where B is [N,K] (PyTorch nn.Linear).
// Shared-memory tiles of 16×16; one thread computes one C element.
// Both tile loads walk K with lid.x so global reads are coalesced (B rows are
// contiguous in K; loading Bs with lid.x along N would stride by K per thread).
// ---------------------------------------------------------------------------
export const MATMUL = /* wgsl */ `
struct Dims { M: u32, K: u32, N: u32, _pad: u32 };
@group(0) @binding(0) var<storage, read>       A : array<f32>;
@group(0) @binding(1) var<storage, read>       Bm: array<f32>;
@group(0) @binding(2) var<storage, read_write> C : array<f32>;
@group(0) @binding(3) var<uniform>             d : Dims;

const TS : u32 = 16u;
var<workgroup> As : array<f32, 256>; // [mLocal][kLocal]
var<workgroup> Bs : array<f32, 256>; // [nLocal][kLocal]

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(local_invocation_id) lid : vec3<u32>,
        @builtin(workgroup_id) wid : vec3<u32>) {
  let row = wid.y * TS + lid.y;   // m
  let col = wid.x * TS + lid.x;   // n
  var acc = 0.0;
  let ntiles = (d.K + TS - 1u) / TS;
  for (var t = 0u; t < ntiles; t = t + 1u) {
    let kk = t * TS + lid.x;
    let bRow = wid.x * TS + lid.y;
    As[lid.y * TS + lid.x] = select(0.0, A[row * d.K + kk], row < d.M && kk < d.K);
    Bs[lid.y * TS + lid.x] = select(0.0, Bm[bRow * d.K + kk], bRow < d.N && kk < d.K);
    workgroupBarrier();
    for (var p = 0u; p < TS; p = p + 1u) {
      acc = acc + As[lid.y * TS + p] * Bs[lid.x * TS + p];
    }
    workgroupBarrier();
  }
  if (row < d.M && col < d.N) { C[row * d.N + col] = acc; }
}
`;

// ---------------------------------------------------------------------------
// f32 matvec:  C[1,N] = A[1,K] · Bᵀ. One workgroup per output row; lanes
// stride K (coalesced along the contiguous B row) and tree-reduce. The tiled
// MATMUL wastes 15/16 of each tile when M == 1; this doesn't.
// ---------------------------------------------------------------------------
export const GEMV = /* wgsl */ `
struct P { K: u32, N: u32, _p0: u32, _p1: u32 };
@group(0) @binding(0) var<storage, read>       A : array<f32>;
@group(0) @binding(1) var<storage, read>       Bm: array<f32>;
@group(0) @binding(2) var<storage, read_write> C : array<f32>;
@group(0) @binding(3) var<uniform>             p : P;

const WG : u32 = 256u;
var<workgroup> red : array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let base = wid.x * p.K;
  var s = 0.0;
  for (var k = lid.x; k < p.K; k = k + WG) { s = s + A[k] * Bm[base + k]; }
  red[lid.x] = s;
  workgroupBarrier();
  for (var st = WG / 2u; st > 0u; st = st >> 1u) {
    if (lid.x < st) { red[lid.x] = red[lid.x] + red[lid.x + st]; }
    workgroupBarrier();
  }
  if (lid.x == 0u) { C[wid.x] = red[0]; }
}
`;

// ---------------------------------------------------------------------------
// Quantized matmul (fast path):  C[M,N] = A[M,K]·dequant(Wq)ᵀ, per-row scales.
//
// A 256-thread workgroup computes 32 output columns × MROWS input rows: each
// output column gets LANES=8 lanes that stride its packed row in whole u32
// words. Eight consecutive u32 loads form a 32-byte segment, wide enough to
// coalesce on current GPUs, while 32 rows per workgroup keeps per-row
// reduction cost at just three barrier rounds — this matters because quantized
// rows are short (a 2-bit 1536-wide row is 96 words) and the weight stream is
// the dominant per-token traffic. Each 4-byte load yields 32/bits values; sign
// extension for every width (2/4/8 bit, two's complement) is one shift-up +
// arithmetic shift-down.
//
// MROWS and BITS are pipeline overrides, so both unpack loops fully unroll
// with constant shift amounts (a runtime `bits` uniform kept this kernel
// value-throughput-bound at ~50 Gvalues/s). MROWS is 1 for decode (single-row
// A: the row guard const-folds away) and 4 for prefill, where reusing each
// loaded word across 4 rows of A quarters the weight traffic.
// Requires K % (32/bits) == 0 and one scale per output row; the backend falls
// back to MATMUL_Q_SCALAR otherwise.
// ---------------------------------------------------------------------------
export const MATMUL_Q = /* wgsl */ `
struct Dims { M:u32, K:u32, N:u32, bits:u32, wordsPerRow:u32, _p0:u32, _p1:u32, _p2:u32 };
@group(0) @binding(0) var<storage, read>       A     : array<f32>;
@group(0) @binding(1) var<storage, read>       Wq    : array<u32>; // packed rows
@group(0) @binding(2) var<storage, read>       scale : array<f32>; // [N]
@group(0) @binding(3) var<storage, read_write> C     : array<f32>;
@group(0) @binding(4) var<uniform>             d     : Dims;

override MROWS : u32 = 4u;
override BITS  : u32 = 4u;
const LANES : u32 = 8u;
const OUTS  : u32 = 32u;
var<workgroup> red : array<vec4<f32>, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let g    = lid.x / LANES;             // output slot within the workgroup
  let lane = lid.x % LANES;
  let n    = wid.x * OUTS + g;
  let nc   = min(n, d.N - 1u);          // clamp: keeps control flow uniform
  let m0   = wid.y * MROWS;
  let rowBase = nc * d.wordsPerRow;
  let mask = (1u << BITS) - 1u;                              // const-folds: BITS is override
  let zp   = f32(1u << (BITS - 1u));
  var acc  = vec4<f32>(0.0);
  for (var w = lane; w < d.wordsPerRow; w = w + LANES) {
    let word = Wq[rowBase + w];
    let k0 = w * (32u / BITS);
    for (var j = 0u; j < 32u / BITS; j = j + 1u) {           // unrolls: BITS is override
      let raw = (word >> (BITS * j)) & mask;
      var qv: f32;
      if (BITS == 8u) {
        qv = f32(bitcast<i32>(raw << 24u) >> 24u);           // native signed I8
      } else {
        qv = f32(raw) - zp;                                  // offset binary (zero-point)
      }
      if (MROWS == 1u) {
        acc.x = acc.x + A[m0 * d.K + k0 + j] * qv;           // m0 < M by dispatch
      } else {
        for (var mi = 0u; mi < MROWS; mi = mi + 1u) {
          if (m0 + mi < d.M) {
            acc[mi] = acc[mi] + A[(m0 + mi) * d.K + k0 + j] * qv;
          }
        }
      }
    }
  }
  red[lid.x] = acc;
  workgroupBarrier();
  for (var st = LANES / 2u; st > 0u; st = st >> 1u) {
    if (lane < st) { red[lid.x] = red[lid.x] + red[lid.x + st]; }
    workgroupBarrier();
  }
  if (lane == 0u && n < d.N) {
    let sn = scale[n];
    for (var mi = 0u; mi < MROWS; mi = mi + 1u) {
      if (m0 + mi < d.M) { C[(m0 + mi) * d.N + n] = red[lid.x][mi] * sn; }
    }
  }
}
`;

// ---------------------------------------------------------------------------
// Quantized matmul (scalar reference):  C[M,N] = A[M,K]·dequant(Wq)ᵀ
// One thread per output element, byte-at-a-time unpack. Slow but shape-
// agnostic; kept as the fallback for K not divisible by the word width.
// ---------------------------------------------------------------------------
export const MATMUL_Q_SCALAR = /* wgsl */ `
struct Dims { M:u32, K:u32, N:u32, bits:u32, perByte:u32, signedI8:u32, _pad0:u32, _pad1:u32 };
@group(0) @binding(0) var<storage, read>       A     : array<f32>;
@group(0) @binding(1) var<storage, read>       Wq    : array<u32>; // packed bytes
@group(0) @binding(2) var<storage, read>       scale : array<f32>; // [N]
@group(0) @binding(3) var<storage, read_write> C     : array<f32>;
@group(0) @binding(4) var<uniform>             d     : Dims;

fn getByte(i: u32) -> u32 { return (Wq[i >> 2u] >> (8u * (i & 3u))) & 0xffu; }

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let n = gid.x;
  let m = gid.y;
  if (n >= d.N || m >= d.M) { return; }
  let s = scale[n];
  let mask = (1u << d.bits) - 1u;
  let rowBytes = select(d.K / d.perByte, d.K, d.signedI8 == 1u);
  let rowBase = n * rowBytes;
  let aBase = m * d.K;
  var acc = 0.0;
  for (var k = 0u; k < d.K; k = k + 1u) {
    var qv: i32;
    if (d.signedI8 == 1u) {
      let b = getByte(rowBase + k);
      qv = select(i32(b), i32(b) - 256, b > 127u); // sign-extend int8
    } else {
      let b = getByte(rowBase + k / d.perByte);
      let shift = (k % d.perByte) * d.bits;
      // offset binary: value = raw - zeroPoint, zeroPoint = 2^(bits-1)
      qv = i32((b >> shift) & mask) - (1 << (d.bits - 1u));
    }
    acc = acc + A[aBase + k] * (f32(qv) * s);
  }
  C[m * d.N + n] = acc;
}
`;

// ---------------------------------------------------------------------------
// Quantized embedding gather + dequant + scale.
//   out[i,dd] = unpack(ids[i], dd) * scale[ids[i], dd/group] * embScale
// scale tensor is [V, D/group] (group=D for per-row, or per-layer block).
// ---------------------------------------------------------------------------
export const EMBED_Q = /* wgsl */ `
struct P { T:u32, D:u32, bits:u32, perByte:u32, group:u32, scaleStride:u32, embScale:f32, _pad:u32 };
@group(0) @binding(0) var<storage, read>       Wq    : array<u32>;
@group(0) @binding(1) var<storage, read>       ids   : array<u32>;
@group(0) @binding(2) var<storage, read>       scale : array<f32>;
@group(0) @binding(3) var<storage, read_write> outp  : array<f32>;
@group(0) @binding(4) var<uniform>             p     : P;

fn getByte2(i: u32) -> u32 { return (Wq[i >> 2u] >> (8u * (i & 3u))) & 0xffu; }

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= p.T * p.D) { return; }
  let i = idx / p.D;
  let dd = idx % p.D;
  let id = ids[i];
  let rowBytes = p.D / p.perByte;
  let b = getByte2(id * rowBytes + dd / p.perByte);
  let shift = (dd % p.perByte) * p.bits;
  let mask = (1u << p.bits) - 1u;
  // offset binary: value = raw - zeroPoint, zeroPoint = 2^(bits-1)
  let qv = i32((b >> shift) & mask) - (1 << (p.bits - 1u));
  let s = scale[id * p.scaleStride + dd / p.group];
  outp[idx] = f32(qv) * s * p.embScale;
}
`;

// ---------------------------------------------------------------------------
// Embedding gather + scale:  out[i, :] = table[ids[i], :] * scale
// ---------------------------------------------------------------------------
export const EMBED = /* wgsl */ `
struct P { T: u32, D: u32, scale: f32, _pad: u32 };
@group(0) @binding(0) var<storage, read>       table : array<f32>;
@group(0) @binding(1) var<storage, read>       ids   : array<u32>;
@group(0) @binding(2) var<storage, read_write> outp  : array<f32>;
@group(0) @binding(3) var<uniform>             p     : P;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= p.T * p.D) { return; }
  let i = idx / p.D;
  let dd = idx % p.D;
  outp[idx] = table[ids[i] * p.D + dd] * p.scale;
}
`;

// ---------------------------------------------------------------------------
// Column slice: out[r, 0:W] = in[r, offset:offset+W]  (in is [rows, stride]).
// ---------------------------------------------------------------------------
export const SLICE_COLS = /* wgsl */ `
struct P { rows: u32, stride: u32, offset: u32, width: u32 };
@group(0) @binding(0) var<storage, read>       inp  : array<f32>;
@group(0) @binding(1) var<storage, read_write> outp : array<f32>;
@group(0) @binding(2) var<uniform>             p    : P;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= p.rows * p.width) { return; }
  let r = idx / p.width;
  let c = idx % p.width;
  outp[idx] = inp[r * p.stride + p.offset + c];
}
`;

// ---------------------------------------------------------------------------
// RMSNorm: y = x / sqrt(mean(x²)+eps) * w. One workgroup per row.
// The qat-mobile checkpoint stores norm weights as the FULL multiplier (plain
// `w`, matching the reference runtime — not HF-Gemma's `1 + w`). hasW = 0
// runs the unweighted variant (v-norm ahead of attention); the w binding is
// still read, so callers pass any large-enough dummy buffer.
// ---------------------------------------------------------------------------
export const RMSNORM = /* wgsl */ `
struct P { rows: u32, D: u32, eps: f32, hasW: u32 };
@group(0) @binding(0) var<storage, read>       x : array<f32>;
@group(0) @binding(1) var<storage, read>       w : array<f32>;
@group(0) @binding(2) var<storage, read_write> y : array<f32>;
@group(0) @binding(3) var<uniform>             p : P;

const WG : u32 = 256u;
var<workgroup> partial : array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let row = wid.x;
  let base = row * p.D;
  var s = 0.0;
  for (var d = lid.x; d < p.D; d = d + WG) { let v = x[base + d]; s = s + v * v; }
  partial[lid.x] = s;
  workgroupBarrier();
  for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
    if (lid.x < stride) { partial[lid.x] = partial[lid.x] + partial[lid.x + stride]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(partial[0] / f32(p.D) + p.eps);
  for (var d = lid.x; d < p.D; d = d + WG) {
    let wv = select(1.0, w[d], p.hasW == 1u);
    y[base + d] = x[base + d] * inv * wv;
  }
}
`;

// ---------------------------------------------------------------------------
// Elementwise binary / unary ops, selected by `op` in params.
//   0: add(a,b)  1: mul(a,b)  2: scale(a, k)  3: geluTanh(a)
//   4: geglu -> a = gelu(a)*b   5: softcap -> k*tanh(a/k)
//   6: scale by tensor scalar -> a * b[0]  (layer_scalar)
// ---------------------------------------------------------------------------
export const ELEMENTWISE = /* wgsl */ `
struct P { n: u32, op: u32, k: f32, _pad: u32 };
@group(0) @binding(0) var<storage, read>       a : array<f32>;
@group(0) @binding(1) var<storage, read>       b : array<f32>;
@group(0) @binding(2) var<storage, read_write> o : array<f32>;
@group(0) @binding(3) var<uniform>             p : P;

// tanh(x) overflows to NaN for large |x| on some backends (e^x = inf), because
// the cubic term in gelu can push the argument into the thousands. Clamp first;
// tanh saturates to ±1 well before ±20.
fn tanh_safe(x: f32) -> f32 { return tanh(clamp(x, -20.0, 20.0)); }

fn gelu(v: f32) -> f32 {
  let c = 0.7978845608028654; // sqrt(2/pi)
  return 0.5 * v * (1.0 + tanh_safe(c * (v + 0.044715 * v * v * v)));
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  switch p.op {
    case 0u: { o[i] = a[i] + b[i]; }
    case 1u: { o[i] = a[i] * b[i]; }
    case 2u: { o[i] = a[i] * p.k; }
    case 3u: { o[i] = gelu(a[i]); }
    case 4u: { o[i] = gelu(a[i]) * b[i]; }
    case 5u: { o[i] = p.k * tanh_safe(a[i] / p.k); }
    case 6u: { o[i] = a[i] * b[0]; }
    default: { o[i] = a[i]; }
  }
}
`;

// ---------------------------------------------------------------------------
// RoPE (half-split, partial rotary). x laid out [T, heads, headDim].
// Rotates only the first `rotaryDim` dims of each head.
// ---------------------------------------------------------------------------
export const ROPE = /* wgsl */ `
struct P { T: u32, heads: u32, headDim: u32, rotaryDim: u32, theta: f32, _p0: f32, _p1: f32, _p2: f32 };
@group(0) @binding(0) var<storage, read>       x   : array<f32>;
@group(0) @binding(1) var<storage, read>       pos : array<u32>;
@group(0) @binding(2) var<storage, read_write> y   : array<f32>;
@group(0) @binding(3) var<uniform>             p   : P;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let total = p.T * p.heads * p.headDim;
  let idx = gid.x;
  if (idx >= total) { return; }
  let d  = idx % p.headDim;
  let hi = idx / p.headDim;        // flattened (t,head)
  let ti = hi / p.heads;
  let half = p.rotaryDim / 2u;
  if (d >= p.rotaryDim) { y[idx] = x[idx]; return; }   // pass-through tail
  let base = hi * p.headDim;
  let angleFreqIdx = select(d, d - half, d >= half);
  let freq = pow(p.theta, -2.0 * f32(angleFreqIdx) / f32(p.rotaryDim));
  let ang = f32(pos[ti]) * freq;
  let c = cos(ang); let s = sin(ang);
  if (d < half) {
    y[idx] = x[base + d] * c - x[base + d + half] * s;
  } else {
    y[idx] = x[base + d] * c + x[base + d - half] * s;
  }
}
`;

// ---------------------------------------------------------------------------
// Attention (tiled online-softmax). One workgroup per (query, head), dispatched
// (T, Hq). Keys are processed in tiles of 128: each thread scores one key of
// the tile (dot over head_dim), two tree reductions produce the tile max and
// exp-sum, then threads switch to striding head_dim to fold the tile's
// probability-weighted V rows into the running accumulator — ~19 barriers per
// 128 keys instead of ~9 per key. Causal + optional sliding-window masking,
// GQA via head→kv map.
//
// kpos is assumed ascending (the KV cache appends in generation order), which
// enables tile-level skips: a tile entirely in the future breaks the loop, a
// tile entirely older than the sliding window is skipped — decode work on
// local layers stays O(window) as the cache grows. Per-key masks still apply
// inside boundary tiles, so results are exact either way.
// q:[T,Hq,Dh]  k:[S,Hkv,Dh]  v:[S,Hkv,Dh]  ->  out:[T,Hq*Dh]
// ---------------------------------------------------------------------------
export const ATTENTION = /* wgsl */ `
struct P {
  T: u32, S: u32, Hq: u32, Hkv: u32,
  Dh: u32, window: u32, _pad0: u32, _pad1: u32,
  scale: f32, _pad2: f32, _pad3: f32, _pad4: f32,
};
@group(0) @binding(0) var<storage, read>       q    : array<f32>;
@group(0) @binding(1) var<storage, read>       k    : array<f32>;
@group(0) @binding(2) var<storage, read>       v    : array<f32>;
@group(0) @binding(3) var<storage, read>       qpos : array<u32>;
@group(0) @binding(4) var<storage, read>       kpos : array<u32>;
@group(0) @binding(5) var<storage, read_write> outp : array<f32>;
@group(0) @binding(6) var<uniform>             p    : P;

const WG   : u32 = 128u;
const TILE : u32 = 128u;
const NEG  : f32 = -3.0e38;
var<workgroup> sc  : array<f32, 128>;   // tile scores, then tile probabilities
var<workgroup> red : array<f32, 128>;
var<workgroup> acc : array<f32, 512>;   // running output, <= max headDim
var<workgroup> m_s : f32;               // running max
var<workgroup> l_s : f32;               // running denom
var<workgroup> skip_s : u32;

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let i = wid.x;                 // query index
  let h = wid.y;                 // head
  let group = p.Hq / p.Hkv;
  let kvh = h / group;
  let tid = lid.x;

  for (var d = tid; d < p.Dh; d = d + WG) { acc[d] = 0.0; }
  if (tid == 0u) { m_s = NEG; l_s = 0.0; }
  workgroupBarrier();

  let qBase = (i * p.Hq + h) * p.Dh;
  let qp = qpos[i];

  for (var j0 = 0u; j0 < p.S; j0 = j0 + TILE) {
    let jEnd = min(j0 + TILE, p.S);

    // tile-level skip decisions (uniform via workgroupUniformLoad)
    if (tid == 0u) {
      var v_ = 0u;
      if (kpos[j0] > qp) { v_ = 2u; }                     // all future: done
      else if (p.window > 0u && qp >= p.window && kpos[jEnd - 1u] <= qp - p.window) {
        v_ = 1u;                                          // all pre-window
      }
      skip_s = v_;
    }
    let sk = workgroupUniformLoad(&skip_s);
    if (sk == 2u) { break; }
    if (sk == 1u) { continue; }

    // 1) one score per thread
    let j = j0 + tid;
    var s = NEG;
    var valid = false;
    if (j < jEnd) {
      let kp = kpos[j];
      let masked = (kp > qp) || (p.window > 0u && (qp - kp) >= p.window);
      if (!masked) {
        let kBase = (j * p.Hkv + kvh) * p.Dh;
        var dot = 0.0;
        for (var d = 0u; d < p.Dh; d = d + 1u) { dot = dot + q[qBase + d] * k[kBase + d]; }
        s = dot * p.scale;
        valid = true;
      }
    }
    red[tid] = s;
    workgroupBarrier();
    for (var st = WG / 2u; st > 0u; st = st >> 1u) {
      if (tid < st) { red[tid] = max(red[tid], red[tid + st]); }
      workgroupBarrier();
    }
    let mOld = m_s;
    let mNew = max(mOld, red[0]);
    let corr = exp(mOld - mNew);
    let pv = select(0.0, exp(s - mNew), valid);
    workgroupBarrier(); // red/m_s reads above must finish before reuse below

    // 2) tile exp-sum
    sc[tid] = pv;
    red[tid] = pv;
    workgroupBarrier();
    for (var st = WG / 2u; st > 0u; st = st >> 1u) {
      if (tid < st) { red[tid] = red[tid] + red[tid + st]; }
      workgroupBarrier();
    }

    // 3) rescale accumulator and fold in the tile's V rows
    let tLen = jEnd - j0;
    for (var d = tid; d < p.Dh; d = d + WG) {
      var a = acc[d] * corr;
      for (var t = 0u; t < tLen; t = t + 1u) {
        a = a + sc[t] * v[((j0 + t) * p.Hkv + kvh) * p.Dh + d];
      }
      acc[d] = a;
    }
    if (tid == 0u) { l_s = l_s * corr + red[0]; m_s = mNew; }
    workgroupBarrier();
  }

  let oBase = (i * p.Hq + h) * p.Dh;
  let inv = select(0.0, 1.0 / l_s, l_s > 0.0);
  for (var d = tid; d < p.Dh; d = d + WG) { outp[oBase + d] = acc[d] * inv; }
}
`;
