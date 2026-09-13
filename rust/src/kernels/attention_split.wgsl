// Grouped, split-K, tiled online-softmax attention (causal + sliding window).
//
// One workgroup per (query token, kv head, key split). The workgroup serves
// every query head that shares its kv head (GQA/MQA group, <= 8), so each K/V
// row is fetched once per kv head instead of once per query head — for
// Gemma-4's 8:1 MQA that is 8x less K/V traffic than one workgroup per head.
// Splits partition the active key range [kStart, kEnd) so decode (T = 1) still
// spreads over the GPU; each split writes its running (m, l, acc) partial and
// ATTN_COMBINE merges them. With a single split the normalized result is
// written directly to `outp`.
//
// Keys are processed in tiles of 64. Score phase: one (key, head) pair per
// thread (2 pairs for a group of 8). Softmax bookkeeping: 32 threads per head
// reduce max and sum through workgroup memory. Accumulate phase: each thread
// owns DPT = group·Dh/256 output dims (dim `tid` of several heads), so a V
// element is loaded once and used for every head in the group.
struct P {
  T: u32, S: u32, Hq: u32, Hkv: u32,
  Dh: u32, window: u32, kStart: u32, kEnd: u32,
  splits: u32, keysPerSplit: u32, _p0: u32, _p1: u32,
  scale: f32, _f1: f32, _f2: f32, _f3: f32,
};
@group(0) @binding(0) var<storage, read>       q    : array<vec4<f32>>;  // [T, Hq, Dh]
@group(0) @binding(1) var<storage, read>       k    : array<vec4<f32>>;  // [S, Hkv, Dh]
@group(0) @binding(2) var<storage, read>       v    : array<f32>;        // [S, Hkv, Dh]
@group(0) @binding(3) var<storage, read>       qpos : array<u32>;
@group(0) @binding(4) var<storage, read>       kpos : array<u32>;
@group(0) @binding(5) var<storage, read_write> part : array<f32>;        // [T,Hq,splits,Dh] acc ++ [T,Hq,splits,2] (m,l)
@group(0) @binding(6) var<storage, read_write> outp : array<f32>;        // [T, Hq*Dh]
@group(0) @binding(7) var<uniform>             p    : P;

override DPT : u32 = 8u;         // output dims per thread = group * Dh / 256 (<= MAXDPT)
const MAXDPT : u32 = 16u;
const WG   : u32 = 256u;
const TILE : u32 = 64u;
const MAXG : u32 = 8u;
const LPH  : u32 = 32u;          // reduction lanes per head (WG / MAXG)
const NEG  : f32 = -3.0e38;
var<workgroup> sc   : array<f32, 512>;   // [MAXG][TILE]: scores, then probabilities
var<workgroup> red  : array<f32, 256>;
var<workgroup> mrun : array<f32, 8>;     // running max per head
var<workgroup> lrun : array<f32, 8>;     // running denominator per head
var<workgroup> mnew : array<f32, 8>;
var<workgroup> corr : array<f32, 8>;
var<workgroup> skip_s : u32;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let i     = wid.x;                    // query index
  let kvh   = wid.y;                    // kv head
  let split = wid.z;
  let tid   = lid.x;
  let group = p.Hq / p.Hkv;
  let dh4   = p.Dh / 4u;
  let qp    = qpos[i];
  let rh    = tid / LPH;                // head slot this thread reduces for
  let rl    = tid % LPH;

  // this thread's output dims: o_c = tid + c*256 -> (head o_c / Dh, dim o_c % Dh)
  var acc : array<f32, MAXDPT>;
  var hc  : array<u32, MAXDPT>;
  var dc  : array<u32, MAXDPT>;
  for (var c = 0u; c < DPT; c = c + 1u) {
    let o = tid + c * WG;
    acc[c] = 0.0;
    hc[c] = o / p.Dh;
    dc[c] = o % p.Dh;
  }
  if (tid < MAXG) { mrun[tid] = NEG; lrun[tid] = 0.0; }
  workgroupBarrier();

  let ks = p.kStart + split * p.keysPerSplit;
  let ke = min(ks + p.keysPerSplit, p.kEnd);

  for (var j0 = ks; j0 < ke; j0 = j0 + TILE) {
    let jEnd = min(j0 + TILE, ke);
    let tLen = jEnd - j0;

    // tile-level skips (uniform via workgroupUniformLoad); kpos is ascending
    if (tid == 0u) {
      var v_ = 0u;
      if (kpos[j0] > qp) { v_ = 2u; }                                   // all future: done
      else if (p.window > 0u && qp >= p.window && kpos[jEnd - 1u] <= qp - p.window) { v_ = 1u; } // all pre-window
      skip_s = v_;
    }
    let sk = workgroupUniformLoad(&skip_s);
    if (sk == 2u) { break; }
    if (sk == 1u) { continue; }

    // 1) scores: one (key, head) pair per thread
    for (var pair = tid; pair < TILE * group; pair = pair + WG) {
      let kk = pair / group;
      let h  = pair % group;
      var s = NEG;
      let j = j0 + kk;
      if (kk < tLen) {
        let kp = kpos[j];
        let masked = (kp > qp) || (p.window > 0u && (qp - kp) >= p.window);
        if (!masked) {
          let qb = (i * p.Hq + kvh * group + h) * dh4;
          let kb = (j * p.Hkv + kvh) * dh4;
          var dot4 = vec4<f32>(0.0);
          for (var d4 = 0u; d4 < dh4; d4 = d4 + 1u) { dot4 = dot4 + q[qb + d4] * k[kb + d4]; }
          s = (dot4.x + dot4.y + dot4.z + dot4.w) * p.scale;
        }
      }
      sc[h * TILE + kk] = s;
    }
    workgroupBarrier();

    // 2) per-head tile max (32 lanes per head slot)
    var pm = NEG;
    for (var t = rl; t < TILE; t = t + LPH) { pm = max(pm, sc[rh * TILE + t]); }
    red[tid] = pm;
    workgroupBarrier();
    for (var st = LPH / 2u; st > 0u; st = st >> 1u) {
      if (rl < st) { red[tid] = max(red[tid], red[tid + st]); }
      workgroupBarrier();
    }
    if (tid < MAXG) { mnew[tid] = max(mrun[tid], red[tid * LPH]); }
    workgroupBarrier();

    // 3) probabilities in place (masked scores stay exactly 0)
    for (var pair = tid; pair < TILE * group; pair = pair + WG) {
      let kk = pair / group;
      let h  = pair % group;
      let s = sc[h * TILE + kk];
      sc[h * TILE + kk] = select(0.0, exp(s - mnew[h]), s > NEG);
    }
    workgroupBarrier();

    // 4) per-head tile sum
    var ps = 0.0;
    for (var t = rl; t < TILE; t = t + LPH) { ps = ps + sc[rh * TILE + t]; }
    red[tid] = ps;
    workgroupBarrier();
    for (var st = LPH / 2u; st > 0u; st = st >> 1u) {
      if (rl < st) { red[tid] = red[tid] + red[tid + st]; }
      workgroupBarrier();
    }
    if (tid < MAXG) {
      let cr = exp(mrun[tid] - mnew[tid]);
      corr[tid] = cr;
      lrun[tid] = lrun[tid] * cr + red[tid * LPH];
      mrun[tid] = mnew[tid];
    }
    workgroupBarrier();

    // 5) rescale the running output and fold in this tile's V rows
    for (var c = 0u; c < DPT; c = c + 1u) { acc[c] = acc[c] * corr[hc[c]]; }
    for (var t = 0u; t < tLen; t = t + 1u) {
      let vb = ((j0 + t) * p.Hkv + kvh) * p.Dh;
      for (var c = 0u; c < DPT; c = c + 1u) {
        acc[c] = acc[c] + sc[hc[c] * TILE + t] * v[vb + dc[c]];
      }
    }
    workgroupBarrier(); // sc/corr are rewritten by the next tile
  }

  if (p.splits == 1u) {
    for (var c = 0u; c < DPT; c = c + 1u) {
      let l = lrun[hc[c]];
      let inv = select(0.0, 1.0 / l, l > 0.0);
      outp[(i * p.Hq + kvh * group + hc[c]) * p.Dh + dc[c]] = acc[c] * inv;
    }
  } else {
    for (var c = 0u; c < DPT; c = c + 1u) {
      let slot = (i * p.Hq + kvh * group + hc[c]) * p.splits + split;
      part[slot * p.Dh + dc[c]] = acc[c];
    }
    if (tid < group) {
      let slot = (i * p.Hq + kvh * group + tid) * p.splits + split;
      let mlBase = p.T * p.Hq * p.splits * p.Dh;
      part[mlBase + slot * 2u] = mrun[tid];
      part[mlBase + slot * 2u + 1u] = lrun[tid];
    }
  }
}
