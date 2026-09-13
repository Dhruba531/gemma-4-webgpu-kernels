// Dequant-on-the-fly matmul  C[M,N] = A[M,K] · Wᵀ  with W packed 2/4/8-bit and
// one f32 scale per output row.
//
// Work split: LANES lanes cooperate on one output row (each owns every LANES-th
// packed word), 256/LANES rows per 256-thread workgroup — the host picks LANES
// by N so narrow projections still spread over the GPU — and MROWS activation
// rows per workgroup so a packed word is unpacked once and reused across MROWS
// rows at prefill. Activations are read as vec4 (K is a multiple of 4 on this
// path), one 16-byte load per four MACs. The word loop is unrolled 4× with the
// loads hoisted so each thread keeps several independent memory requests in
// flight; the kernel is latency-bound otherwise.
//
// Unpack trick: offset-binary values (raw - 2^(bits-1)) are exactly the two's
// complement reading of raw with its top bit flipped, so one XOR per packed
// word turns every sub-byte unpack into a sign-extending shift pair — no mask,
// no subtract. Native I8 (BITS = 8) is already two's complement (XM = 0).
//
// MODE fuses the ops that follow a projection in Gemma-4:
//   0: C = acc·s                                   (plain linear)
//   1: C = gelu(acc·s) · (acc2·s2)                 (gate_proj + up_proj + GeGLU, second weight in Wq2/scale2)
//   2: C = gelu(acc·s) · Bs[m·bStride + bOff + n]  (per-layer-input gate · PLE slice)
struct Dims { M:u32, K:u32, N:u32, wordsPerRow:u32, bOff:u32, bStride:u32, _p0:u32, _p1:u32 };
@group(0) @binding(0) var<storage, read>       A      : array<vec4<f32>>; // [M, K/4]
@group(0) @binding(1) var<storage, read>       Wq     : array<u32>;       // packed rows
@group(0) @binding(2) var<storage, read>       scale  : array<f32>;       // [N]
@group(0) @binding(3) var<storage, read>       Wq2    : array<u32>;       // MODE 1 only
@group(0) @binding(4) var<storage, read>       scale2 : array<f32>;       // MODE 1 only
@group(0) @binding(5) var<storage, read>       Bs     : array<f32>;       // MODE 2 only
@group(0) @binding(6) var<storage, read_write> C      : array<f32>;
@group(0) @binding(7) var<uniform>             d      : Dims;

override MROWS : u32 = 4u;   // activation rows per workgroup: 1 (decode), 4 or 8 (prefill)
override BITS  : u32 = 4u;   // 2, 4 or 8
override MODE  : u32 = 0u;
override LANES : u32 = 8u;   // lanes per output row (power of two, 4..64)
const WG : u32 = 256u;
var<workgroup> red : array<vec4<f32>, 256>;

fn tanh_safe(x: f32) -> f32 { return tanh(clamp(x, -20.0, 20.0)); }
fn gelu(v: f32) -> f32 {
  return 0.5 * v * (1.0 + tanh_safe(0.7978845608028654 * (v + 0.044715 * v * v * v)));
}

// Values 4c..4c+3 of a (pre-XORed) packed word as signed floats. Signed
// extractBits is a single bit-field-extract instruction on Metal.
fn unpack4(word: u32, c: u32) -> vec4<f32> {
  let b0 = BITS * 4u * c;
  let sw = bitcast<i32>(word);
  return vec4<f32>(
    f32(extractBits(sw, b0, BITS)),
    f32(extractBits(sw, b0 + BITS, BITS)),
    f32(extractBits(sw, b0 + 2u * BITS, BITS)),
    f32(extractBits(sw, b0 + 3u * BITS, BITS)));
}

// Multiply-accumulate one packed word (values k0 .. k0 + 32/BITS) of the
// weight row(s) against the MROWS activation rows.
fn mac_word(word: u32, word2: u32, w: u32, m0: u32, mLast: u32, k4row: u32,
            acc: ptr<function, array<vec4<f32>, 2>>, acc2: ptr<function, array<vec4<f32>, 2>>) {
  let v4 = 8u / BITS;                   // vec4 chunks per packed word
  let kb = w * v4;
  for (var c = 0u; c < v4; c = c + 1u) {
    let qv = unpack4(word, c);
    var qv2 = vec4<f32>(0.0);
    if (MODE == 1u) { qv2 = unpack4(word2, c); }
    if (MROWS == 1u) {
      let a = A[m0 * k4row + kb + c];
      (*acc)[0].x = (*acc)[0].x + dot(a, qv);
      if (MODE == 1u) { (*acc2)[0].x = (*acc2)[0].x + dot(a, qv2); }
    } else {
      for (var mi = 0u; mi < MROWS; mi = mi + 1u) {
        // rows past M re-read the last row (discarded at the store): keeps
        // the loop bound constant so it unrolls and acc stays in registers
        let a = A[min(m0 + mi, mLast) * k4row + kb + c];
        (*acc)[mi >> 2u][mi & 3u] = (*acc)[mi >> 2u][mi & 3u] + dot(a, qv);
        if (MODE == 1u) { (*acc2)[mi >> 2u][mi & 3u] = (*acc2)[mi >> 2u][mi & 3u] + dot(a, qv2); }
      }
    }
  }
}

// Sum a per-lane partial across the LANES lanes of each row group; every lane
// of the group receives the total.
fn reduce(v: vec4<f32>, tid: u32, lane: u32) -> vec4<f32> {
  red[tid] = v;
  workgroupBarrier();
  for (var st = LANES / 2u; st > 0u; st = st >> 1u) {
    if (lane < st) { red[tid] = red[tid] + red[tid + st]; }
    workgroupBarrier();
  }
  let r = red[tid - lane];
  workgroupBarrier();
  return r;
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let outs = WG / LANES;                // output rows per workgroup
  let tid  = lid.x;
  let g    = tid / LANES;               // output row within the workgroup
  let lane = tid % LANES;
  let n    = wid.x * outs + g;
  let nc   = min(n, d.N - 1u);          // clamp keeps control flow uniform
  let m0   = wid.y * MROWS;
  let mLast = d.M - 1u;
  let rowBase = nc * d.wordsPerRow;
  let k4row = d.K / 4u;
  var xm = 0u;                          // top-bit flip for offset-binary sub-byte values
  if (BITS == 4u) { xm = 0x88888888u; }
  if (BITS == 2u) { xm = 0xAAAAAAAAu; }

  var acc  : array<vec4<f32>, 2>;       // up to 8 activation rows
  var acc2 : array<vec4<f32>, 2>;
  let nw = d.wordsPerRow;
  var w = lane;
  // 4 words per iteration, loads first: independent requests overlap
  for (; w + 3u * LANES < nw; w = w + 4u * LANES) {
    let w0 = Wq[rowBase + w] ^ xm;
    let w1 = Wq[rowBase + w + LANES] ^ xm;
    let w2 = Wq[rowBase + w + 2u * LANES] ^ xm;
    let w3 = Wq[rowBase + w + 3u * LANES] ^ xm;
    var u0 = 0u; var u1 = 0u; var u2 = 0u; var u3 = 0u;
    if (MODE == 1u) {
      u0 = Wq2[rowBase + w] ^ xm;
      u1 = Wq2[rowBase + w + LANES] ^ xm;
      u2 = Wq2[rowBase + w + 2u * LANES] ^ xm;
      u3 = Wq2[rowBase + w + 3u * LANES] ^ xm;
    }
    mac_word(w0, u0, w, m0, mLast, k4row, &acc, &acc2);
    mac_word(w1, u1, w + LANES, m0, mLast, k4row, &acc, &acc2);
    mac_word(w2, u2, w + 2u * LANES, m0, mLast, k4row, &acc, &acc2);
    mac_word(w3, u3, w + 3u * LANES, m0, mLast, k4row, &acc, &acc2);
  }
  for (; w < nw; w = w + LANES) {
    let w0 = Wq[rowBase + w] ^ xm;
    var u0 = 0u;
    if (MODE == 1u) { u0 = Wq2[rowBase + w] ^ xm; }
    mac_word(w0, u0, w, m0, mLast, k4row, &acc, &acc2);
  }

  var r0 = reduce(acc[0], tid, lane);
  var r1 = vec4<f32>(0.0);
  var s0 = vec4<f32>(0.0);
  var s1 = vec4<f32>(0.0);
  if (MROWS > 4u) { r1 = reduce(acc[1], tid, lane); }
  if (MODE == 1u) {
    s0 = reduce(acc2[0], tid, lane);
    if (MROWS > 4u) { s1 = reduce(acc2[1], tid, lane); }
  }

  if (lane == 0u && n < d.N) {
    let sn = scale[n];
    var sn2 = 0.0;
    if (MODE == 1u) { sn2 = scale2[n]; }
    for (var mi = 0u; mi < MROWS; mi = mi + 1u) {
      let m = m0 + mi;
      if (m < d.M) {
        var v: f32;
        var v2: f32;
        if (mi < 4u) { v = r0[mi] * sn; v2 = s0[mi] * sn2; } else { v = r1[mi - 4u] * sn; v2 = s1[mi - 4u] * sn2; }
        var o = v;
        if (MODE == 1u) { o = gelu(v) * v2; }
        if (MODE == 2u) { o = gelu(v) * Bs[m * d.bStride + d.bOff + n]; }
        C[m * d.N + n] = o;
      }
    }
  }
}
