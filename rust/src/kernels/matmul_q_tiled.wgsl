// Prefill GEMM  C[M,N] = A[M,K] · Wᵀ  with W packed 2/4/8-bit, per-row scales.
//
// Classic shared-memory tiling for M >= 16: a workgroup owns a 64×64 output
// tile and walks K in chunks of 32. Per chunk it stages the A tile (64 rows ×
// 32 k, stored k-major so four consecutive m are one vec4) and the W tile
// (dequantized once — 64 n × 32 k, n-major) in workgroup memory, then each of
// the 256 threads accumulates a 4×4 register block from one vec4 of A and one
// vec4 of W per k: 2 loads per 16 MACs, and every packed value is unpacked
// once per 64 activation rows instead of once per 8.
//
// MODE as in matmul_q.wgsl: 0 plain, 1 gelu(·)·(second weight), 2 gelu(·)·Bs slice.
struct Dims { M:u32, K:u32, N:u32, wordsPerRow:u32, bOff:u32, bStride:u32, _p0:u32, _p1:u32 };
@group(0) @binding(0) var<storage, read>       A      : array<vec4<f32>>; // [M, K/4]
@group(0) @binding(1) var<storage, read>       Wq     : array<u32>;
@group(0) @binding(2) var<storage, read>       scale  : array<f32>;
@group(0) @binding(3) var<storage, read>       Wq2    : array<u32>;       // MODE 1
@group(0) @binding(4) var<storage, read>       scale2 : array<f32>;       // MODE 1
@group(0) @binding(5) var<storage, read>       Bs     : array<f32>;       // MODE 2
@group(0) @binding(6) var<storage, read_write> C      : array<f32>;
@group(0) @binding(7) var<uniform>             d      : Dims;

override BITS : u32 = 4u;
override MODE : u32 = 0u;
const TM : u32 = 64u;
const TN : u32 = 64u;
const TK : u32 = 32u;
var<workgroup> As  : array<vec4<f32>, 512>;   // [TK][TM/4]
var<workgroup> Ws  : array<vec4<f32>, 512>;   // [TK][TN/4]
var<workgroup> Ws2 : array<vec4<f32>, 512>;   // MODE 1

fn tanh_safe(x: f32) -> f32 { return tanh(clamp(x, -20.0, 20.0)); }
fn gelu(v: f32) -> f32 {
  return 0.5 * v * (1.0 + tanh_safe(0.7978845608028654 * (v + 0.044715 * v * v * v)));
}

// Unpack value j of a (pre-XORed) packed word.
fn val(word: u32, j: u32) -> f32 { return f32(extractBits(bitcast<i32>(word), j * BITS, BITS)); }

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let tid = lid.x;
  let n0 = wid.x * TN;
  let m0 = wid.y * TM;
  let tm = tid / 16u;                   // 4-row block
  let tn = tid % 16u;                   // 4-col block
  let k4row = d.K / 4u;
  let vpw = 32u / BITS;
  let wordsPerChunk = TK / vpw;         // packed words per row per K chunk
  var xm = 0u;
  if (BITS == 4u) { xm = 0x88888888u; }
  if (BITS == 2u) { xm = 0xAAAAAAAAu; }

  var acc  : array<vec4<f32>, 4>;       // [i: m][j: n]
  var acc2 : array<vec4<f32>, 4>;

  for (var k0 = 0u; k0 < d.K; k0 = k0 + TK) {
    // A tile: thread stages a 4(m) x 4(k) block — four row loads, transposed in
    // registers, written as four whole vec4s (one per k). Whole-vector writes
    // keep every workgroup-memory store race-free.
    for (var e = tid; e < (TM / 4u) * (TK / 4u); e = e + 256u) {
      let mb = e / (TK / 4u);           // 4-row block
      let k4 = e % (TK / 4u);           // 4-k block
      var r : array<vec4<f32>, 4>;
      for (var i = 0u; i < 4u; i = i + 1u) {
        let m = m0 + mb * 4u + i;
        r[i] = vec4<f32>(0.0);
        if (m < d.M) { r[i] = A[m * k4row + k0 / 4u + k4]; }
      }
      let base = (k4 * 4u) * (TM / 4u) + mb;
      As[base]                   = vec4<f32>(r[0].x, r[1].x, r[2].x, r[3].x);
      As[base + TM / 4u]         = vec4<f32>(r[0].y, r[1].y, r[2].y, r[3].y);
      As[base + 2u * (TM / 4u)]  = vec4<f32>(r[0].z, r[1].z, r[2].z, r[3].z);
      As[base + 3u * (TM / 4u)]  = vec4<f32>(r[0].w, r[1].w, r[2].w, r[3].w);
    }
    // W tile: thread stages one packed word of four consecutive rows n..n+3,
    // emitting one whole vec4 (the four rows' values) per k.
    for (var e = tid; e < (TN / 4u) * wordsPerChunk; e = e + 256u) {
      let nb = e / wordsPerChunk;
      let wi = e % wordsPerChunk;
      var wds : array<u32, 4>;
      var wds2 : array<u32, 4>;
      for (var i = 0u; i < 4u; i = i + 1u) {
        let n = n0 + nb * 4u + i;
        wds[i] = 0u;                    // rows past N unpack to 0
        wds2[i] = 0u;
        if (n < d.N) {
          let widx = n * d.wordsPerRow + k0 / vpw + wi;
          wds[i] = Wq[widx] ^ xm;
          if (MODE == 1u) { wds2[i] = Wq2[widx] ^ xm; }
        }
      }
      for (var j = 0u; j < vpw; j = j + 1u) {
        let idx = (wi * vpw + j) * (TN / 4u) + nb;
        Ws[idx] = vec4<f32>(val(wds[0], j), val(wds[1], j), val(wds[2], j), val(wds[3], j));
        if (MODE == 1u) { Ws2[idx] = vec4<f32>(val(wds2[0], j), val(wds2[1], j), val(wds2[2], j), val(wds2[3], j)); }
      }
    }
    workgroupBarrier();

    for (var k = 0u; k < TK; k = k + 1u) {
      let a4 = As[k * (TM / 4u) + tm];
      let w4 = Ws[k * (TN / 4u) + tn];
      for (var i = 0u; i < 4u; i = i + 1u) { acc[i] = acc[i] + a4[i] * w4; }
      if (MODE == 1u) {
        let u4 = Ws2[k * (TN / 4u) + tn];
        for (var i = 0u; i < 4u; i = i + 1u) { acc2[i] = acc2[i] + a4[i] * u4; }
      }
    }
    workgroupBarrier();
  }

  for (var i = 0u; i < 4u; i = i + 1u) {
    let m = m0 + tm * 4u + i;
    if (m >= d.M) { continue; }
    for (var j = 0u; j < 4u; j = j + 1u) {
      let n = n0 + tn * 4u + j;
      if (n >= d.N) { continue; }
      let v = acc[i][j] * scale[n];
      var o = v;
      if (MODE == 1u) { o = gelu(v) * (acc2[i][j] * scale2[n]); }
      if (MODE == 2u) { o = gelu(v) * Bs[m * d.bStride + d.bOff + n]; }
      C[m * d.N + n] = o;
    }
  }
}
