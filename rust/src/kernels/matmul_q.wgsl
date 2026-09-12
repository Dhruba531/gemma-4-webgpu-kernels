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
