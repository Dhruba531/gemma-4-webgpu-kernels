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
