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
