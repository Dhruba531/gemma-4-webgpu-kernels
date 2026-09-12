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
