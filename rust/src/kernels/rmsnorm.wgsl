// RMSNorm, one workgroup per row:  y = x · rsqrt(mean(x²) + eps) · w
// flags: 1 = multiply by w (else unweighted), 2 = add the residual row first
// (y = res + norm), 4 = multiply the result by the scalar s[0] (layer_scalar).
// The residual add and per-layer scalar are fused here because every Gemma-4
// post-norm is immediately followed by them.
struct P { rows: u32, D: u32, eps: f32, flags: u32 };
@group(0) @binding(0) var<storage, read>       x   : array<f32>;
@group(0) @binding(1) var<storage, read>       w   : array<f32>;
@group(0) @binding(2) var<storage, read>       res : array<f32>;
@group(0) @binding(3) var<storage, read>       s   : array<f32>;
@group(0) @binding(4) var<storage, read_write> y   : array<f32>;
@group(0) @binding(5) var<uniform>             p   : P;

const WG : u32 = 256u;
var<workgroup> partial : array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let row = wid.x;
  let base = row * p.D;
  var ss = 0.0;
  for (var d = lid.x; d < p.D; d = d + WG) { let v = x[base + d]; ss = ss + v * v; }
  partial[lid.x] = ss;
  workgroupBarrier();
  for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
    if (lid.x < stride) { partial[lid.x] = partial[lid.x] + partial[lid.x + stride]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(partial[0] / f32(p.D) + p.eps);
  let hasW = (p.flags & 1u) != 0u;
  let addRes = (p.flags & 2u) != 0u;
  let mulS = (p.flags & 4u) != 0u;
  let sc = select(1.0, s[0], mulS);
  for (var d = lid.x; d < p.D; d = d + WG) {
    let wv = select(1.0, w[d], hasW);
    let r = select(0.0, res[base + d], addRes);
    y[base + d] = (r + x[base + d] * inv * wv) * sc;
  }
}
