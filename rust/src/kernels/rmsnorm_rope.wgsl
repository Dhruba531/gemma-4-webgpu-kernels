// Per-head RMSNorm fused with half-split partial RoPE, one workgroup per
// (token, head) row of headDim:  y = rope(x · rsqrt(mean(x²) + eps) · w).
// Gemma-4 normalizes q and k per head right before rotating them, so this
// replaces two dispatches (and one intermediate) per projection.
struct P { T: u32, heads: u32, headDim: u32, rotaryDim: u32, theta: f32, eps: f32, _p0: f32, _p1: f32 };
@group(0) @binding(0) var<storage, read>       x   : array<f32>;
@group(0) @binding(1) var<storage, read>       w   : array<f32>;
@group(0) @binding(2) var<storage, read>       pos : array<u32>;
@group(0) @binding(3) var<storage, read_write> y   : array<f32>;
@group(0) @binding(4) var<uniform>             p   : P;

const WG : u32 = 256u;
var<workgroup> partial : array<f32, 256>;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let row = wid.x;                      // flattened (t, head)
  let base = row * p.headDim;
  let ti = row / p.heads;
  var ss = 0.0;
  for (var d = lid.x; d < p.headDim; d = d + WG) { let v = x[base + d]; ss = ss + v * v; }
  partial[lid.x] = ss;
  workgroupBarrier();
  for (var stride = WG / 2u; stride > 0u; stride = stride / 2u) {
    if (lid.x < stride) { partial[lid.x] = partial[lid.x] + partial[lid.x + stride]; }
    workgroupBarrier();
  }
  let inv = inverseSqrt(partial[0] / f32(p.headDim) + p.eps);
  let half = p.rotaryDim / 2u;
  let posf = f32(pos[ti]);
  for (var d = lid.x; d < half; d = d + WG) {
    let a = x[base + d] * inv * w[d];
    let b = x[base + d + half] * inv * w[d + half];
    let freq = pow(p.theta, -2.0 * f32(d) / f32(p.rotaryDim));
    let ang = posf * freq;
    let c = cos(ang); let s = sin(ang);
    y[base + d] = a * c - b * s;
    y[base + d + half] = b * c + a * s;
  }
  for (var d = p.rotaryDim + lid.x; d < p.headDim; d = d + WG) {   // pass-through tail
    y[base + d] = x[base + d] * inv * w[d];
  }
}
