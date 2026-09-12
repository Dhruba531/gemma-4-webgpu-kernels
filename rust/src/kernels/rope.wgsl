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
