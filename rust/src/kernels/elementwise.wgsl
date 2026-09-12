struct P { n: u32, op: u32, k: f32, _pad: u32 };
@group(0) @binding(0) var<storage, read>       a : array<f32>;
@group(0) @binding(1) var<storage, read>       b : array<f32>;
@group(0) @binding(2) var<storage, read_write> o : array<f32>;
@group(0) @binding(3) var<uniform>             p : P;

// tanh(x) overflows to NaN for large |x| on some backends (e^x = inf), because
// the cubic term in gelu can push the argument into the thousands. Clamp first;
// tanh saturates to ±1 well before ±20.
fn tanh_safe(x: f32) -> f32 { return tanh(clamp(x, -20.0, 20.0)); }

fn gelu(v: f32) -> f32 {
  let c = 0.7978845608028654; // sqrt(2/pi)
  return 0.5 * v * (1.0 + tanh_safe(c * (v + 0.044715 * v * v * v)));
}

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  switch p.op {
    case 0u: { o[i] = a[i] + b[i]; }
    case 1u: { o[i] = a[i] * b[i]; }
    case 2u: { o[i] = a[i] * p.k; }
    case 3u: { o[i] = gelu(a[i]); }
    case 4u: { o[i] = gelu(a[i]) * b[i]; }
    case 5u: { o[i] = p.k * tanh_safe(a[i] / p.k); }
    case 6u: { o[i] = a[i] * b[0]; }
    default: { o[i] = a[i]; }
  }
}
