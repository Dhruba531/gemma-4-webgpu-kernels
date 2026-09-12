struct P { rows: u32, stride: u32, offset: u32, width: u32 };
@group(0) @binding(0) var<storage, read>       inp  : array<f32>;
@group(0) @binding(1) var<storage, read_write> outp : array<f32>;
@group(0) @binding(2) var<uniform>             p    : P;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= p.rows * p.width) { return; }
  let r = idx / p.width;
  let c = idx % p.width;
  outp[idx] = inp[r * p.stride + p.offset + c];
}
