struct P { T: u32, D: u32, scale: f32, _pad: u32 };
@group(0) @binding(0) var<storage, read>       table : array<f32>;
@group(0) @binding(1) var<storage, read>       ids   : array<u32>;
@group(0) @binding(2) var<storage, read_write> outp  : array<f32>;
@group(0) @binding(3) var<uniform>             p     : P;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= p.T * p.D) { return; }
  let i = idx / p.D;
  let dd = idx % p.D;
  outp[idx] = table[ids[i] * p.D + dd] * p.scale;
}
