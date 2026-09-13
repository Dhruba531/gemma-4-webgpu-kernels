// Buffer-to-buffer copy as a compute dispatch:  out[dstOff + i] = in[srcOff + i].
// Used instead of copyBufferToBuffer inside a forward pass so KV-cache appends
// and row slices don't split the single compute pass a step records into.
struct P { n: u32, srcOff: u32, dstOff: u32, _p: u32 };
@group(0) @binding(0) var<storage, read>       inp  : array<f32>;
@group(0) @binding(1) var<storage, read_write> outp : array<f32>;
@group(0) @binding(2) var<uniform>             p    : P;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  outp[p.dstOff + i] = inp[p.srcOff + i];
}
