struct P { T:u32, D:u32, bits:u32, perByte:u32, group:u32, scaleStride:u32, embScale:f32, _pad:u32 };
@group(0) @binding(0) var<storage, read>       Wq    : array<u32>;
@group(0) @binding(1) var<storage, read>       ids   : array<u32>;
@group(0) @binding(2) var<storage, read>       scale : array<f32>;
@group(0) @binding(3) var<storage, read_write> outp  : array<f32>;
@group(0) @binding(4) var<uniform>             p     : P;

fn getByte2(i: u32) -> u32 { return (Wq[i >> 2u] >> (8u * (i & 3u))) & 0xffu; }

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let idx = gid.x;
  if (idx >= p.T * p.D) { return; }
  let i = idx / p.D;
  let dd = idx % p.D;
  let id = ids[i];
  let rowBytes = p.D / p.perByte;
  let b = getByte2(id * rowBytes + dd / p.perByte);
  let shift = (dd % p.perByte) * p.bits;
  let mask = (1u << p.bits) - 1u;
  // offset binary: value = raw - zeroPoint, zeroPoint = 2^(bits-1)
  let qv = i32((b >> shift) & mask) - (1 << (p.bits - 1u));
  let s = scale[id * p.scaleStride + dd / p.group];
  outp[idx] = f32(qv) * s * p.embScale;
}
