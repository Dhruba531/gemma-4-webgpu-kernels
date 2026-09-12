struct Dims { M:u32, K:u32, N:u32, bits:u32, perByte:u32, signedI8:u32, _pad0:u32, _pad1:u32 };
@group(0) @binding(0) var<storage, read>       A     : array<f32>;
@group(0) @binding(1) var<storage, read>       Wq    : array<u32>; // packed bytes
@group(0) @binding(2) var<storage, read>       scale : array<f32>; // [N]
@group(0) @binding(3) var<storage, read_write> C     : array<f32>;
@group(0) @binding(4) var<uniform>             d     : Dims;

fn getByte(i: u32) -> u32 { return (Wq[i >> 2u] >> (8u * (i & 3u))) & 0xffu; }

@compute @workgroup_size(64, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let n = gid.x;
  let m = gid.y;
  if (n >= d.N || m >= d.M) { return; }
  let s = scale[n];
  let mask = (1u << d.bits) - 1u;
  let rowBytes = select(d.K / d.perByte, d.K, d.signedI8 == 1u);
  let rowBase = n * rowBytes;
  let aBase = m * d.K;
  var acc = 0.0;
  for (var k = 0u; k < d.K; k = k + 1u) {
    var qv: i32;
    if (d.signedI8 == 1u) {
      let b = getByte(rowBase + k);
      qv = select(i32(b), i32(b) - 256, b > 127u); // sign-extend int8
    } else {
      let b = getByte(rowBase + k / d.perByte);
      let shift = (k % d.perByte) * d.bits;
      // offset binary: value = raw - zeroPoint, zeroPoint = 2^(bits-1)
      qv = i32((b >> shift) & mask) - (1 << (d.bits - 1u));
    }
    acc = acc + A[aBase + k] * (f32(qv) * s);
  }
  C[m * d.N + n] = acc;
}
