// Merge the per-split (m, l, acc) partials written by ATTN_SPLIT:
//   out = Σ_s acc_s·exp(m_s − m) / Σ_s l_s·exp(m_s − m),  m = max_s m_s
// One thread per output element.
struct P { T: u32, Hq: u32, splits: u32, Dh: u32 };
@group(0) @binding(0) var<storage, read>       part : array<f32>;
@group(0) @binding(1) var<storage, read_write> outp : array<f32>;
@group(0) @binding(2) var<uniform>             p    : P;

@compute @workgroup_size(256, 1, 1)
fn main(@builtin(global_invocation_id) gid : vec3<u32>) {
  let e = gid.x;
  let total = p.T * p.Hq * p.Dh;
  if (e >= total) { return; }
  let ih = e / p.Dh;
  let dd = e % p.Dh;
  let mlBase = p.T * p.Hq * p.splits * p.Dh;
  var m = -3.0e38;
  for (var s = 0u; s < p.splits; s = s + 1u) { m = max(m, part[mlBase + (ih * p.splits + s) * 2u]); }
  var l = 0.0;
  var a = 0.0;
  for (var s = 0u; s < p.splits; s = s + 1u) {
    let slot = ih * p.splits + s;
    let w = exp(part[mlBase + slot * 2u] - m);
    l = l + part[mlBase + slot * 2u + 1u] * w;
    a = a + part[slot * p.Dh + dd] * w;
  }
  outp[e] = select(0.0, a / l, l > 0.0);
}
