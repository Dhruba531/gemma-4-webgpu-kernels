struct P {
  T: u32, S: u32, Hq: u32, Hkv: u32,
  Dh: u32, window: u32, _pad0: u32, _pad1: u32,
  scale: f32, _pad2: f32, _pad3: f32, _pad4: f32,
};
@group(0) @binding(0) var<storage, read>       q    : array<f32>;
@group(0) @binding(1) var<storage, read>       k    : array<f32>;
@group(0) @binding(2) var<storage, read>       v    : array<f32>;
@group(0) @binding(3) var<storage, read>       qpos : array<u32>;
@group(0) @binding(4) var<storage, read>       kpos : array<u32>;
@group(0) @binding(5) var<storage, read_write> outp : array<f32>;
@group(0) @binding(6) var<uniform>             p    : P;

const WG   : u32 = 128u;
const TILE : u32 = 128u;
const NEG  : f32 = -3.0e38;
var<workgroup> sc  : array<f32, 128>;   // tile scores, then tile probabilities
var<workgroup> red : array<f32, 128>;
var<workgroup> acc : array<f32, 512>;   // running output, <= max headDim
var<workgroup> m_s : f32;               // running max
var<workgroup> l_s : f32;               // running denom
var<workgroup> skip_s : u32;

@compute @workgroup_size(128, 1, 1)
fn main(@builtin(workgroup_id) wid : vec3<u32>,
        @builtin(local_invocation_id) lid : vec3<u32>) {
  let i = wid.x;                 // query index
  let h = wid.y;                 // head
  let group = p.Hq / p.Hkv;
  let kvh = h / group;
  let tid = lid.x;

  for (var d = tid; d < p.Dh; d = d + WG) { acc[d] = 0.0; }
  if (tid == 0u) { m_s = NEG; l_s = 0.0; }
  workgroupBarrier();

  let qBase = (i * p.Hq + h) * p.Dh;
  let qp = qpos[i];

  for (var j0 = 0u; j0 < p.S; j0 = j0 + TILE) {
    let jEnd = min(j0 + TILE, p.S);

    // tile-level skip decisions (uniform via workgroupUniformLoad)
    if (tid == 0u) {
      var v_ = 0u;
      if (kpos[j0] > qp) { v_ = 2u; }                     // all future: done
      else if (p.window > 0u && qp >= p.window && kpos[jEnd - 1u] <= qp - p.window) {
        v_ = 1u;                                          // all pre-window
      }
      skip_s = v_;
    }
    let sk = workgroupUniformLoad(&skip_s);
    if (sk == 2u) { break; }
    if (sk == 1u) { continue; }

    // 1) one score per thread
    let j = j0 + tid;
    var s = NEG;
    var valid = false;
    if (j < jEnd) {
      let kp = kpos[j];
      let masked = (kp > qp) || (p.window > 0u && (qp - kp) >= p.window);
      if (!masked) {
        let kBase = (j * p.Hkv + kvh) * p.Dh;
        var dot = 0.0;
        for (var d = 0u; d < p.Dh; d = d + 1u) { dot = dot + q[qBase + d] * k[kBase + d]; }
        s = dot * p.scale;
        valid = true;
      }
    }
    red[tid] = s;
    workgroupBarrier();
    for (var st = WG / 2u; st > 0u; st = st >> 1u) {
      if (tid < st) { red[tid] = max(red[tid], red[tid + st]); }
      workgroupBarrier();
    }
    let mOld = m_s;
    let mNew = max(mOld, red[0]);
    let corr = exp(mOld - mNew);
    let pv = select(0.0, exp(s - mNew), valid);
    workgroupBarrier(); // red/m_s reads above must finish before reuse below

    // 2) tile exp-sum
    sc[tid] = pv;
    red[tid] = pv;
    workgroupBarrier();
    for (var st = WG / 2u; st > 0u; st = st >> 1u) {
      if (tid < st) { red[tid] = red[tid] + red[tid + st]; }
      workgroupBarrier();
    }

    // 3) rescale accumulator and fold in the tile's V rows
    let tLen = jEnd - j0;
    for (var d = tid; d < p.Dh; d = d + WG) {
      var a = acc[d] * corr;
      for (var t = 0u; t < tLen; t = t + 1u) {
        a = a + sc[t] * v[((j0 + t) * p.Hkv + kvh) * p.Dh + d];
      }
      acc[d] = a;
    }
    if (tid == 0u) { l_s = l_s * corr + red[0]; m_s = mNew; }
    workgroupBarrier();
  }

  let oBase = (i * p.Hq + h) * p.Dh;
  let inv = select(0.0, 1.0 / l_s, l_s > 0.0);
  for (var d = tid; d < p.Dh; d = d + WG) { outp[oBase + d] = acc[d] * inv; }
}
