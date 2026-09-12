// WebGPU backend.
//
// Implements exactly the same op surface as CpuBackend, so model/gemma4.js runs
// unchanged. Tensors live in GPU storage buffers; ops enqueue compute passes and
// return a handle synchronously. Only readback() is async.
//
// f32 throughout.
//
// Batching: ops record into ONE shared command encoder (and, where possible, one
// open compute pass), submitted on flush() — which readback() triggers. A forward
// pass is therefore a single queue.submit instead of one per op (~900/token),
// which removes almost all CPU-side driver overhead.
//
// Memory lifecycle: the model brackets each forward pass with beginFrame()/
// endFrame(). Every intermediate activation alloc()'d inside the frame is
// recycled at endFrame() except the tensors the caller asks to keep (the live
// KV cache), so GPU memory stays bounded per step instead of growing with every
// generated token. Recycled activation buffers go to a size-bucketed pool and
// are reused by later frames (buffer creation is expensive in Dawn/Chrome).
// Weight uploads (tensor()/quantTensor*) are never frame-tracked — they live
// for the model's lifetime.
//
// Node/Dawn GC hazard: some WebGPU bindings let JS GC destroy a native object
// (buffer/bind group/command buffer) whose handle was dropped before the GPU
// consumed it. Everything created while recording — bind groups, command
// buffers, index buffers — is retained in `_holds`/`_transient` until endFrame,
// which only runs after an awaited readback (GPU provably idle).

import * as K from "./kernels.js";

export class GpuTensor {
  constructor(buffer, shape, backend) {
    this.buffer = buffer;
    this.shape = shape;
    this.size = shape.reduce((a, b) => a * b, 1);
    this._b = backend;
  }
}

// Computed lazily: in Node-WebGPU the GPU* globals are installed at runtime,
// after this module is first evaluated.
const storageUsage = () => GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST;

const POOL_MAX_BYTES = 128 << 20; // cap on recycled activation buffers

export class WebGpuBackend {
  constructor(device) {
    this.name = "webgpu";
    this.device = device;
    this.pipelines = new Map();
    // Transient uniform/index buffers, retained until a frame ends (see GC note).
    this._transient = [];
    // Buffers alloc()'d inside the current beginFrame()/endFrame() bracket,
    // or null when no frame is active (e.g. direct kernel tests).
    this._frame = null;
    // Shared command encoder + open compute pass for the current batch.
    this._enc = null;
    this._pass = null;
    // JS refs to bind groups / command buffers until the frame ends (GC note).
    this._holds = [];
    // Uniform ring: segments of one big UNIFORM buffer, suballocated per op at
    // 256-byte alignment, written once per flush.
    this._rings = [];
    this._uniAlign = 256;
    // Per-frame memo for _u32 uploads keyed by the source array object, so the
    // same positions array isn't re-uploaded for each of the 35 layers.
    this._u32memo = new Map();
    // Recycled activation buffers: pow2 byte size -> GPUBuffer[].
    this._pool = new Map();
    this._pooledBytes = 0;
  }
  async init() {}
  destroy() {
    this.device.destroy?.();
  }

  // --- shared encoder / batching --------------------------------------------
  _encoder() {
    if (!this._enc) this._enc = this.device.createCommandEncoder();
    return this._enc;
  }

  _computePass() {
    if (!this._pass) this._pass = this._encoder().beginComputePass();
    return this._pass;
  }

  _endPass() {
    if (this._pass) {
      this._pass.end();
      this._pass = null;
    }
  }

  // Record a buffer-to-buffer copy into the shared encoder (splits the pass).
  _copy(src, srcOff, dst, dstOff, bytes) {
    this._endPass();
    this._encoder().copyBufferToBuffer(src, srcOff, dst, dstOff, bytes);
  }

  /** Submit everything recorded so far (uniform ring writes go first). */
  flush() {
    this._endPass();
    if (!this._enc) return;
    for (const seg of this._rings) {
      if (seg.off > 0) this.device.queue.writeBuffer(seg.buf, 0, seg.data, 0, seg.off);
      seg.off = 0; // safe to reuse: queue operations execute in order
    }
    const cb = this._enc.finish();
    this._holds.push(cb);
    this.device.queue.submit([cb]);
    this._enc = null;
  }

  // --- per-forward memory frame -------------------------------------------
  beginFrame() {
    this._frame = [];
    this._u32memo.clear();
  }

  /**
   * Recycle everything allocated since beginFrame() except `keep`, plus the
   * `alsoFree` tensors (buffers from earlier frames the caller has retired,
   * e.g. KV tensors superseded by this step's growth).
   *
   * Only call this after an awaited readback(): mapAsync resolving means every
   * previously submitted command finished, so the GPU cannot still be using
   * any buffer this frame touched.
   */
  endFrame(keep = [], alsoFree = []) {
    this.flush();
    const keepBufs = new Set(keep.filter(Boolean).map((t) => t.buffer));
    for (const buf of this._frame ?? []) {
      if (!keepBufs.has(buf)) this._recycle(buf);
    }
    this._frame = null;
    for (const t of alsoFree) {
      if (t && !keepBufs.has(t.buffer)) t.buffer.destroy();
    }
    for (const buf of this._transient) buf.destroy();
    this._transient = [];
    this._holds = [];
    this._u32memo.clear();
  }

  /** Free a single tensor's buffer (caller guarantees the GPU is done with it). */
  freeTensor(t) {
    t?.buffer.destroy();
  }

  _recycle(buf) {
    if (!buf._poolSize || this._pooledBytes + buf._poolSize > POOL_MAX_BYTES) {
      buf.destroy();
      return;
    }
    let list = this._pool.get(buf._poolSize);
    if (!list) this._pool.set(buf._poolSize, (list = []));
    list.push(buf);
    this._pooledBytes += buf._poolSize;
  }

  // --- low level -----------------------------------------------------------
  pipeline(key, wgsl, constants) {
    let p = this.pipelines.get(key);
    if (!p) {
      const module = this.device.createShaderModule({ code: wgsl });
      p = this.device.createComputePipeline({
        layout: "auto",
        compute: constants ? { module, entryPoint: "main", constants } : { module, entryPoint: "main" },
      });
      this.pipelines.set(key, p);
    }
    return p;
  }

  _buf(byteLen, usage = storageUsage()) {
    return this.device.createBuffer({ size: Math.max(16, Math.ceil(byteLen / 4) * 4), usage });
  }

  // Activation allocation: pow2-bucketed so frames can recycle buffers.
  // Kernels never rely on buffer length (all use explicit dims) nor on
  // zero-initialization (every op fully writes its output), so a recycled
  // buffer with stale contents is fine.
  alloc(shape) {
    const n = shape.reduce((a, b) => a * b, 1);
    const want = Math.max(16, n * 4);
    let size = 256;
    while (size < want) size <<= 1;
    const list = this._pool.get(size);
    let buf;
    if (list && list.length > 0) {
      buf = list.pop();
      this._pooledBytes -= size;
    } else {
      buf = this._buf(size);
      buf._poolSize = size;
    }
    this._frame?.push(buf);
    return new GpuTensor(buf, shape, this);
  }

  // Retain a transient (uniform/index) buffer until the frame ends. Outside a
  // frame, cap the list so long kernel-test runs don't grow it unboundedly.
  _retainTransient(buf) {
    this._transient.push(buf);
    if (!this._frame && this._transient.length > 4096) this._transient.shift();
    return buf;
  }

  tensor(data, shape) {
    const f = data instanceof Float32Array ? data : Float32Array.from(data);
    const buf = this.device.createBuffer({
      size: Math.max(16, f.byteLength),
      usage: storageUsage(),
      mappedAtCreation: true,
    });
    new Float32Array(buf.getMappedRange(0, Math.max(16, f.byteLength))).set(f);
    buf.unmap();
    return new GpuTensor(buf, shape, this);
  }

  // upload raw bytes (packed quantized weights) as a storage buffer addressable
  // as array<u32> in WGSL.
  _bytesBuf(u8) {
    const padded = Math.ceil(u8.byteLength / 4) * 4;
    const buf = this.device.createBuffer({ size: Math.max(16, padded), usage: storageUsage(), mappedAtCreation: true });
    new Uint8Array(buf.getMappedRange(0, Math.max(16, padded))).set(u8);
    buf.unmap();
    return buf;
  }

  // Keep a weight quantized on the GPU. q = { qbytes, scales, shape:[N,K],
  // bits, signedI8, groupSize }.
  quantTensor(q) {
    return this.quantTensorFromGpu({ ...q, buffer: this._bytesBuf(q.qbytes) });
  }

  // Build a quant tensor from an already-uploaded GPU byte buffer (streaming
  // path) — only the small scale array still needs uploading here.
  quantTensorFromGpu(q) {
    const perByte = q.signedI8 ? 1 : 8 / q.bits;
    const tns = new GpuTensor(q.buffer, q.shape, this);
    tns.scaleBuffer = this.tensor(q.scales, [q.scales.length]).buffer;
    tns.quant = { ...q, perByte };
    return tns;
  }

  // upload a u32 index buffer (token ids / positions); memoized per frame by
  // the identity of the source array.
  _u32(arr) {
    const memoKey = typeof arr === "object" && arr !== null ? arr : null;
    if (memoKey && this._u32memo.has(memoKey)) return this._u32memo.get(memoKey);
    const u = arr instanceof Uint32Array ? arr : Uint32Array.from(arr);
    const buf = this.device.createBuffer({
      size: Math.max(16, u.byteLength),
      usage: storageUsage(),
      mappedAtCreation: true,
    });
    new Uint32Array(buf.getMappedRange(0, Math.max(16, u.byteLength))).set(u);
    buf.unmap();
    this._retainTransient(buf);
    if (memoKey) this._u32memo.set(memoKey, buf);
    return buf;
  }

  // Suballocate a uniform slot from the ring; contents are staged CPU-side and
  // written in one queue.writeBuffer per segment at flush().
  _uniformSlot(bytes) {
    const need = Math.ceil(bytes.byteLength / this._uniAlign) * this._uniAlign;
    let seg = null;
    for (const s of this._rings) {
      if (s.off + need <= s.cap) { seg = s; break; }
    }
    if (!seg) {
      const cap = Math.max(need, 1 << 18);
      seg = {
        buf: this.device.createBuffer({ size: cap, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST }),
        data: new Uint8Array(cap),
        off: 0,
        cap,
      };
      this._rings.push(seg);
    }
    seg.data.set(new Uint8Array(bytes), seg.off);
    const slot = { buffer: seg.buf, offset: seg.off, size: bytes.byteLength };
    seg.off += need;
    return slot;
  }

  _dispatch(key, wgsl, storageBufs, uniformBytes, workgroups, constants) {
    const pipeline = this.pipeline(key, wgsl, constants);
    const entries = storageBufs.map((b, i) => ({ binding: i, resource: { buffer: b } }));
    entries.push({ binding: storageBufs.length, resource: this._uniformSlot(uniformBytes) });
    const bg = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0), entries });
    this._holds.push(bg);
    if (!this._frame && this._holds.length > 8192) this._holds.shift();
    const pass = this._computePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(...workgroups);
  }

  async readback(t) {
    const bytes = t.size * 4;
    const staging = this.device.createBuffer({ size: Math.max(16, bytes), usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    this._copy(t.buffer, 0, staging, 0, bytes);
    this.flush();
    await staging.mapAsync(GPUMapMode.READ, 0, bytes);
    const out = new Float32Array(staging.getMappedRange(0, bytes).slice(0));
    staging.unmap();
    staging.destroy();
    return out;
  }

  // --- uniform encoders ----------------------------------------------------
  static _u32f32(values) {
    // values: [{u:n}|{f:x}, ...] -> ArrayBuffer (16-byte aligned)
    const padded = Math.ceil(values.length / 4) * 4;
    const buf = new ArrayBuffer(padded * 4);
    const dv = new DataView(buf);
    values.forEach((v, i) => {
      if ("u" in v) dv.setUint32(i * 4, v.u, true);
      else if ("i" in v) dv.setInt32(i * 4, v.i, true);
      else dv.setFloat32(i * 4, v.f, true);
    });
    return buf;
  }

  // --- ops -----------------------------------------------------------------
  zeros(shape) {
    const t = this.alloc(shape);
    // pooled buffers carry stale contents; clear explicitly
    this._endPass();
    this._encoder().clearBuffer(t.buffer, 0, t.size * 4);
    return t;
  }

  reshape(x, shape) {
    return new GpuTensor(x.buffer, shape, this);
  }

  embedRows(table, ids, scale = 1.0) {
    const D = table.shape[1];
    const T = ids.length;
    const out = this.alloc([T, D]);
    const idbuf = this._u32(ids);
    if (table.quant) {
      const q = table.quant;
      const sStride = Math.ceil(D / q.groupSize);
      const u = WebGpuBackend._u32f32([
        { u: T }, { u: D }, { u: q.bits }, { u: q.perByte },
        { u: q.groupSize }, { u: sStride }, { f: scale }, { u: 0 },
      ]);
      this._dispatch("embedq", K.EMBED_Q, [table.buffer, idbuf, table.scaleBuffer, out.buffer], u, [Math.ceil((T * D) / 256)]);
      return out;
    }
    const u = WebGpuBackend._u32f32([{ u: T }, { u: D }, { f: scale }, { u: 0 }]);
    this._dispatch("embed", K.EMBED, [table.buffer, idbuf, out.buffer], u, [Math.ceil((T * D) / 256)]);
    return out;
  }

  // weight = null runs the unweighted variant (the kernel still needs a valid
  // w binding of >= D elements, so bind x itself as a dummy).
  rmsNorm(x, weight, eps) {
    const [rows, D] = x.shape.length === 2 ? x.shape : [x.size / x.shape[x.shape.length - 1], x.shape[x.shape.length - 1]];
    const out = this.alloc(x.shape);
    const u = WebGpuBackend._u32f32([{ u: rows }, { u: D }, { f: eps }, { u: weight ? 1 : 0 }]);
    this._dispatch("rmsnorm", K.RMSNORM, [x.buffer, (weight ?? x).buffer, out.buffer], u, [rows]);
    return out;
  }

  linear(x, W) {
    const M = x.shape[0];
    const Kd = x.shape[1];
    const N = W.shape[0];
    const out = this.alloc([M, N]);
    if (W.quant) {
      const q = W.quant;
      const bits = q.signedI8 ? 8 : q.bits;
      const valsPerWord = 32 / bits;
      // Fast path: whole u32 words per lane, per-row scales. Covers every real
      // checkpoint shape; the scalar kernel remains for odd K / grouped scales.
      if (Kd % valsPerWord === 0 && q.groupSize === Kd && M <= 262140) {
        const mrows = M === 1 ? 1 : 4; // pipeline variant: see MROWS in the kernel
        const u = WebGpuBackend._u32f32([
          { u: M }, { u: Kd }, { u: N }, { u: bits },
          { u: Kd / valsPerWord }, { u: 0 }, { u: 0 }, { u: 0 },
        ]);
        this._dispatch(`matmulq_b${bits}m${mrows}`, K.MATMUL_Q, [x.buffer, W.buffer, W.scaleBuffer, out.buffer], u,
          [Math.ceil(N / 32), Math.ceil(M / mrows)], { MROWS: mrows, BITS: bits });
        return out;
      }
      const u = WebGpuBackend._u32f32([
        { u: M }, { u: Kd }, { u: N }, { u: q.bits },
        { u: q.perByte }, { u: q.signedI8 ? 1 : 0 }, { u: 0 }, { u: 0 },
      ]);
      this._dispatch("matmulq", K.MATMUL_Q_SCALAR, [x.buffer, W.buffer, W.scaleBuffer, out.buffer], u, [Math.ceil(N / 64), M]);
      return out;
    }
    if (M === 1 && N <= 65535) {
      // decode-time f32 matvec (e.g. per_layer_model_projection): one workgroup
      // per output row, lanes stride K — coalesced, no wasted tile threads.
      const u = WebGpuBackend._u32f32([{ u: Kd }, { u: N }, { u: 0 }, { u: 0 }]);
      this._dispatch("gemv", K.GEMV, [x.buffer, W.buffer, out.buffer], u, [N]);
      return out;
    }
    const u = WebGpuBackend._u32f32([{ u: M }, { u: Kd }, { u: N }, { u: 0 }]);
    this._dispatch("matmul", K.MATMUL, [x.buffer, W.buffer, out.buffer], u, [Math.ceil(N / 16), Math.ceil(M / 16)]);
    return out;
  }

  _ew(a, b, op, k = 0) {
    const out = this.alloc(a.shape);
    const u = WebGpuBackend._u32f32([{ u: a.size }, { u: op }, { f: k }, { u: 0 }]);
    this._dispatch("ew", K.ELEMENTWISE, [a.buffer, (b ?? a).buffer, out.buffer], u, [Math.ceil(a.size / 256)]);
    return out;
  }
  add(a, b) { return this._ew(a, b, 0); }
  mul(a, b) { return this._ew(a, b, 1); }
  scale(a, s) { return this._ew(a, null, 2, s); }
  geluTanh(a) { return this._ew(a, null, 3); }
  geglu(gate, up) { return this._ew(gate, up, 4); }
  softcap(a, cap) { return this._ew(a, null, 5, cap); }
  scaleByTensor(a, s) { return this._ew(a, s, 6); } // a * s[0] (layer_scalar)

  rope(x, positions, theta, headDim, heads, rotaryDim) {
    const T = x.shape[0];
    const out = this.alloc(x.shape);
    const posbuf = this._u32(positions);
    const u = WebGpuBackend._u32f32([
      { u: T }, { u: heads }, { u: headDim }, { u: rotaryDim },
      { f: theta }, { f: 0 }, { f: 0 }, { f: 0 },
    ]);
    this._dispatch("rope", K.ROPE, [x.buffer, posbuf, out.buffer], u, [Math.ceil((T * heads * headDim) / 256)]);
    return out;
  }

  attention(q, k, v, { qPos, kPos, scale, slidingWindow }) {
    const [T, Hq, Dh] = q.shape;
    const S = k.shape[0];
    const Hkv = k.shape[1];
    const out = this.alloc([T, Hq * Dh]);
    const qp = this._u32(qPos);
    const kp = this._u32(kPos);
    const u = WebGpuBackend._u32f32([
      { u: T }, { u: S }, { u: Hq }, { u: Hkv },
      { u: Dh }, { u: slidingWindow }, { u: 0 }, { u: 0 },
      { f: scale }, { f: 0 }, { f: 0 }, { f: 0 },
    ]);
    this._dispatch("attn", K.ATTENTION, [q.buffer, k.buffer, v.buffer, qp, kp, out.buffer], u, [T, Hq]);
    return out;
  }

  sliceRows(x, start, count) {
    const dim = x.shape[1];
    const out = this.alloc([count, dim]);
    this._copy(x.buffer, start * dim * 4, out.buffer, 0, count * dim * 4);
    return out;
  }

  sliceCols(x, offset, width) {
    const [rows, stride] = x.shape;
    const out = this.alloc([rows, width]);
    const u = WebGpuBackend._u32f32([{ u: rows }, { u: stride }, { u: offset }, { u: width }]);
    this._dispatch("slicecols", K.SLICE_COLS, [x.buffer, out.buffer], u, [Math.ceil((rows * width) / 256)]);
    return out;
  }

  concatRows(a, b) {
    if (!a) {
      const out = this.alloc(b.shape.slice());
      this._copy(b.buffer, 0, out.buffer, 0, b.size * 4);
      return out;
    }
    const out = this.alloc([a.shape[0] + b.shape[0], ...a.shape.slice(1)]);
    this._copy(a.buffer, 0, out.buffer, 0, a.size * 4);
    this._copy(b.buffer, 0, out.buffer, a.size * 4, b.size * 4);
    return out;
  }

  /**
   * Append `add` ([rows, ...]) to the KV tensor `existing` in place, growing a
   * capacity buffer geometrically. Returns { tensor, dead }: `tensor` is a view
   * with the new logical row count (sharing the capacity buffer), `dead` is the
   * retired old buffer's tensor when growth reallocated (caller frees it at a
   * safe point). Turns the per-step O(S) cache copy into O(new rows).
   */
  kvAppend(existing, add) {
    const rows = add.shape[0];
    const dim = add.size / rows;
    if (!existing) {
      const cap = Math.max(rows, 256);
      const buf = this._buf(cap * dim * 4);
      this._frame?.push(buf);
      const t = new GpuTensor(buf, add.shape.slice(), this);
      t.capRows = cap;
      this._copy(add.buffer, 0, buf, 0, add.size * 4);
      return { tensor: t, dead: null };
    }
    const len = existing.shape[0];
    const capRows = existing.capRows ?? len;
    let dst = existing;
    let dead = null;
    if (len + rows > capRows) {
      const cap = Math.max(capRows * 2, len + rows);
      const buf = this._buf(cap * dim * 4);
      this._frame?.push(buf);
      dst = new GpuTensor(buf, existing.shape.slice(), this);
      dst.capRows = cap;
      this._copy(existing.buffer, 0, buf, 0, len * dim * 4);
      dead = existing;
    }
    this._copy(add.buffer, 0, dst.buffer, len * dim * 4, add.size * 4);
    const view = new GpuTensor(dst.buffer, [len + rows, ...add.shape.slice(1)], this);
    view.capRows = dst.capRows;
    return { tensor: view, dead };
  }
}
