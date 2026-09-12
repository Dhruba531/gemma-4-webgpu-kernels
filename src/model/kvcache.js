// KV cache.
//
// Holds per-layer key/value tensors and the absolute position of every cached
// token. Layers that share KV (Gemma-4 reuses one physical cache across
// `num_kv_shared_layers` layers) point at the same slot, so we only store the
// cache for *source* layers and let consumers read through.
//
// Storage grows through the backend's kvAppend: a capacity buffer that doubles
// when full, with new rows copied in place — O(new rows) per step instead of
// re-copying the whole history. When growth retires an old buffer the backend
// reports it in `dead`, and the model frees it at the end of the forward pass
// (see WebGpuBackend.endFrame).

export class KVCache {
  constructor(config, backend) {
    this.config = config;
    this.B = backend;
    /** @type {Array<{k:any,v:any}|null>} indexed by source-layer id */
    this.slots = new Array(config.num_hidden_layers).fill(null);
    this.positions = []; // absolute positions of cached tokens
    this.dead = []; // superseded k/v tensors awaiting a safe free point
  }

  get length() {
    return this.positions.length;
  }

  /** Append this step's positions once per generation step. */
  pushPositions(positions) {
    for (const p of positions) this.positions.push(p);
  }

  /**
   * Append newly computed K/V for the source layer and return the full cached
   * K/V (old ++ new) for attention to read.
   */
  appendAndRead(srcLayer, kNew, vNew) {
    const slot = this.slots[srcLayer];
    const rk = this.B.kvAppend(slot?.k ?? null, kNew);
    const rv = this.B.kvAppend(slot?.v ?? null, vNew);
    if (this.B.freeTensor) {
      if (rk.dead) this.dead.push(rk.dead);
      if (rv.dead) this.dead.push(rv.dead);
    }
    this.slots[srcLayer] = { k: rk.tensor, v: rv.tensor };
    return { k: rk.tensor, v: rv.tensor };
  }

  /** Read the existing cache for a layer that shares (does not write) KV. */
  read(srcLayer) {
    return this.slots[srcLayer];
  }

  /** All k/v tensors that must survive this forward pass. */
  liveTensors() {
    return this.slots.flatMap((s) => (s ? [s.k, s.v] : []));
  }

  /** Hand over the superseded tensors for freeing and forget them. */
  takeDead() {
    const dead = this.dead;
    this.dead = [];
    return dead;
  }

  reset() {
    for (const s of this.slots) {
      if (s) {
        this.B.freeTensor?.(s.k);
        this.B.freeTensor?.(s.v);
      }
    }
    for (const t of this.dead) this.B.freeTensor?.(t);
    this.slots.fill(null);
    this.dead = [];
    this.positions = [];
  }
}
