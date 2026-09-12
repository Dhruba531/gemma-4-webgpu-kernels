//! KV cache.
//!
//! Holds per-layer key/value tensors and the absolute position of every cached
//! token. Layers that share KV (Gemma-4 reuses one physical cache across
//! `num_kv_shared_layers` layers) point at the same slot, so we only store the
//! cache for *source* layers and let consumers read through.
//!
//! Storage grows through the backend's `kv_append`: a capacity buffer that
//! doubles when full, with new rows copied in place — O(new rows) per step
//! instead of re-copying the whole history. When growth retires an old buffer
//! the backend reports it as `dead`, and the model frees it at the end of the
//! forward pass.

use crate::backend::Backend;

pub struct KvSlot<T> {
    pub k: T,
    pub v: T,
}

pub struct KVCache<B: Backend> {
    pub slots: Vec<Option<KvSlot<B::Tensor>>>,
    /// Absolute positions of cached tokens.
    pub positions: Vec<u32>,
    /// Superseded k/v tensors awaiting a safe free point.
    dead: Vec<B::Tensor>,
}

impl<B: Backend> KVCache<B> {
    pub fn new(num_layers: usize) -> Self {
        Self { slots: (0..num_layers).map(|_| None).collect(), positions: Vec::new(), dead: Vec::new() }
    }

    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Append this step's positions once per generation step.
    pub fn push_positions(&mut self, positions: &[u32]) {
        self.positions.extend_from_slice(positions);
    }

    /// Append newly computed K/V for the source layer and return the full
    /// cached K/V (old ++ new) for attention to read.
    pub fn append_and_read(&mut self, backend: &B, src_layer: usize, k_new: &B::Tensor, v_new: &B::Tensor) -> (B::Tensor, B::Tensor) {
        let slot = self.slots[src_layer].as_ref();
        let rk = backend.kv_append(slot.map(|s| &s.k), k_new);
        let rv = backend.kv_append(slot.map(|s| &s.v), v_new);
        if let Some(d) = rk.dead {
            self.dead.push(d);
        }
        if let Some(d) = rv.dead {
            self.dead.push(d);
        }
        let (k, v) = (rk.tensor.clone(), rv.tensor.clone());
        self.slots[src_layer] = Some(KvSlot { k: rk.tensor, v: rv.tensor });
        (k, v)
    }

    /// Read the existing cache for a layer that shares (does not write) KV.
    pub fn read(&self, src_layer: usize) -> Option<&KvSlot<B::Tensor>> {
        self.slots[src_layer].as_ref()
    }

    /// All k/v tensors that must survive this forward pass.
    pub fn live_tensors(&self) -> Vec<B::Tensor> {
        self.slots.iter().flatten().flat_map(|s| [s.k.clone(), s.v.clone()]).collect()
    }

    /// Hand over the superseded tensors for freeing and forget them.
    pub fn take_dead(&mut self) -> Vec<B::Tensor> {
        std::mem::take(&mut self.dead)
    }

    pub fn reset(&mut self, backend: &B) {
        for s in self.slots.iter_mut() {
            if let Some(s) = s.take() {
                backend.free_tensor(s.k);
                backend.free_tensor(s.v);
            }
        }
        for t in self.dead.drain(..) {
            backend.free_tensor(t);
        }
        self.positions.clear();
    }
}
