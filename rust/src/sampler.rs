//! Token sampling from a logits row (length = vocab).
//! Supports greedy, temperature, top-k, top-p (nucleus) and repetition penalty.

/// Small deterministic PRNG (splitmix64) so runs are reproducible with a seed.
#[derive(Clone, Debug)]
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed)
    }
    pub fn from_entropy() -> Self {
        let t = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_nanos() as u64).unwrap_or(0x9e3779b97f4a7c15);
        Self(t ^ (std::process::id() as u64).rotate_left(32))
    }
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    }
    /// Uniform in [0, 1).
    pub fn next_f64(&mut self) -> f64 {
        (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64
    }
}

#[derive(Clone, Debug)]
pub struct SampleOpts<'a> {
    pub temperature: f32,
    /// 0 disables.
    pub top_k: usize,
    /// >= 1.0 disables.
    pub top_p: f32,
    /// 1.0 disables.
    pub repetition_penalty: f32,
    pub recent_tokens: Option<&'a [u32]>,
}

impl Default for SampleOpts<'_> {
    fn default() -> Self {
        Self { temperature: 1.0, top_k: 0, top_p: 1.0, repetition_penalty: 1.0, recent_tokens: None }
    }
}

fn softmax_over(logits: &[f32], indices: &[usize], inv_temperature: f32) -> Vec<f32> {
    let max = indices.iter().map(|&i| logits[i]).fold(f32::NEG_INFINITY, f32::max);
    let mut probs: Vec<f32> = indices.iter().map(|&i| ((logits[i] - max) * inv_temperature).exp()).collect();
    let sum: f32 = probs.iter().sum();
    for p in probs.iter_mut() {
        *p /= sum;
    }
    probs
}

// Sample the full vocabulary without materializing an index array — the common
// unfiltered path.
fn sample_all(logits: &[f32], inv_temperature: f32, rng: &mut dyn FnMut() -> f64) -> u32 {
    let max = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let sum: f32 = logits.iter().map(|&l| ((l - max) * inv_temperature).exp()).sum();
    let mut r = rng() as f32 * sum;
    for (i, &l) in logits.iter().enumerate() {
        r -= ((l - max) * inv_temperature).exp();
        if r <= 0.0 {
            return i as u32;
        }
    }
    (logits.len() - 1) as u32
}

// Indices of the k largest values, sorted descending. A bounded min-heap keeps
// this O(V log k) — sorting the full 262k-entry vocab per token is what we're avoiding.
fn top_k_desc(values: &[f32], k: usize) -> Vec<usize> {
    let mut heap: Vec<usize> = Vec::with_capacity(k); // heap[0] holds the smallest value
    let sift_up = |heap: &mut Vec<usize>, mut i: usize| {
        while i > 0 {
            let parent = (i - 1) / 2;
            if values[heap[i]] >= values[heap[parent]] {
                break;
            }
            heap.swap(i, parent);
            i = parent;
        }
    };
    let sift_down = |heap: &mut Vec<usize>, mut i: usize| loop {
        let (l, r) = (2 * i + 1, 2 * i + 2);
        let mut m = i;
        if l < heap.len() && values[heap[l]] < values[heap[m]] {
            m = l;
        }
        if r < heap.len() && values[heap[r]] < values[heap[m]] {
            m = r;
        }
        if m == i {
            break;
        }
        heap.swap(i, m);
        i = m;
    };
    for i in 0..values.len() {
        if heap.len() < k {
            heap.push(i);
            let n = heap.len() - 1;
            sift_up(&mut heap, n);
        } else if values[i] > values[heap[0]] {
            heap[0] = i;
            sift_down(&mut heap, 0);
        }
    }
    heap.sort_by(|&a, &b| values[b].partial_cmp(&values[a]).unwrap_or(std::cmp::Ordering::Equal));
    heap
}

/// Pick the next token id from a logits row.
pub fn sample(logits: &[f32], opts: &SampleOpts, rng: &mut dyn FnMut() -> f64) -> u32 {
    let vocab = logits.len();

    // Repetition penalty (CTRL-style), applied to a copy so the caller's logits
    // are left untouched. Each vocabulary item is penalized once — repeated
    // occurrences in the history don't compound.
    let owned;
    let l: &[f32] = if opts.repetition_penalty != 1.0 && opts.recent_tokens.is_some() {
        let mut v = logits.to_vec();
        let mut seen = std::collections::HashSet::new();
        for &tok in opts.recent_tokens.unwrap() {
            if (tok as usize) < v.len() && seen.insert(tok) {
                let x = v[tok as usize];
                v[tok as usize] = if x > 0.0 { x / opts.repetition_penalty } else { x * opts.repetition_penalty };
            }
        }
        owned = v;
        &owned
    } else {
        logits
    };

    if opts.temperature <= 0.0 {
        // greedy
        let mut best = 0usize;
        let mut bv = f32::NEG_INFINITY;
        for (i, &x) in l.iter().enumerate() {
            if x > bv {
                bv = x;
                best = i;
            }
        }
        return best as u32;
    }

    let inv_t = 1.0 / opts.temperature;

    // Fast path: sampling from the full vocabulary needs no candidate list.
    if opts.top_k == 0 && opts.top_p >= 1.0 {
        return sample_all(l, inv_t, rng);
    }

    // candidate indices, sorted by logit desc — only materialized when filtering
    let mut indices: Vec<usize> = if opts.top_k > 0 {
        top_k_desc(l, opts.top_k.min(vocab))
    } else {
        let mut v: Vec<usize> = (0..vocab).collect();
        v.sort_by(|&a, &b| l[b].partial_cmp(&l[a]).unwrap_or(std::cmp::Ordering::Equal));
        v
    };
    if opts.top_p < 1.0 {
        let probs = softmax_over(l, &indices, inv_t);
        let mut cum = 0.0f32;
        let mut keep = Vec::new();
        for (n, &idx) in indices.iter().enumerate() {
            keep.push(idx);
            cum += probs[n];
            if cum >= opts.top_p {
                break;
            }
        }
        indices = keep;
    }
    let probs = softmax_over(l, &indices, inv_t);
    let mut r = rng() as f32;
    for (n, &idx) in indices.iter().enumerate() {
        r -= probs[n];
        if r <= 0.0 {
            return idx as u32;
        }
    }
    *indices.last().unwrap() as u32
}
