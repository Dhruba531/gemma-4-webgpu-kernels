// Token sampling from a logits row (Float32Array, length = vocab).
// Supports greedy, temperature, top-k and top-p (nucleus).

function softmaxOver(logits, indices, invTemperature) {
  let max = -Infinity;
  for (const i of indices) if (logits[i] > max) max = logits[i];
  let sum = 0;
  const probs = new Float32Array(indices.length);
  for (let n = 0; n < indices.length; n++) {
    const e = Math.exp((logits[indices[n]] - max) * invTemperature);
    probs[n] = e;
    sum += e;
  }
  for (let n = 0; n < probs.length; n++) probs[n] /= sum;
  return probs;
}

// Sample the full vocabulary without materializing a [0..vocab) index array or
// a temperature-scaled copy of the logits. This is the common unfiltered path
// and avoids several megabytes of short-lived allocations per generated token.
function sampleAll(logits, invTemperature, rng) {
  let max = -Infinity;
  for (let i = 0; i < logits.length; i++) if (logits[i] > max) max = logits[i];
  let sum = 0;
  for (let i = 0; i < logits.length; i++) sum += Math.exp((logits[i] - max) * invTemperature);
  let r = rng() * sum;
  for (let i = 0; i < logits.length; i++) {
    r -= Math.exp((logits[i] - max) * invTemperature);
    if (r <= 0) return i;
  }
  return logits.length - 1;
}

// Indices of the k largest values, sorted descending. A bounded min-heap keeps
// this O(V log k) — sorting the full 262k-entry vocab per token is what we're
// avoiding.
function topKDesc(values, k) {
  const heap = new Int32Array(k); // indices; heap[0] holds the smallest value
  let size = 0;
  const siftUp = (i) => {
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (values[heap[i]] >= values[heap[parent]]) break;
      const t = heap[i]; heap[i] = heap[parent]; heap[parent] = t;
      i = parent;
    }
  };
  const siftDown = (i) => {
    for (;;) {
      const l = 2 * i + 1, r = l + 1;
      let m = i;
      if (l < size && values[heap[l]] < values[heap[m]]) m = l;
      if (r < size && values[heap[r]] < values[heap[m]]) m = r;
      if (m === i) break;
      const t = heap[i]; heap[i] = heap[m]; heap[m] = t;
      i = m;
    }
  };
  for (let i = 0; i < values.length; i++) {
    if (size < k) {
      heap[size] = i;
      siftUp(size++);
    } else if (values[i] > values[heap[0]]) {
      heap[0] = i;
      siftDown(0);
    }
  }
  return Array.from(heap.subarray(0, size)).sort((a, b) => values[b] - values[a]);
}

export function sample(logits, opts = {}) {
  const {
    temperature = 1.0,
    topK = 0,
    topP = 1.0,
    repetitionPenalty = 1.0,
    recentTokens = null,
    rng = Math.random,
  } = opts;

  const vocab = logits.length;

  // Repetition penalty (CTRL-style), applied to a copy so the caller's logits
  // are left untouched.
  let l = logits;
  if (repetitionPenalty !== 1.0 && recentTokens) {
    l = Float32Array.from(logits);
    // Penalize each vocabulary item once. Repeated occurrences in the history
    // should not compound the CTRL-style penalty.
    for (const tok of new Set(recentTokens)) {
      if (tok < 0 || tok >= l.length) continue;
      const v = l[tok];
      l[tok] = v > 0 ? v / repetitionPenalty : v * repetitionPenalty;
    }
  }

  if (temperature <= 0) {
    // greedy
    let best = 0, bv = -Infinity;
    for (let i = 0; i < vocab; i++) if (l[i] > bv) { bv = l[i]; best = i; }
    return best;
  }

  const invTemperature = 1 / temperature;

  // Fast path: sampling from the full vocabulary needs no candidate list.
  if (topK <= 0 && topP >= 1.0) return sampleAll(l, invTemperature, rng);

  // candidate indices, sorted by logit desc — only materialized when filtering
  let indices = null;
  if (topK > 0) {
    indices = topKDesc(l, Math.min(topK, vocab));
  } else if (topP < 1.0) {
    indices = Array.from({ length: vocab }, (_, i) => i).sort((a, b) => l[b] - l[a]);
  }
  if (indices && topP < 1.0) {
    const probs = softmaxOver(l, indices, invTemperature);
    let cum = 0;
    const keep = [];
    for (let n = 0; n < indices.length; n++) {
      keep.push(indices[n]);
      cum += probs[n];
      if (cum >= topP) break;
    }
    indices = keep;
  }
  const probs = softmaxOver(l, indices, invTemperature);
  let r = rng();
  for (let n = 0; n < indices.length; n++) {
    r -= probs[n];
    if (r <= 0) return indices[n];
  }
  return indices[indices.length - 1];
}
