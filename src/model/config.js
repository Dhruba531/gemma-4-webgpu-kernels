// Gemma-4 configuration parsing.
//
// The HF `config.json` for google/gemma-4-E2B-it-qat-mobile-transformers is a
// multimodal config (text + vision + audio). This engine implements the *text*
// path only, so we read `text_config` and normalise it into the flat shape the
// rest of the runtime consumes.

/** @typedef {"sliding_attention"|"full_attention"} LayerType */

const DEFAULTS = {
  vocab_size: 262144,
  hidden_size: 1536,
  intermediate_size: 6144,
  num_hidden_layers: 35,
  num_attention_heads: 8,
  num_key_value_heads: 1,
  head_dim: 256,
  global_head_dim: 512,
  rms_norm_eps: 1e-6,
  hidden_activation: "gelu_pytorch_tanh",
  sliding_window: 512,
  final_logit_softcapping: 30.0,
  // Per-Layer Embeddings (PLE)
  hidden_size_per_layer_input: 256,
  vocab_size_per_layer_input: 262144,
  // KV cache sharing: the last `num_kv_shared_layers` reuse KV from the layer
  // `num_kv_shared_layers` positions earlier (Gemma-3n style).
  num_kv_shared_layers: 20,
  use_double_wide_mlp: true,
  tie_word_embeddings: false,
  bos_token_id: 2,
  eos_token_id: [1, 106],
  pad_token_id: 0,
};

export class Gemma4Config {
  constructor(raw = {}) {
    // Accept either a full multimodal config or a bare text_config.
    const t = raw.text_config ?? raw;
    Object.assign(this, DEFAULTS, t);

    // eos can be a scalar or list; always expose an array.
    this.eos_token_ids = Array.isArray(this.eos_token_id)
      ? this.eos_token_id
      : [this.eos_token_id];

    // Per-layer attention type. If the model omits layer_types, fall back to
    // "every 5th layer is full attention" which is the Gemma-3 pattern.
    if (!Array.isArray(this.layer_types) || this.layer_types.length !== this.num_hidden_layers) {
      this.layer_types = Array.from({ length: this.num_hidden_layers }, (_, i) =>
        (i + 1) % 5 === 0 ? "full_attention" : "sliding_attention"
      );
    }

    // RoPE parameters per attention class. The text_config nests these under
    // `rope_parameters.{full_attention,sliding_attention}`.
    const rp = t.rope_parameters ?? {};
    this.rope = {
      full_attention: {
        theta: rp.full_attention?.rope_theta ?? 1_000_000,
        // partial_rotary_factor < 1 means only a prefix of head_dim is rotated.
        partial_rotary_factor: rp.full_attention?.partial_rotary_factor ?? 1.0,
      },
      sliding_attention: {
        theta: rp.sliding_attention?.rope_theta ?? 10_000,
        partial_rotary_factor: rp.sliding_attention?.partial_rotary_factor ?? 1.0,
      },
    };

    // Mixed-precision QAT: map a weight name -> bit width via the regex table.
    const qc = raw.quantization_config?.module_quant_configs ?? {};
    this._quantRules = Object.entries(qc).map(([pattern, cfg]) => ({
      re: new RegExp(pattern),
      bits: cfg.num_bits,
    }));
  }

  layerType(i) {
    return this.layer_types[i];
  }

  isGlobal(i) {
    return this.layer_types[i] === "full_attention";
  }

  /** head_dim depends on whether the layer is global or local. */
  headDim(i) {
    return this.isGlobal(i) ? this.global_head_dim : this.head_dim;
  }

  ropeFor(i) {
    return this.isGlobal(i) ? this.rope.full_attention : this.rope.sliding_attention;
  }

  /**
   * Which physical layer owns the KV cache that layer `i` reads from.
   *
   * The last `num_kv_shared_layers` layers don't compute their own KV; they
   * reuse the KV of the most recent earlier layer of the SAME attention type
   * that lives in the non-shared region. Matching the type is mandatory — it
   * guarantees the cached K/V have the same head_dim and RoPE base as the query.
   */
  kvSourceLayer(i) {
    const firstShared = this.num_hidden_layers - this.num_kv_shared_layers;
    if (i < firstShared) return i;
    const type = this.layer_types[i];
    for (let j = firstShared - 1; j >= 0; j--) {
      if (this.layer_types[j] === type) return j;
    }
    return i; // no same-type source: fall back to computing own KV
  }

  /** Quantized bit width for a weight name, or 16 for unquantized (bf16/f32). */
  bitsFor(name) {
    for (const rule of this._quantRules) {
      if (rule.re.test(name)) return rule.bits;
    }
    return 16;
  }
}
