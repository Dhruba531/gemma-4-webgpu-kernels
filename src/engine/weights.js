// Weight provider for the real Gemma-4 QAT checkpoint.
//
// The model asks for logical names like "language_model.layers.0.self_attn.q_proj.weight";
// this resolves them to the on-disk tensors and returns either:
//   • an f32 tensor (norms / projections stored as BF16/F32), or
//   • a *quantized* tensor kept packed on the GPU (U8 2/4-bit or I8 8-bit) plus
//     its f32 scale — dequantized on the fly inside the matmul/embed kernels.
//
// On-disk layout (from the checkpoint header):
//   model.language_model.<...>.weight            U8/I8  + .weight_scale
//   model.language_model.embed_tokens.embedding_quantized + .embedding_scale
//   lm_head.weight (U8) + lm_head.weight_scale
//   norms / per_layer_model_projection           BF16
// Bit widths come from config.quantization_config (config.bitsFor).

/**
 * Map a logical weight name to its on-disk tensor names.
 * `bare` covers tensors stored without a ".weight" suffix (e.g. layer_scalar).
 * Shared by ModelWeights (whole-file path) and StreamedWeights (streaming path).
 */
export function resolveTensorNames(logical) {
  const base = logical.replace(/\.weight$/, "");
  // lm_head has no "model." prefix on disk
  if (base === "lm_head") {
    return { weight: "lm_head.weight", scale: "lm_head.weight_scale" };
  }
  const real = "model." + base;
  if (base.endsWith("embed_tokens") || base.endsWith("embed_tokens_per_layer")) {
    return { weight: real + ".embedding_quantized", scale: real + ".embedding_scale" };
  }
  return { weight: real + ".weight", scale: real + ".weight_scale", bare: real };
}

export class ModelWeights {
  constructor(files, config, backend) {
    this.files = Array.isArray(files) ? files : [files];
    this.config = config;
    this.B = backend;
    this.cache = new Map();
    this.index = new Map();
    this.files.forEach((f, fi) => f.names().forEach((n) => this.index.set(n, fi)));
  }

  _file(name) {
    const fi = this.index.get(name);
    return fi == null ? null : this.files[fi];
  }

  get(logical) {
    if (this.cache.has(logical)) return this.cache.get(logical);
    let { weight, scale, bare } = resolveTensorNames(logical);
    let file = this._file(weight);
    if (!file && bare) {
      file = this._file(bare);
      if (file) weight = bare;
    }
    if (!file) return null;

    const meta = file.meta(weight);
    const base = logical.replace(/\.weight$/, "");
    let tensor;

    if (meta.dtype === "U8" || meta.dtype === "I8") {
      const signedI8 = meta.dtype === "I8";
      const bits = signedI8 ? 8 : this.config.bitsFor(base);
      const perByte = signedI8 ? 1 : 8 / bits;
      const N = meta.shape[0];
      const storedCols = meta.shape[1];
      const K = storedCols * perByte; // logical in-features
      const scaleMeta = file.meta(scale);
      const scaleCols = scaleMeta.shape[1] ?? 1;
      const groupSize = K / scaleCols;
      tensor = this.B.quantTensor({
        qbytes: file.rawBytes(weight),
        scales: file.tensor(scale).data,
        shape: [N, K],
        bits,
        signedI8: signedI8 ? 1 : 0,
        groupSize,
      });
    } else {
      // BF16 / F16 / F32 -> f32
      const t = file.tensor(weight);
      tensor = this.B.tensor(t.data, t.shape);
    }

    this.cache.set(logical, tensor);
    return tensor;
  }
}
