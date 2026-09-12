// QAT dequantization.
//
// Gemma-4 "qat-mobile" ships weights at mixed bit widths (2/4/8) — see the
// `quantization_config.module_quant_configs` regex table in config.json.
//
// Verified against the reference runtime (reference/gemma-4-e2b.js) and the
// real checkpoint header:
//   • sub-byte payloads are U8, packed LSB-first, OFFSET-BINARY:
//       value = (raw - 2^(bits-1)) * scale     (zero-point = 2^(bits-1))
//     (NOT two's-complement sign extension — raw 0 is the most negative value)
//   • 8-bit payloads are native signed I8: value = q * scale
//   • scales are per output row ([N,1]) except embed_tokens_per_layer, which
//     groups one scale per (row, layer) block of 256.

// Sign-extend an 8-bit value (native signed I8 payloads).
function signExtend8(v) {
  return (v << 24) >> 24;
}

// Unpack `count` ints of `bits` width from a byte buffer (LSB first).
// Sub-byte values decode as offset-binary (raw - zeroPoint); 8-bit as signed.
export function unpackInt(bytes, bits, count) {
  const out = new Int32Array(count);
  if (bits === 8) {
    for (let i = 0; i < count; i++) out[i] = signExtend8(bytes[i] & 0xff);
    return out;
  }
  const perByte = 8 / bits;
  const mask = (1 << bits) - 1;
  const zp = 1 << (bits - 1);
  for (let i = 0; i < count; i++) {
    const byte = bytes[Math.floor(i / perByte)];
    const shift = (i % perByte) * bits;
    out[i] = ((byte >> shift) & mask) - zp;
  }
  return out;
}

// Pack signed ints (inverse of unpackInt) — used by tests.
// Sub-byte: raw = value + zeroPoint (offset binary); 8-bit: two's complement.
export function packInt(values, bits) {
  if (bits === 8) {
    const out = new Uint8Array(values.length);
    for (let i = 0; i < values.length; i++) out[i] = values[i] & 0xff;
    return out;
  }
  const perByte = 8 / bits;
  const mask = (1 << bits) - 1;
  const zp = 1 << (bits - 1);
  const out = new Uint8Array(Math.ceil(values.length / perByte));
  for (let i = 0; i < values.length; i++) {
    const shift = (i % perByte) * bits;
    out[Math.floor(i / perByte)] |= ((values[i] + zp) & mask) << shift;
  }
  return out;
}

/**
 * Dequantize a 2-D weight [rows, cols] from packed ints + per-group scales.
 * @param {Int32Array} q       unpacked signed ints, length rows*cols, row-major
 * @param {Float32Array} scales one scale per group, row-major
 * @param {number} rows
 * @param {number} cols
 * @param {number} groupSize   columns sharing one scale (cols => per-row)
 */
export function dequantize(q, scales, rows, cols, groupSize) {
  const out = new Float32Array(rows * cols);
  const groupsPerRow = Math.ceil(cols / groupSize);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const g = r * groupsPerRow + Math.floor(c / groupSize);
      out[r * cols + c] = q[r * cols + c] * scales[g];
    }
  }
  return out;
}

// Convenience: bytes -> f32 weight, given shape/bits/groupSize and scales.
export function dequantizeWeight(bytes, scales, shape, bits, groupSize) {
  const [rows, cols] = shape;
  const q = unpackInt(bytes, bits, rows * cols);
  return dequantize(q, scales, rows, cols, groupSize ?? cols);
}
