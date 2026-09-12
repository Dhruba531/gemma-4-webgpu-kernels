//! QAT (de)quantization helpers.
//!
//! Gemma-4 "qat-mobile" ships weights at mixed bit widths (2/4/8) — see the
//! `quantization_config.module_quant_configs` regex table in config.json.
//!
//! Verified against the reference runtime and the real checkpoint header:
//!   • sub-byte payloads are U8, packed LSB-first, OFFSET-BINARY:
//!       value = (raw - 2^(bits-1)) * scale     (zero-point = 2^(bits-1))
//!     (NOT two's-complement sign extension — raw 0 is the most negative value)
//!   • 8-bit payloads are native signed I8: value = q * scale
//!   • scales are per output row ([N,1]) except embed_tokens_per_layer, which
//!     groups one scale per (row, layer) block of 256.

/// Unpack `count` ints of `bits` width from a byte buffer (LSB first).
/// Sub-byte values decode as offset-binary (raw - zeroPoint); 8-bit as signed.
pub fn unpack_int(bytes: &[u8], bits: u32, count: usize) -> Vec<i32> {
    let mut out = vec![0i32; count];
    if bits == 8 {
        for i in 0..count {
            out[i] = bytes[i] as i8 as i32;
        }
        return out;
    }
    let per_byte = (8 / bits) as usize;
    let mask = (1u32 << bits) - 1;
    let zp = 1i32 << (bits - 1);
    for i in 0..count {
        let byte = bytes[i / per_byte] as u32;
        let shift = ((i % per_byte) as u32) * bits;
        out[i] = ((byte >> shift) & mask) as i32 - zp;
    }
    out
}

/// Pack signed ints (inverse of [`unpack_int`]).
/// Sub-byte: raw = value + zeroPoint (offset binary); 8-bit: two's complement.
pub fn pack_int(values: &[i32], bits: u32) -> Vec<u8> {
    if bits == 8 {
        return values.iter().map(|&v| v as u8).collect();
    }
    let per_byte = (8 / bits) as usize;
    let mask = (1i32 << bits) - 1;
    let zp = 1i32 << (bits - 1);
    let mut out = vec![0u8; values.len().div_ceil(per_byte)];
    for (i, &v) in values.iter().enumerate() {
        let shift = ((i % per_byte) as u32) * bits;
        out[i / per_byte] |= (((v + zp) & mask) << shift) as u8;
    }
    out
}

/// Pack raw unsigned sub-byte values LSB-first (no zero-point shift) — test helper.
pub fn pack_unsigned(values: &[u32], bits: u32) -> Vec<u8> {
    let per_byte = (8 / bits) as usize;
    let mask = (1u32 << bits) - 1;
    let mut out = vec![0u8; values.len().div_ceil(per_byte)];
    for (i, &v) in values.iter().enumerate() {
        let shift = ((i % per_byte) as u32) * bits;
        out[i / per_byte] |= ((v & mask) << shift) as u8;
    }
    out
}

/// Dequantize a 2-D weight `[rows, cols]` from unpacked ints + per-group scales.
pub fn dequantize(q: &[i32], scales: &[f32], rows: usize, cols: usize, group_size: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    let groups_per_row = cols.div_ceil(group_size);
    for r in 0..rows {
        for c in 0..cols {
            let g = r * groups_per_row + c / group_size;
            out[r * cols + c] = q[r * cols + c] as f32 * scales[g];
        }
    }
    out
}

/// Bytes → f32 weight, given shape/bits/group size and scales.
pub fn dequantize_weight(bytes: &[u8], scales: &[f32], shape: [usize; 2], bits: u32, group_size: Option<usize>) -> Vec<f32> {
    let [rows, cols] = shape;
    let q = unpack_int(bytes, bits, rows * cols);
    dequantize(&q, scales, rows, cols, group_size.unwrap_or(cols))
}
