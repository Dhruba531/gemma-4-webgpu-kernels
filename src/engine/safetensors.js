// Safetensors reader + dtype decoding.
//
// Layout:  [u64 headerLen][JSON header][raw tensor bytes]
// Header maps name -> { dtype, shape, data_offsets:[start,end] } (offsets are
// relative to the start of the byte payload). We decode F32/F16/BF16 to
// Float32Array; quantized integer payloads are handed to dequant.js.

const HEADER_LEN_BYTES = 8;

export function parseHeader(buf) {
  const dv = new DataView(buf);
  // u64 little-endian; header length comfortably fits in 53 bits.
  const lenLo = dv.getUint32(0, true);
  const lenHi = dv.getUint32(4, true);
  const headerLen = lenHi * 2 ** 32 + lenLo;
  const json = new TextDecoder().decode(new Uint8Array(buf, HEADER_LEN_BYTES, headerLen));
  const header = JSON.parse(json);
  const dataStart = HEADER_LEN_BYTES + headerLen;
  return { header, dataStart };
}

// f16 -> f32
function f16to32(h) {
  const s = (h & 0x8000) >> 15;
  const e = (h & 0x7c00) >> 10;
  const f = h & 0x03ff;
  if (e === 0) return (s ? -1 : 1) * 2 ** -14 * (f / 1024);
  if (e === 0x1f) return f ? NaN : (s ? -Infinity : Infinity);
  return (s ? -1 : 1) * 2 ** (e - 15) * (1 + f / 1024);
}

export function decodeToF32(dtype, bytes) {
  // Tensor payloads aren't guaranteed to start on a 4/2-byte boundary, so copy
  // into a fresh aligned buffer before creating typed-array views.
  const aligned = bytes.byteOffset % 4 === 0 ? bytes : bytes.slice();
  switch (dtype) {
    case "F32":
      return new Float32Array(aligned.buffer, aligned.byteOffset, aligned.byteLength / 4).slice();
    case "F16": {
      const n = aligned.byteLength / 2;
      const u16 = new Uint16Array(aligned.buffer, aligned.byteOffset, n);
      const out = new Float32Array(n);
      for (let i = 0; i < n; i++) out[i] = f16to32(u16[i]);
      return out;
    }
    case "BF16": {
      // bf16 is the high 16 bits of an f32.
      const n = aligned.byteLength / 2;
      const u16 = new Uint16Array(aligned.buffer, aligned.byteOffset, n);
      const out = new Float32Array(n);
      const u32 = new Uint32Array(out.buffer);
      for (let i = 0; i < n; i++) u32[i] = u16[i] << 16;
      return out;
    }
    default:
      throw new Error(`decodeToF32: unsupported dtype ${dtype}`);
  }
}

export class SafetensorsFile {
  constructor(arrayBuffer) {
    const { header, dataStart } = parseHeader(arrayBuffer);
    this.buf = arrayBuffer;
    this.header = header;
    this.dataStart = dataStart;
  }

  names() {
    return Object.keys(this.header).filter((k) => k !== "__metadata__");
  }

  meta(name) {
    const m = this.header[name];
    if (!m) throw new Error(`tensor not found: ${name}`);
    return m;
  }

  /** Raw bytes for a tensor (no decode). */
  rawBytes(name) {
    const m = this.meta(name);
    const [start, end] = m.data_offsets;
    return new Uint8Array(this.buf, this.dataStart + start, end - start);
  }

  /** Decoded Float32Array + shape for a float tensor. */
  tensor(name) {
    const m = this.meta(name);
    return { data: decodeToF32(m.dtype, this.rawBytes(name)), shape: m.shape, dtype: m.dtype };
  }
}

// Build a safetensors blob in memory (used by tests).
export function buildSafetensors(tensors) {
  // tensors: { name: {dtype, shape, bytes:Uint8Array} }
  const header = {};
  let offset = 0;
  const chunks = [];
  for (const [name, t] of Object.entries(tensors)) {
    header[name] = { dtype: t.dtype, shape: t.shape, data_offsets: [offset, offset + t.bytes.byteLength] };
    chunks.push(t.bytes);
    offset += t.bytes.byteLength;
  }
  const json = new TextEncoder().encode(JSON.stringify(header));
  const out = new Uint8Array(HEADER_LEN_BYTES + json.byteLength + offset);
  const dv = new DataView(out.buffer);
  dv.setUint32(0, json.byteLength, true);
  dv.setUint32(4, 0, true);
  out.set(json, HEADER_LEN_BYTES);
  let p = HEADER_LEN_BYTES + json.byteLength;
  for (const c of chunks) { out.set(c, p); p += c.byteLength; }
  return out.buffer;
}
