//! Safetensors reader + dtype decoding.
//!
//! Layout: `[u64 headerLen][JSON header][raw tensor bytes]`. The header maps
//! name → `{ dtype, shape, data_offsets:[start,end] }` (offsets relative to the
//! start of the byte payload). F32/F16/BF16 decode to `Vec<f32>`; quantized
//! integer payloads (U8/I8) are handed to the backend packed.
//!
//! Natively the 2.46 GB checkpoint is memory-mapped, so — unlike the browser
//! streaming loader — a tensor's bytes are only touched when it is uploaded.

use std::collections::HashMap;
use std::path::Path;

use serde::Deserialize;

const HEADER_LEN_BYTES: usize = 8;

#[derive(Clone, Debug, Deserialize)]
pub struct TensorMeta {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub data_offsets: [usize; 2],
}

enum Bytes {
    Mmap(memmap2::Mmap),
    Owned(Vec<u8>),
}

impl Bytes {
    fn as_slice(&self) -> &[u8] {
        match self {
            Bytes::Mmap(m) => m,
            Bytes::Owned(v) => v,
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum SafetensorsError {
    #[error("io: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid safetensors header: {0}")]
    Header(String),
    #[error("tensor not found: {0}")]
    NotFound(String),
    #[error("unsupported dtype {0}")]
    Dtype(String),
}

/// Parse the header of a safetensors blob. Returns (name → meta, data_start).
pub fn parse_header(buf: &[u8]) -> Result<(HashMap<String, TensorMeta>, usize), SafetensorsError> {
    if buf.len() < HEADER_LEN_BYTES {
        return Err(SafetensorsError::Header("file shorter than 8 bytes".into()));
    }
    let header_len = u64::from_le_bytes(buf[..8].try_into().unwrap()) as usize;
    if header_len == 0 || HEADER_LEN_BYTES + header_len > buf.len() {
        return Err(SafetensorsError::Header(format!("header length {header_len} out of range")));
    }
    let json = &buf[HEADER_LEN_BYTES..HEADER_LEN_BYTES + header_len];
    let raw: HashMap<String, serde_json::Value> = serde_json::from_slice(json).map_err(|e| SafetensorsError::Header(e.to_string()))?;
    let mut header = HashMap::new();
    for (k, v) in raw {
        if k == "__metadata__" {
            continue;
        }
        let meta: TensorMeta = serde_json::from_value(v).map_err(|e| SafetensorsError::Header(format!("{k}: {e}")))?;
        header.insert(k, meta);
    }
    Ok((header, HEADER_LEN_BYTES + header_len))
}

/// f16 → f32 (bit-exact, no `half` dependency).
pub fn f16_to_f32(h: u16) -> f32 {
    let s = if h & 0x8000 != 0 { -1.0f32 } else { 1.0 };
    let e = ((h & 0x7c00) >> 10) as i32;
    let f = (h & 0x03ff) as f32;
    if e == 0 {
        return s * 2f32.powi(-14) * (f / 1024.0);
    }
    if e == 0x1f {
        return if f != 0.0 { f32::NAN } else { s * f32::INFINITY };
    }
    s * 2f32.powi(e - 15) * (1.0 + f / 1024.0)
}

/// bf16 is the high 16 bits of an f32.
pub fn bf16_to_f32(b: u16) -> f32 {
    f32::from_bits((b as u32) << 16)
}

pub fn f32_to_bf16(f: f32) -> u16 {
    (f.to_bits() >> 16) as u16
}

/// Decode a float payload to f32 (handles unaligned input).
pub fn decode_to_f32(dtype: &str, bytes: &[u8]) -> Result<Vec<f32>, SafetensorsError> {
    match dtype {
        "F32" => Ok(bytes.chunks_exact(4).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect()),
        "F16" => Ok(bytes.chunks_exact(2).map(|c| f16_to_f32(u16::from_le_bytes(c.try_into().unwrap()))).collect()),
        "BF16" => Ok(bytes.chunks_exact(2).map(|c| bf16_to_f32(u16::from_le_bytes(c.try_into().unwrap()))).collect()),
        other => Err(SafetensorsError::Dtype(other.to_string())),
    }
}

/// A decoded float tensor.
pub struct FloatTensor {
    pub data: Vec<f32>,
    pub shape: Vec<usize>,
    pub dtype: String,
}

pub struct SafetensorsFile {
    bytes: Bytes,
    header: HashMap<String, TensorMeta>,
    data_start: usize,
}

impl SafetensorsFile {
    /// Memory-map a file on disk.
    pub fn open(path: impl AsRef<Path>) -> Result<Self, SafetensorsError> {
        let file = std::fs::File::open(path)?;
        // SAFETY: the checkpoint is treated as read-only for the life of the map.
        let mmap = unsafe { memmap2::Mmap::map(&file)? };
        Self::from_bytes(Bytes::Mmap(mmap))
    }

    /// Wrap an in-memory blob (tests).
    pub fn from_vec(v: Vec<u8>) -> Result<Self, SafetensorsError> {
        Self::from_bytes(Bytes::Owned(v))
    }

    fn from_bytes(bytes: Bytes) -> Result<Self, SafetensorsError> {
        let (header, data_start) = parse_header(bytes.as_slice())?;
        Ok(Self { bytes, header, data_start })
    }

    pub fn names(&self) -> impl Iterator<Item = &str> {
        self.header.keys().map(|s| s.as_str())
    }

    pub fn contains(&self, name: &str) -> bool {
        self.header.contains_key(name)
    }

    pub fn meta(&self, name: &str) -> Result<&TensorMeta, SafetensorsError> {
        self.header.get(name).ok_or_else(|| SafetensorsError::NotFound(name.to_string()))
    }

    /// Raw bytes for a tensor (no decode).
    pub fn raw_bytes(&self, name: &str) -> Result<&[u8], SafetensorsError> {
        let m = self.meta(name)?;
        let [start, end] = m.data_offsets;
        let buf = self.bytes.as_slice();
        let (a, b) = (self.data_start + start, self.data_start + end);
        if b > buf.len() || a > b {
            return Err(SafetensorsError::Header(format!("{name}: offsets {start}..{end} out of range")));
        }
        Ok(&buf[a..b])
    }

    /// Decoded f32 data + shape for a float tensor.
    pub fn tensor(&self, name: &str) -> Result<FloatTensor, SafetensorsError> {
        let m = self.meta(name)?.clone();
        Ok(FloatTensor { data: decode_to_f32(&m.dtype, self.raw_bytes(name)?)?, shape: m.shape, dtype: m.dtype })
    }

    pub fn total_bytes(&self) -> usize {
        self.bytes.as_slice().len()
    }
}

/// A tensor to serialize with [`build_safetensors`].
pub struct RawTensor {
    pub dtype: String,
    pub shape: Vec<usize>,
    pub bytes: Vec<u8>,
}

/// Build a safetensors blob in memory (used by tests). Entries are laid out in
/// the given order.
pub fn build_safetensors(tensors: &[(String, RawTensor)]) -> Vec<u8> {
    let mut header = serde_json::Map::new();
    let mut offset = 0usize;
    for (name, t) in tensors {
        header.insert(
            name.clone(),
            serde_json::json!({ "dtype": t.dtype, "shape": t.shape, "data_offsets": [offset, offset + t.bytes.len()] }),
        );
        offset += t.bytes.len();
    }
    let json = serde_json::to_vec(&serde_json::Value::Object(header)).unwrap();
    let mut out = Vec::with_capacity(HEADER_LEN_BYTES + json.len() + offset);
    out.extend_from_slice(&(json.len() as u64).to_le_bytes());
    out.extend_from_slice(&json);
    for (_, t) in tensors {
        out.extend_from_slice(&t.bytes);
    }
    out
}
