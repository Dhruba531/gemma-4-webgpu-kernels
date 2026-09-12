//! Tokenizer loaded from a HuggingFace `tokenizer.json`.
//!
//! Gemma ships a SentencePiece model exported to the HF format. We support the
//! two model types that show up in practice — "Unigram" (Viterbi best-path) and
//! "BPE" (merge-rank) — plus the metaspace convention (spaces -> ▁), byte
//! fallback, and explicit added/special tokens. This is a from-scratch encoder,
//! not a wrapper around the `tokenizers` crate.

use std::collections::{HashMap, HashSet};

use serde_json::Value;

const METASPACE: char = '\u{2581}'; // ▁

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ModelType {
    Unigram,
    Bpe,
}

pub struct Tokenizer {
    pub model_type: ModelType,
    pub id_to_piece: Vec<Option<String>>,
    pub piece_to_id: HashMap<String, u32>,
    scores: Vec<f64>,
    merges: HashMap<(String, String), usize>,
    pub unk_id: u32,
    pub byte_fallback: bool,
    /// Special/added tokens: content -> id, plus the same list sorted longest-first.
    pub specials: HashMap<String, u32>,
    specials_by_len: Vec<String>,
    pub bos_id: Option<u32>,
    pub eos_ids: Vec<u32>,
    eos_set: HashSet<u32>,
    /// Longest piece (in chars) bounds the Unigram search window.
    max_piece_chars: usize,
    /// Byte-token id lookup: `<0xNN>` -> byte, built once.
    byte_tokens: HashMap<u32, u8>,
}

#[derive(Clone, Copy, Debug)]
pub struct TokenizerConfig {
    pub bos_token_id: Option<u32>,
    pub eos_token_ids: Option<&'static [u32]>,
}

fn parse_byte_token(piece: &str) -> Option<u8> {
    let inner = piece.strip_prefix("<0x")?.strip_suffix('>')?;
    if inner.len() != 2 {
        return None;
    }
    u8::from_str_radix(inner, 16).ok()
}

impl Tokenizer {
    /// Build from parsed `tokenizer.json`. `bos_token_id`/`eos_token_ids` override
    /// the vocab's `<bos>`/`<eos>` lookups when given (config.json wins).
    pub fn from_json(json: &Value, bos_token_id: Option<u32>, eos_token_ids: Option<&[u32]>) -> Result<Self, String> {
        let model = json.get("model").cloned().unwrap_or(Value::Null);
        let model_type = match model.get("type").and_then(Value::as_str) {
            Some("Unigram") => ModelType::Unigram,
            Some("BPE") => ModelType::Bpe,
            Some(other) => return Err(format!("unsupported tokenizer model type {other}")),
            None => {
                if model.get("merges").is_some() { ModelType::Bpe } else { ModelType::Unigram }
            }
        };

        let mut id_to_piece: Vec<Option<String>> = Vec::new();
        let mut piece_to_id: HashMap<String, u32> = HashMap::new();
        let mut scores = Vec::new();
        let mut merges = HashMap::new();
        let set = |id_to_piece: &mut Vec<Option<String>>, id: u32, piece: &str| {
            let i = id as usize;
            if id_to_piece.len() <= i {
                id_to_piece.resize(i + 1, None);
            }
            id_to_piece[i] = Some(piece.to_string());
        };
        let unk_id;
        match model_type {
            ModelType::Unigram => {
                let vocab = model.get("vocab").and_then(Value::as_array).ok_or("Unigram vocab missing")?;
                scores.resize(vocab.len(), f64::NEG_INFINITY);
                for (id, entry) in vocab.iter().enumerate() {
                    let piece = entry.get(0).and_then(Value::as_str).ok_or("bad vocab entry")?;
                    let score = entry.get(1).and_then(Value::as_f64).unwrap_or(0.0);
                    set(&mut id_to_piece, id as u32, piece);
                    piece_to_id.insert(piece.to_string(), id as u32);
                    scores[id] = score;
                }
                unk_id = model.get("unk_id").and_then(Value::as_u64).unwrap_or(0) as u32;
            }
            ModelType::Bpe => {
                let vocab = model.get("vocab").and_then(Value::as_object).ok_or("BPE vocab missing")?;
                for (piece, id) in vocab {
                    let id = id.as_u64().ok_or("bad BPE id")? as u32;
                    set(&mut id_to_piece, id, piece);
                    piece_to_id.insert(piece.clone(), id);
                }
                if let Some(ms) = model.get("merges").and_then(Value::as_array) {
                    for (rank, m) in ms.iter().enumerate() {
                        let (a, b) = match m {
                            Value::Array(ab) => (ab[0].as_str().unwrap_or("").to_string(), ab[1].as_str().unwrap_or("").to_string()),
                            Value::String(s) => {
                                let mut it = s.splitn(2, ' ');
                                (it.next().unwrap_or("").to_string(), it.next().unwrap_or("").to_string())
                            }
                            _ => continue,
                        };
                        merges.insert((a, b), rank);
                    }
                }
                unk_id = model.get("unk_token").and_then(Value::as_str).and_then(|u| piece_to_id.get(u).copied()).unwrap_or(0);
            }
        }

        let byte_fallback = model.get("byte_fallback").and_then(Value::as_bool).unwrap_or(false);

        // added / special tokens (e.g. <|turn>, <eos>, <bos>)
        let mut specials = HashMap::new();
        if let Some(added) = json.get("added_tokens").and_then(Value::as_array) {
            for a in added {
                let (Some(content), Some(id)) = (a.get("content").and_then(Value::as_str), a.get("id").and_then(Value::as_u64)) else { continue };
                let id = id as u32;
                specials.insert(content.to_string(), id);
                if id_to_piece.get(id as usize).map_or(true, |p| p.is_none()) {
                    set(&mut id_to_piece, id, content);
                }
                piece_to_id.insert(content.to_string(), id);
            }
        }
        let mut specials_by_len: Vec<String> = specials.keys().cloned().collect();
        specials_by_len.sort_by(|a, b| b.len().cmp(&a.len()).then_with(|| a.cmp(b)));

        let bos_id = bos_token_id.or_else(|| piece_to_id.get("<bos>").copied());
        let eos_ids: Vec<u32> = match eos_token_ids {
            Some(e) => e.to_vec(),
            None => piece_to_id.get("<eos>").copied().into_iter().collect(),
        };
        let eos_set = eos_ids.iter().copied().collect();

        let max_piece_chars = piece_to_id.keys().map(|p| p.chars().count()).max().unwrap_or(1).max(1);
        let byte_tokens = id_to_piece
            .iter()
            .enumerate()
            .filter_map(|(id, p)| p.as_deref().and_then(parse_byte_token).map(|b| (id as u32, b)))
            .collect();

        Ok(Self {
            model_type,
            id_to_piece,
            piece_to_id,
            scores,
            merges,
            unk_id,
            byte_fallback,
            specials,
            specials_by_len,
            bos_id,
            eos_ids,
            eos_set,
            max_piece_chars,
            byte_tokens,
        })
    }

    pub fn vocab_len(&self) -> usize {
        self.id_to_piece.len()
    }

    pub fn piece(&self, id: u32) -> Option<&str> {
        self.id_to_piece.get(id as usize).and_then(|p| p.as_deref())
    }

    pub fn is_eos(&self, id: u32) -> bool {
        self.eos_set.contains(&id)
    }

    /// Metaspace pre-tokenizer: spaces -> ▁, prefix a leading ▁.
    pub fn normalize(&self, text: &str) -> String {
        let mut s = String::with_capacity(text.len() + 4);
        s.push(METASPACE);
        for ch in text.chars() {
            s.push(if ch == ' ' { METASPACE } else { ch });
        }
        s
    }

    // Split off special tokens so they map to single ids and are never broken up.
    // Leftmost match wins; at a tie position the longest special wins.
    fn split_specials<'a>(&self, text: &'a str) -> Vec<Part<'a>> {
        let mut parts = Vec::new();
        if self.specials.is_empty() {
            parts.push(Part::Text(text));
            return parts;
        }
        let mut seg_start = 0usize;
        let mut i = 0usize;
        let bytes = text.as_bytes();
        while i < bytes.len() {
            if !text.is_char_boundary(i) {
                i += 1;
                continue;
            }
            let rest = &text[i..];
            if let Some(sp) = self.specials_by_len.iter().find(|sp| rest.starts_with(sp.as_str())) {
                if seg_start < i {
                    parts.push(Part::Text(&text[seg_start..i]));
                }
                parts.push(Part::Special(self.specials[sp]));
                i += sp.len();
                seg_start = i;
            } else {
                i += 1;
            }
        }
        if seg_start < bytes.len() {
            parts.push(Part::Text(&text[seg_start..]));
        }
        parts
    }

    pub fn encode(&self, text: &str, bos: bool) -> Vec<u32> {
        let mut ids = Vec::new();
        if bos {
            if let Some(b) = self.bos_id {
                ids.push(b);
            }
        }
        for part in self.split_specials(text) {
            match part {
                Part::Special(id) => ids.push(id),
                Part::Text(t) => {
                    let norm = self.normalize(t);
                    match self.model_type {
                        ModelType::Unigram => self.encode_unigram(&norm, &mut ids),
                        ModelType::Bpe => self.encode_bpe(&norm, &mut ids),
                    }
                }
            }
        }
        ids
    }

    fn emit_piece(&self, piece: &str, ids: &mut Vec<u32>) -> bool {
        if let Some(&id) = self.piece_to_id.get(piece) {
            ids.push(id);
            return true;
        }
        if self.byte_fallback {
            let mut ok = true;
            for b in piece.bytes() {
                let tok = format!("<0x{b:02X}>");
                match self.piece_to_id.get(&tok) {
                    Some(&bid) => ids.push(bid),
                    None => ok = false,
                }
            }
            return ok;
        }
        false
    }

    // Viterbi best-segmentation for Unigram, over char boundaries.
    fn encode_unigram(&self, text: &str, ids: &mut Vec<u32>) {
        let bounds: Vec<usize> = text.char_indices().map(|(i, _)| i).chain(std::iter::once(text.len())).collect();
        let n = bounds.len() - 1; // number of chars
        let mut best = vec![f64::NEG_INFINITY; n + 1];
        let mut back = vec![-1i64; n + 1];
        let mut back_id = vec![-1i64; n + 1];
        best[0] = 0.0;
        for i in 0..n {
            if best[i] == f64::NEG_INFINITY {
                continue;
            }
            let j_end = n.min(i + self.max_piece_chars);
            for j in i + 1..=j_end {
                let sub = &text[bounds[i]..bounds[j]];
                if let Some(&id) = self.piece_to_id.get(sub) {
                    let score = best[i] + self.scores.get(id as usize).copied().unwrap_or(0.0);
                    if score > best[j] {
                        best[j] = score;
                        back[j] = i as i64;
                        back_id[j] = id as i64;
                    }
                }
            }
            // single-character fallback so the path never breaks (a whole code point)
            if best[i + 1] == f64::NEG_INFINITY {
                best[i + 1] = best[i] - 1e6;
                back[i + 1] = i as i64;
                back_id[i + 1] = -2;
            }
        }
        let mut segs: Vec<(i64, &str)> = Vec::new();
        let mut i = n;
        while i > 0 {
            let b = back[i] as usize;
            segs.push((back_id[i], &text[bounds[b]..bounds[i]]));
            i = b;
        }
        segs.reverse();
        for (id, piece) in segs {
            if id >= 0 {
                ids.push(id as u32);
            } else if !self.emit_piece(piece, ids) {
                ids.push(self.unk_id);
            }
        }
    }

    // Greedy merge-rank BPE over characters.
    fn encode_bpe(&self, text: &str, ids: &mut Vec<u32>) {
        let mut symbols: Vec<String> = text.chars().map(|c| c.to_string()).collect();
        loop {
            let mut best_rank = usize::MAX;
            let mut best_pos = None;
            for i in 0..symbols.len().saturating_sub(1) {
                if let Some(&rank) = self.merges.get(&(symbols[i].clone(), symbols[i + 1].clone())) {
                    if rank < best_rank {
                        best_rank = rank;
                        best_pos = Some(i);
                    }
                }
            }
            let Some(p) = best_pos else { break };
            let merged = format!("{}{}", symbols[p], symbols[p + 1]);
            symbols.splice(p..p + 2, [merged]);
        }
        for s in &symbols {
            if !self.emit_piece(s, ids) {
                ids.push(self.unk_id);
            }
        }
    }

    pub fn decode(&self, ids: &[u32], skip_special: bool) -> String {
        let mut d = self.decoder(skip_special);
        for &id in ids {
            d.push(id);
        }
        d.finish();
        d.state.text
    }

    /// Stateful token decoder for generation.
    pub fn decoder(&self, skip_special: bool) -> Decoder<'_> {
        Decoder { tok: self, state: DecoderState::new(skip_special) }
    }
}

enum Part<'a> {
    Text(&'a str),
    Special(u32),
}

/// Incremental decoder state: `push(id)` returns only the text appended by
/// that token while `text` exposes the accumulated result. Keeps generation
/// O(tokens) and correctly joins multi-token UTF-8 byte sequences.
#[derive(Clone, Debug)]
pub struct DecoderState {
    pub skip_special: bool,
    pub text: String,
    at_start: bool,
    /// Bytes from `<0xNN>` tokens not yet forming a complete UTF-8 sequence.
    pending: Vec<u8>,
}

impl DecoderState {
    pub fn new(skip_special: bool) -> Self {
        Self { skip_special, text: String::new(), at_start: true, pending: Vec::new() }
    }

    fn append(&mut self, chunk: &str) -> String {
        let mut c: String = chunk.chars().map(|ch| if ch == METASPACE { ' ' } else { ch }).collect();
        if self.at_start && !c.is_empty() {
            if let Some(rest) = c.strip_prefix(' ') {
                c = rest.to_string();
            }
            self.at_start = false;
        }
        self.text.push_str(&c);
        c
    }

    /// Decode as much of `pending` as forms complete UTF-8 (streaming
    /// TextDecoder semantics: invalid bytes become U+FFFD, a trailing
    /// incomplete sequence waits for more bytes unless `flush`).
    fn drain_pending(&mut self, flush: bool) -> String {
        let mut out = String::new();
        loop {
            match std::str::from_utf8(&self.pending) {
                Ok(s) => {
                    out.push_str(s);
                    self.pending.clear();
                    break;
                }
                Err(e) => {
                    let valid = e.valid_up_to();
                    out.push_str(std::str::from_utf8(&self.pending[..valid]).unwrap());
                    match e.error_len() {
                        Some(bad) => {
                            out.push('\u{FFFD}');
                            self.pending.drain(..valid + bad);
                        }
                        None => {
                            self.pending.drain(..valid);
                            if flush {
                                out.push('\u{FFFD}');
                                self.pending.clear();
                            }
                            break;
                        }
                    }
                }
            }
        }
        out
    }

    pub fn push(&mut self, tok: &Tokenizer, id: u32) -> String {
        let Some(piece) = tok.piece(id) else { return String::new() };
        if self.skip_special && (tok.specials.contains_key(piece) || Some(id) == tok.bos_id || tok.is_eos(id)) {
            return String::new();
        }
        if let Some(&b) = tok.byte_tokens.get(&id) {
            self.pending.push(b);
            let s = self.drain_pending(false);
            return self.append(&s);
        }
        let mut s = self.drain_pending(true);
        s.push_str(piece);
        self.append(&s)
    }

    pub fn finish(&mut self) -> String {
        let s = self.drain_pending(true);
        self.append(&s)
    }
}

/// Convenience wrapper binding a [`DecoderState`] to its tokenizer.
pub struct Decoder<'a> {
    tok: &'a Tokenizer,
    pub state: DecoderState,
}

impl Decoder<'_> {
    pub fn push(&mut self, id: u32) -> String {
        self.state.push(self.tok, id)
    }
    pub fn finish(&mut self) -> String {
        self.state.finish()
    }
    pub fn text(&self) -> &str {
        &self.state.text
    }
}
