//! Tokenizer + chat template tests (mirrors test/tokenizer.test.mjs).
use gemma_wgpu::template::{apply_chat_template, fold_system_prompt, Message};
use gemma_wgpu::tokenizer::Tokenizer;

fn tok() -> Tokenizer {
    let json = serde_json::json!({
        "added_tokens": [
            { "id": 0, "content": "<bos>" }, { "id": 1, "content": "<eos>" },
            { "id": 2, "content": "<|turn>" }, { "id": 3, "content": "<turn|>" },
        ],
        "model": {
            "type": "Unigram", "unk_id": 4, "byte_fallback": true,
            "vocab": [
                ["<bos>", 0], ["<eos>", 0], ["<|turn>", 0], ["<turn|>", 0], ["<unk>", 0],
                ["▁hello", -1], ["▁world", -1], ["▁", -3],
                ["h", -5], ["e", -5], ["l", -5], ["o", -5],
                ["<0x21>", -6], ["<0xE2>", -6], ["<0x82>", -6], ["<0xAC>", -6]
            ]
        }
    });
    Tokenizer::from_json(&json, Some(0), Some(&[1])).unwrap()
}

#[test]
fn encode_decode_roundtrip() {
    let t = tok();
    let ids = t.encode("hello world", true);
    assert_eq!(ids[0], 0, "starts with bos");
    assert!(ids.contains(&t.piece_to_id["▁hello"]), "encodes ▁hello as one piece");
    assert!(ids.contains(&t.piece_to_id["▁world"]), "encodes ▁world as one piece");
    assert_eq!(t.decode(&ids, true), "hello world");
}

#[test]
fn byte_fallback() {
    let t = tok();
    let bf = t.encode("o!", false);
    assert!(bf.contains(&t.piece_to_id["<0x21>"]), "falls back to byte token for '!'");
    let euro: Vec<u32> = ["<0xE2>", "<0x82>", "<0xAC>"].iter().map(|p| t.piece_to_id[*p]).collect();
    assert_eq!(t.decode(&euro, true), "€", "decodes a multi-token UTF-8 character");
    let mut inc = t.decoder(true);
    let frags: String = euro.iter().map(|&id| inc.push(id)).collect();
    assert_eq!(frags, "€");
    assert_eq!(inc.text(), "€", "incremental decoder joins UTF-8 byte tokens");
    // an unfinished byte sequence flushes to U+FFFD
    let mut inc = t.decoder(true);
    inc.push(euro[0]);
    assert_eq!(inc.finish(), "\u{FFFD}");
}

#[test]
fn special_tokens_stay_atomic() {
    let t = tok();
    let st = t.encode("<|turn>hello", false);
    assert_eq!(st[0], 2, "leading special token -> single id");
    let mid = t.encode("hello<turn|>hello", false);
    assert_eq!(mid.iter().filter(|&&i| i == 3).count(), 1);
    assert!(!t.decode(&mid, true).contains("<turn|>"));
    assert!(t.decode(&mid, false).contains("<turn|>"));
}

#[test]
fn bpe_model() {
    let json = serde_json::json!({
        "model": { "type": "BPE", "unk_token": "<unk>",
            "vocab": { "<unk>": 0, "▁": 1, "a": 2, "b": 3, "ab": 4, "▁ab": 5 },
            "merges": ["a b", "▁ ab"] }
    });
    let t = Tokenizer::from_json(&json, None, None).unwrap();
    assert_eq!(t.encode("ab", false), vec![5]);
    assert_eq!(t.encode("ba", false), vec![1, 3, 2]);
}

#[test]
fn chat_template() {
    let msgs = fold_system_prompt(&[Message::system("Be nice."), Message::user("Hi")]);
    let text = apply_chat_template(&msgs, true);
    assert!(text.contains("<|turn>system\nBe nice.<turn|>"), "system gets its own turn");
    assert!(text.ends_with("<|turn>model\n"), "ends with model generation prompt");
    assert!(apply_chat_template(&[Message::assistant("x")], false).starts_with("<|turn>model\nx<turn|>\n"));
}
