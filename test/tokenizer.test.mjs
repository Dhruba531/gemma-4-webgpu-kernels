// Tokenizer + chat template tests. Run: node test/tokenizer.test.mjs
import assert from "node:assert";
import { Tokenizer } from "../src/tokenizer/tokenizer.js";
import { applyChatTemplate, foldSystemPrompt } from "../src/tokenizer/template.js";

let pass = 0;
const ok = (c, m) => { assert.ok(c, m); console.log("  ✓ " + m); pass++; };

// Minimal Unigram vocab incl. metaspace pieces, byte fallback, specials.
const unigram = (piece, score) => [piece, score];
const json = {
  added_tokens: [
    { id: 0, content: "<bos>" },
    { id: 1, content: "<eos>" },
    { id: 2, content: "<|turn>" },
    { id: 3, content: "<turn|>" },
  ],
  model: {
    type: "Unigram",
    unk_id: 4,
    byte_fallback: true,
    vocab: [
      unigram("<bos>", 0), unigram("<eos>", 0), unigram("<|turn>", 0),
      unigram("<turn|>", 0), unigram("<unk>", 0),
      unigram("▁hello", -1), unigram("▁world", -1), unigram("▁", -3),
      unigram("h", -5), unigram("e", -5), unigram("l", -5), unigram("o", -5),
      unigram("<0x21>", -6), // '!'
      unigram("<0xE2>", -6), unigram("<0x82>", -6), unigram("<0xAC>", -6), // '€'
    ],
  },
};

const tok = new Tokenizer(json, { bos_token_id: 0, eos_token_ids: [1] });

console.log("encode/decode");
const ids = tok.encode("hello world", { bos: true });
ok(ids[0] === 0, "starts with bos");
ok(ids.includes(tok.pieceToId.get("▁hello")), "encodes ▁hello as one piece");
ok(ids.includes(tok.pieceToId.get("▁world")), "encodes ▁world as one piece");
const dec = tok.decode(ids);
ok(dec === "hello world", `roundtrips text (got "${dec}")`);

console.log("byte fallback");
const bf = tok.encode("o!", { bos: false });
ok(bf.includes(tok.pieceToId.get("<0x21>")), "falls back to byte token for '!'");
const euro = ["<0xE2>", "<0x82>", "<0xAC>"].map((p) => tok.pieceToId.get(p));
ok(tok.decode(euro) === "€", "decodes a multi-token UTF-8 character");
const incremental = tok.createDecoder();
const fragments = euro.map((id) => incremental.push(id));
ok(fragments.join("") === "€" && incremental.text === "€", "incremental decoder joins UTF-8 byte tokens");

console.log("special tokens stay atomic");
const st = tok.encode("<|turn>hello", { bos: false });
ok(st[0] === 2, "leading special token -> single id");

console.log("chat template");
const msgs = foldSystemPrompt([
  { role: "system", content: "Be nice." },
  { role: "user", content: "Hi" },
]);
const text = applyChatTemplate(msgs, { addGenerationPrompt: true });
ok(text.includes("<|turn>system\nBe nice.<turn|>"), "system gets its own turn");
ok(text.endsWith("<|turn>model\n"), "ends with model generation prompt");

console.log(`\n${pass} checks passed`);
