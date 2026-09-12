// Chat template rendering.
//
// The real model carries a Jinja `chat_template` (chat_template.jinja on the
// Hub). A full Jinja engine is large; we implement the canonical Gemma-4 turn
// shape natively and expose it as `applyChatTemplate`.
//
// Gemma-4 turn format (verified against the published chat_template.jinja —
// the turn delimiters CHANGED from Gemma-3's <start_of_turn>/<end_of_turn>):
//   <bos><|turn>user\n{content}<turn|>\n
//        <|turn>model\n{content}<turn|>\n
//   ...and a trailing <|turn>model\n when add_generation_prompt is set.
// <|turn> is token 105, <turn|> (eot_token) is 106 — one of the stop ids.
// System prompts get their own `<|turn>system` turn in Gemma-4.
// The <bos> is added by the tokenizer (bos=true), so we DON'T emit it here.

export function applyChatTemplate(messages, { addGenerationPrompt = true } = {}) {
  let s = "";
  for (const m of messages) {
    const role = m.role === "assistant" ? "model" : m.role;
    const content = (m.content ?? "").trim();
    s += `<|turn>${role}\n${content}<turn|>\n`;
  }
  if (addGenerationPrompt) s += `<|turn>model\n`;
  return s;
}

// Gemma-4 has a real system turn; keep messages as-is (earlier Gemma versions
// folded system prompts into the first user turn — no longer needed).
export function foldSystemPrompt(messages) {
  return messages;
}
