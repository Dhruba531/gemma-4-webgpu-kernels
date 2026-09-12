//! Chat template rendering.
//!
//! The real model carries a Jinja `chat_template`. A full Jinja engine is
//! large; we implement the canonical Gemma-4 turn shape natively.
//!
//! Gemma-4 turn format (verified against the published chat_template.jinja —
//! the turn delimiters CHANGED from Gemma-3's <start_of_turn>/<end_of_turn>):
//!   <bos><|turn>user\n{content}<turn|>\n
//!        <|turn>model\n{content}<turn|>\n
//!   ...and a trailing <|turn>model\n when add_generation_prompt is set.
//! <|turn> is token 105, <turn|> (eot_token) is 106 — one of the stop ids.
//! System prompts get their own `<|turn>system` turn in Gemma-4.
//! The <bos> is added by the tokenizer (bos=true), so we DON'T emit it here.

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Message {
    pub role: String,
    pub content: String,
}

impl Message {
    pub fn new(role: &str, content: &str) -> Self {
        Self { role: role.into(), content: content.into() }
    }
    pub fn user(content: &str) -> Self {
        Self::new("user", content)
    }
    pub fn system(content: &str) -> Self {
        Self::new("system", content)
    }
    pub fn assistant(content: &str) -> Self {
        Self::new("assistant", content)
    }
}

pub fn apply_chat_template(messages: &[Message], add_generation_prompt: bool) -> String {
    let mut s = String::new();
    for m in messages {
        let role = if m.role == "assistant" { "model" } else { m.role.as_str() };
        s.push_str(&format!("<|turn>{role}\n{}<turn|>\n", m.content.trim()));
    }
    if add_generation_prompt {
        s.push_str("<|turn>model\n");
    }
    s
}

/// Gemma-4 has a real system turn; keep messages as-is (earlier Gemma versions
/// folded system prompts into the first user turn — no longer needed).
pub fn fold_system_prompt(messages: &[Message]) -> Vec<Message> {
    messages.to_vec()
}
