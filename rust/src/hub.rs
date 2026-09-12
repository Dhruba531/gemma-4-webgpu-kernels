//! Minimal HuggingFace Hub client: download a repo's files to a local
//! directory with a progress callback. Honors `HF_TOKEN` for gated repos
//! (the Gemma checkpoints require accepting Google's license).

use std::io::{Read, Write};
use std::path::{Path, PathBuf};

const BASE: &str = "https://huggingface.co";

pub fn file_url(repo: &str, file: &str, revision: &str) -> String {
    format!("{BASE}/{repo}/resolve/{revision}/{file}")
}

/// Progress: (file, downloaded bytes, total bytes if known).
pub type Progress<'a> = &'a mut dyn FnMut(&str, u64, Option<u64>);

fn agent() -> ureq::Agent {
    ureq::Agent::new_with_config(
        ureq::config::Config::builder()
            .http_status_as_error(false)
            .timeout_global(None)
            .timeout_recv_body(None)
            .build(),
    )
}

fn token() -> Option<String> {
    std::env::var("HF_TOKEN").ok().filter(|t| !t.is_empty())
}

/// Download `file` from `repo` into `dir` (skipped when a complete copy is
/// already present). Returns the local path.
pub fn download_file(repo: &str, file: &str, revision: &str, dir: &Path, progress: Progress) -> Result<PathBuf, String> {
    std::fs::create_dir_all(dir).map_err(|e| format!("create {}: {e}", dir.display()))?;
    let dest = dir.join(file);
    let url = file_url(repo, file, revision);
    let agent = agent();
    let mut req = agent.get(&url);
    if let Some(t) = token() {
        req = req.header("Authorization", &format!("Bearer {t}"));
    }
    let mut resp = req.call().map_err(|e| format!("GET {url}: {e}"))?;
    let status = resp.status().as_u16();
    if status == 401 || status == 403 {
        return Err(format!(
            "{file}: HTTP {status}. This repo is gated — accept the license on huggingface.co/{repo} and set HF_TOKEN to a read token."
        ));
    }
    if status != 200 {
        return Err(format!("{file}: HTTP {status}"));
    }
    let total = resp.body().content_length();
    if let (Some(t), Ok(meta)) = (total, std::fs::metadata(&dest)) {
        if meta.len() == t {
            progress(file, t, total);
            return Ok(dest);
        }
    }
    let part = dir.join(format!("{file}.part"));
    let mut out = std::io::BufWriter::new(std::fs::File::create(&part).map_err(|e| format!("create {}: {e}", part.display()))?);
    let mut reader = resp.body_mut().with_config().limit(u64::MAX).reader();
    let mut buf = vec![0u8; 1 << 20];
    let mut done = 0u64;
    loop {
        let n = reader.read(&mut buf).map_err(|e| format!("{file}: read: {e}"))?;
        if n == 0 {
            break;
        }
        out.write_all(&buf[..n]).map_err(|e| format!("{file}: write: {e}"))?;
        done += n as u64;
        progress(file, done, total);
    }
    out.flush().map_err(|e| e.to_string())?;
    drop(out);
    if let Some(t) = total {
        if done != t {
            return Err(format!("{file}: truncated download ({done} of {t} bytes)"));
        }
    }
    std::fs::rename(&part, &dest).map_err(|e| format!("rename {}: {e}", part.display()))?;
    Ok(dest)
}

/// Fetch a small JSON file straight into memory.
pub fn fetch_json(repo: &str, file: &str, revision: &str) -> Result<serde_json::Value, String> {
    let url = file_url(repo, file, revision);
    let mut req = agent().get(&url);
    if let Some(t) = token() {
        req = req.header("Authorization", &format!("Bearer {t}"));
    }
    let mut resp = req.call().map_err(|e| format!("GET {url}: {e}"))?;
    if resp.status().as_u16() != 200 {
        return Err(format!("{file}: HTTP {}", resp.status().as_u16()));
    }
    let bytes = resp.body_mut().with_config().limit(u64::MAX).read_to_vec().map_err(|e| e.to_string())?;
    serde_json::from_slice(&bytes).map_err(|e| format!("{file}: {e}"))
}

/// The files the text engine needs.
pub const MODEL_FILES: &[&str] = &["config.json", "tokenizer.json", "tokenizer_config.json", "generation_config.json", "model.safetensors"];
pub const DEFAULT_REPO: &str = "google/gemma-4-E2B-it-qat-mobile-transformers";
