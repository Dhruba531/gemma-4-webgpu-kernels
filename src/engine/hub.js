// Minimal HuggingFace Hub client (browser fetch).

const BASE = "https://huggingface.co";

export function fileUrl(repo, file, revision = "main") {
  return `${BASE}/${repo}/resolve/${revision}/${file}`;
}

export async function fetchJson(repo, file, revision = "main", { signal } = {}) {
  const r = await fetch(fileUrl(repo, file, revision), { signal });
  if (!r.ok) throw new Error(`fetch ${file}: ${r.status}`);
  return r.json();
}

// Fetch a byte range [start, end] (inclusive) of a file.
export async function fetchRange(repo, file, start, end, revision = "main", { signal } = {}) {
  const r = await fetch(fileUrl(repo, file, revision), {
    headers: { Range: `bytes=${start}-${end}` },
    signal,
  });
  // A 200 means the server ignored Range. Accepting it for a multi-gigabyte
  // checkpoint would accidentally buffer the entire file just to read 8 bytes.
  if (r.status !== 206) throw new Error(`range fetch ${file}: expected 206, got ${r.status}`);
  const out = await r.arrayBuffer();
  const expected = end - start + 1;
  if (out.byteLength !== expected) {
    throw new Error(`range fetch ${file}: expected ${expected} bytes, got ${out.byteLength}`);
  }
  return out;
}

export async function fetchOptionalJson(repo, file, revision = "main", { signal } = {}) {
  try {
    const r = await fetch(fileUrl(repo, file, revision), { signal });
    if (!r.ok) return null;
    return await r.json();
  } catch (error) {
    // Optional means "missing", not "ignore cancellation".
    if (signal?.aborted) throw error;
    return null;
  }
}

/**
 * Download a binary file with progress. Streams the body so the UI can show a
 * byte counter for the (large) weight file.
 * @param {(loaded:number,total:number)=>void} onProgress
 */
export async function fetchArrayBuffer(repo, file, { revision = "main", onProgress, signal } = {}) {
  const r = await fetch(fileUrl(repo, file, revision), { signal });
  if (!r.ok) throw new Error(`fetch ${file}: ${r.status}`);
  const total = Number(r.headers.get("content-length")) || 0;
  if (!r.body || !onProgress) return r.arrayBuffer();

  const reader = r.body.getReader();
  const chunks = [];
  let loaded = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    loaded += value.byteLength;
    onProgress(loaded, total);
  }
  const out = new Uint8Array(loaded);
  let p = 0;
  for (const c of chunks) { out.set(c, p); p += c.byteLength; }
  return out.buffer;
}
