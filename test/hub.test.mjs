// Hugging Face Hub transport validation. Range requests must never silently
// fall back to a full multi-gigabyte response, and cancellation must propagate.
import assert from "node:assert";
import { fetchOptionalJson, fetchRange } from "../src/engine/hub.js";

let pass = 0;
const ok = (condition, message) => {
  assert.ok(condition, message);
  console.log("  ✓ " + message);
  pass++;
};
const originalFetch = globalThis.fetch;

try {
  console.log("range fetch");
  const controller = new AbortController();
  let request;
  globalThis.fetch = async (_url, options) => {
    request = options;
    return new Response(Uint8Array.of(1, 2, 3, 4), { status: 206 });
  };
  const range = await fetchRange("repo", "weights.bin", 4, 7, "main", { signal: controller.signal });
  ok(range.byteLength === 4, "accepts an exact partial response");
  ok(request.headers.Range === "bytes=4-7" && request.signal === controller.signal, "forwards range and abort signal");

  globalThis.fetch = async () => new Response(new Uint8Array(16), { status: 200 });
  await assert.rejects(() => fetchRange("repo", "weights.bin", 0, 3), /expected 206, got 200/);
  ok(true, "rejects a server that ignores Range");

  globalThis.fetch = async () => new Response(Uint8Array.of(1, 2, 3), { status: 206 });
  await assert.rejects(() => fetchRange("repo", "weights.bin", 0, 3), /expected 4 bytes, got 3/);
  ok(true, "rejects a short partial response");

  console.log("cancellation");
  const aborted = new AbortController();
  aborted.abort();
  globalThis.fetch = async (_url, { signal }) => {
    assert.equal(signal, aborted.signal);
    throw new DOMException("aborted", "AbortError");
  };
  await assert.rejects(
    () => fetchOptionalJson("repo", "optional.json", "main", { signal: aborted.signal }),
    { name: "AbortError" },
  );
  ok(true, "optional metadata fetch preserves cancellation");
} finally {
  globalThis.fetch = originalFetch;
}

console.log(`\n${pass} checks passed`);
