// Tiny static file server. Usage: node serve.mjs [port]
import http from "node:http";
import { readFile } from "node:fs/promises";
import { extname, join, normalize } from "node:path";

const root = process.cwd();
const port = Number(process.argv[2]) || 8080;
const MIME = {
  ".html": "text/html", ".js": "text/javascript", ".mjs": "text/javascript",
  ".json": "application/json", ".css": "text/css", ".svg": "image/svg+xml",
  ".wasm": "application/wasm", ".map": "application/json",
};

http.createServer(async (req, res) => {
  try {
    let p = decodeURIComponent(req.url.split("?")[0]);
    if (p === "/") p = "/index.html";
    const file = join(root, normalize(p).replace(/^(\.\.[/\\])+/, ""));
    const data = await readFile(file);
    res.writeHead(200, {
      "Content-Type": MIME[extname(file)] || "application/octet-stream",
      // harmless; lets the page use WebGPU + cross-origin Hub fetches
      "Cross-Origin-Opener-Policy": "same-origin",
    });
    res.end(data);
  } catch {
    res.writeHead(404).end("not found");
  }
}).listen(port, () => console.log(`serving ${root} at http://localhost:${port}`));
