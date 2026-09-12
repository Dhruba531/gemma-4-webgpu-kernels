// App bootstrap: wires the landing page, model loading, and streaming chat UI
// to the from-scratch WebGPU engine.

import { Gemma4Mobile } from "./engine/gemma.js";
import { initLanding } from "../landing.js";

const $ = (sel) => document.querySelector(sel);
const fmtBytes = (n) => (n > 1e9 ? (n / 1e9).toFixed(2) + " GB" : (n / 1e6).toFixed(0) + " MB");
const escapeHtml = (s) => s.replace(/[&<>]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;" }[c]));

// minimal markdown: fenced code, inline code, bold, line breaks
function renderMarkdown(text) {
  const parts = text.split(/```/);
  let html = "";
  parts.forEach((chunk, i) => {
    if (i % 2 === 1) {
      const nl = chunk.indexOf("\n");
      const body = nl >= 0 ? chunk.slice(nl + 1) : chunk;
      html += `<pre><code>${escapeHtml(body)}</code></pre>`;
    } else {
      html += escapeHtml(chunk)
        .replace(/`([^`]+)`/g, "<code>$1</code>")
        .replace(/\*\*([^*]+)\*\*/g, "<strong>$1</strong>")
        .replace(/\n/g, "<br>");
    }
  });
  return html;
}

const state = {
  engine: null,
  messages: [],
  generating: false,
  abort: null,
};

function setStatus(text, cls = "") {
  const el = $("#status");
  el.textContent = text;
  el.className = "status " + cls;
}

async function loadModel() {
  const btn = $("#load-btn");
  btn.disabled = true;
  const backend = $("#backend-select")?.value || "webgpu";
  const bar = $("#progress-bar");
  const wrap = $("#progress");
  wrap.style.display = "block";

  try {
    state.engine = await Gemma4Mobile.load({
      backend,
      onProgress: (p) => {
        if (p.stage === "weights" && p.total) {
          const pct = Math.min(100, (p.loaded / p.total) * 100);
          bar.style.width = pct + "%";
          setStatus(`Downloading weights ${fmtBytes(p.loaded)} / ${fmtBytes(p.total)} (shard ${p.shard}/${p.shards})`, "busy");
        } else {
          setStatus(`Preparing: ${p.stage}…`, "busy");
        }
      },
    });
    wrap.style.display = "none";
    setStatus(`Ready · ${state.engine.info.vendor || "GPU"} · ${state.engine.config.num_hidden_layers}L / hidden ${state.engine.config.hidden_size}`, "ok");
    $("#composer").classList.remove("disabled");
    $("#input").disabled = false;
    $("#input").focus();
  } catch (e) {
    console.error(e);
    setStatus("Load failed: " + e.message, "err");
    btn.disabled = false;
  }
}

function addMessage(role, html = "") {
  const list = $("#messages");
  const el = document.createElement("div");
  el.className = "msg " + role;
  el.innerHTML = `<div class="role">${role === "assistant" ? "Gemma" : "You"}</div><div class="body">${html}</div>`;
  list.appendChild(el);
  list.scrollTop = list.scrollHeight;
  return el.querySelector(".body");
}

async function send() {
  if (state.generating || !state.engine) return;
  const input = $("#input");
  const content = input.value.trim();
  if (!content) return;
  input.value = "";
  input.style.height = "auto";

  state.messages.push({ role: "user", content });
  addMessage("user", escapeHtml(content));

  const bodyEl = addMessage("assistant", '<span class="cursor">▍</span>');
  state.generating = true;
  state.abort = new AbortController();
  $("#send-btn").textContent = "Stop";

  const opts = {
    temperature: parseFloat($("#temp").value),
    topK: parseInt($("#topk").value, 10) || 0,
    topP: parseFloat($("#topp").value),
    maxNewTokens: parseInt($("#maxtok").value, 10) || 512,
    signal: state.abort.signal,
  };

  let full = "";
  const t0 = performance.now();
  let n = 0;
  try {
    for await (const ev of state.engine.generate(state.messages, opts)) {
      full = ev.text;
      n++;
      bodyEl.innerHTML = renderMarkdown(full) + '<span class="cursor">▍</span>';
      $("#messages").scrollTop = $("#messages").scrollHeight;
    }
  } catch (e) {
    console.error(e);
    full += "\n\n*[error: " + e.message + "]*";
  }
  const dt = (performance.now() - t0) / 1000;
  bodyEl.innerHTML = renderMarkdown(full) || "<em>(stopped)</em>";
  // Don't record an empty assistant turn (abort/error before the first token) —
  // it would pollute the chat template on the next send.
  if (full.trim()) state.messages.push({ role: "assistant", content: full });
  setStatus(`Ready · ${n} tokens in ${dt.toFixed(1)}s · ${(n / dt).toFixed(1)} tok/s`, "ok");

  state.generating = false;
  $("#send-btn").textContent = "Send";
  input.focus();
}

function stopOrSend() {
  if (state.generating) {
    state.abort?.abort();
  } else {
    send();
  }
}

function enterChat() {
  $("#landing").classList.add("hidden");
  $("#chat").classList.add("visible");
}

window.addEventListener("DOMContentLoaded", () => {
  // landing animation
  try {
    initLanding($("#crt-canvas"));
    requestAnimationFrame(() => $("#crt-frame")?.classList.add("visible"));
    requestAnimationFrame(() => $(".hero-fade")?.classList.add("in"));
  } catch (e) {
    console.warn("landing animation unavailable", e);
  }

  $("#enter-btn")?.addEventListener("click", enterChat);
  $("#load-btn")?.addEventListener("click", loadModel);
  $("#send-btn")?.addEventListener("click", stopOrSend);

  const input = $("#input");
  input?.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      stopOrSend();
    }
  });
  input?.addEventListener("input", () => {
    input.style.height = "auto";
    input.style.height = Math.min(input.scrollHeight, 200) + "px";
  });

  // sync slider value labels
  document.querySelectorAll("input[type=range]").forEach((r) => {
    const out = document.querySelector(`[data-for="${r.id}"]`);
    if (out) {
      const upd = () => (out.textContent = r.value);
      r.addEventListener("input", upd);
      upd();
    }
  });
});
