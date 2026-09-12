// Gemma4Mobile — top-level engine: load the model from the Hub and stream chat.
//
//   const engine = await Gemma4Mobile.load({ onProgress });
//   for await (const { token, text } of engine.generate(messages)) { ... }

import { requestDevice } from "./device.js";
import { WebGpuBackend } from "./backend-webgpu.js";
import { CpuBackend } from "./backend-cpu.js";
import { parseHeader } from "./safetensors.js";
import { assembleFromStream, StreamedWeights } from "./stream-loader.js";
import * as hub from "./hub.js";
import { Gemma4Config } from "../model/config.js";
import { Gemma4Model } from "../model/gemma4.js";
import { Tokenizer } from "../tokenizer/tokenizer.js";
import { applyChatTemplate, foldSystemPrompt } from "../tokenizer/template.js";
import { sample } from "../model/sampler.js";

const DEFAULT_REPO = "google/gemma-4-E2B-it-qat-mobile-transformers";

export class Gemma4Mobile {
  constructor({ config, tokenizer, model, backend, info }) {
    this.config = config;
    this.tokenizer = tokenizer;
    this.model = model;
    this.backend = backend;
    this.info = info;
  }

  static async load({ repo = DEFAULT_REPO, revision = "main", backend = "webgpu", onProgress = () => {}, signal } = {}) {
    onProgress({ stage: "device" });
    let be, info;
    try {
      if (backend === "cpu") {
        be = new CpuBackend();
        info = { vendor: "cpu" };
      } else {
        const { device, info: gpuInfo } = await requestDevice();
        be = new WebGpuBackend(device);
        info = gpuInfo;
      }

      // These metadata files are independent; fetch them concurrently to cut
      // startup latency and propagate cancellation through every request.
      onProgress({ stage: "config" });
      const requestOpts = { signal };
      const metadata = Promise.all([
        hub.fetchJson(repo, "config.json", revision, requestOpts),
        hub.fetchJson(repo, "tokenizer.json", revision, requestOpts),
        hub.fetchOptionalJson(repo, "tokenizer_config.json", revision, requestOpts),
        hub.fetchOptionalJson(repo, "generation_config.json", revision, requestOpts),
      ]);
      onProgress({ stage: "tokenizer" });
      const [rawConfig, tokJson, rawTokCfg, rawGenCfg] = await metadata;
      const config = new Gemma4Config(rawConfig);
      const tokCfg = rawTokCfg ?? {};
      const genCfg = rawGenCfg ?? {};
      // EOS: union of config.json and generation_config.json ids. The generation
      // config carries extra stop ids the model config omits (this checkpoint
      // stops on [1, 106, 50] — id 50 is the one actually emitted at turn end).
      const genEos = (Array.isArray(genCfg.eos_token_id)
        ? genCfg.eos_token_id
        : [genCfg.eos_token_id]
      ).filter((x) => x != null);
      config.eos_token_ids = [...new Set([...(config.eos_token_ids ?? []), ...genEos])];
      const tokenizer = new Tokenizer(tokJson, {
        bos_token_id: config.bos_token_id ?? tokCfg.bos_token_id,
        eos_token_ids: config.eos_token_ids,
      });

      // Weights: stream the single 2.46 GB safetensors and assemble one tensor at
      // a time straight to the GPU (never buffering the whole file in JS), skipping
      // the unused vision/audio towers.
      onProgress({ stage: "weights", loaded: 0, total: 0 });
      const file = "model.safetensors";

      // header: read its length (u64) then the JSON via range requests
      const lenBuf = await hub.fetchRange(repo, file, 0, 7, revision, requestOpts);
      const dv = new DataView(lenBuf);
      const headerLen = dv.getUint32(4, true) * 2 ** 32 + dv.getUint32(0, true);
      if (!Number.isSafeInteger(headerLen) || headerLen <= 0) {
        throw new Error(`invalid safetensors header length: ${headerLen}`);
      }
      const headerBuf = await hub.fetchRange(repo, file, 0, 8 + headerLen - 1, revision, requestOpts);
      const { header, dataStart } = parseHeader(headerBuf);

      const resp = await fetch(hub.fileUrl(repo, file, revision), { signal });
      if (!resp.ok) throw new Error(`weights fetch: ${resp.status}`);
      if (!resp.body) throw new Error("weights fetch: streaming response body unavailable");
      const total = Number(resp.headers.get("content-length")) || 0;
      const reader = resp.body.getReader();
      async function* chunks() {
        for (;;) {
          const { done, value } = await reader.read();
          if (done) break;
          yield value;
        }
      }
      const parts = await assembleFromStream({
        chunks: chunks(), header, dataStart, backend: be,
        onProgress: (loaded) => onProgress({ stage: "weights", loaded, total, shard: 1, shards: 1 }),
      });

      const weights = new StreamedWeights(parts, config, be);
      const model = new Gemma4Model(config, be, weights);

      onProgress({ stage: "ready" });
      return new Gemma4Mobile({ config, tokenizer, model, backend: be, info });
    } catch (error) {
      // Loading may have allocated substantial GPU memory before failing.
      be?.destroy();
      throw error;
    }
  }

  /** Tokenize a chat into input ids, applying the Gemma template. */
  encodeChat(messages) {
    const folded = foldSystemPrompt(messages);
    const text = applyChatTemplate(folded, { addGenerationPrompt: true });
    return this.tokenizer.encode(text, { bos: true });
  }

  /**
   * Stream generation. Yields { token, text, tokenId } per step.
   * opts: { maxNewTokens, temperature, topK, topP, repetitionPenalty, signal }
   */
  async *generate(messages, opts = {}) {
    const {
      maxNewTokens = 512,
      temperature = 1.0,
      topK = 64,
      topP = 0.95,
      repetitionPenalty = 1.0,
      signal,
    } = opts;

    this.model.reset();
    const ids = this.encodeChat(messages);
    const eos = new Set(this.config.eos_token_ids);
    const textDecoder = this.tokenizer.createDecoder({ skipSpecial: true });
    const tokenDecoder = this.tokenizer.createDecoder({ skipSpecial: false });
    const recentTokens = repetitionPenalty === 1.0 ? null : [];

    // Prefill the prompt, then decode one token at a time.
    let pos = 0;
    let logits = await this.model.forward(ids, ids.map((_, i) => i));
    pos = ids.length;

    for (let step = 0; step < maxNewTokens; step++) {
      if (signal?.aborted) return;
      const next = sample(logits, {
        temperature,
        topK,
        topP,
        repetitionPenalty,
        recentTokens,
      });
      if (eos.has(next)) return;
      if (recentTokens) {
        recentTokens.push(next);
        if (recentTokens.length > 64) recentTokens.shift();
      }
      const token = tokenDecoder.push(next);
      textDecoder.push(next);
      yield { tokenId: next, token, text: textDecoder.text };
      logits = await this.model.forward([next], [pos]);
      pos++;
    }
  }
}
