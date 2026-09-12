// Decode/prefill throughput benchmark on a headless Dawn device.
// Run: node test/bench.mjs [prefillTokens] [decodeSteps] [vocabDivisor]
//
// NOTE: the Node Dawn binding (webgpu@0.4) segfaults in its event pump under
// multi-GiB workloads (dawn::native::InstanceBase::ProcessEvents), so full-size
// runs may die mid-decode. Use vocabDivisor 4 (or 8) for stable Node runs; for
// full-size numbers open test/bench.html in a browser — its awaited readbacks
// are also lower-latency than the Node binding's polled event loop, so browser
// timings are the representative ones.

import { readFileSync } from "node:fs";
import { create, globals } from "webgpu";
Object.assign(globalThis, globals);

import { runBench } from "./bench-core.mjs";

const PREFILL = Number(process.argv[2] ?? 64);
const DECODE = Number(process.argv[3] ?? 24);
const VDIV = Number(process.argv[4] ?? 4);

const gpu = create([]);
const adapter = await gpu.requestAdapter();
const wantBytes = Math.min(adapter.limits.maxStorageBufferBindingSize, adapter.limits.maxBufferSize);
const device = await adapter.requestDevice({
  requiredLimits: { maxStorageBufferBindingSize: wantBytes, maxBufferSize: wantBytes },
});
device.lost.then((info) => console.error("DEVICE LOST:", info.reason, info.message));

const configRaw = JSON.parse(readFileSync(new URL("../reference/config.json", import.meta.url)));
await runBench({ device, configRaw, log: console.log, prefill: PREFILL, decode: DECODE, vdiv: VDIV });

// stdout to a pipe is async; flush before exit or the last lines are lost
await new Promise((r) => process.stdout.write("", r));
process.exit(0);
