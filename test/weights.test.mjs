// Safetensors + dequant tests. Run: node test/weights.test.mjs
import assert from "node:assert";
import { SafetensorsFile, buildSafetensors, decodeToF32 } from "../src/engine/safetensors.js";
import { packInt, unpackInt, dequantize, dequantizeWeight } from "../src/engine/dequant.js";

let pass = 0;
const ok = (c, m) => { assert.ok(c, m); console.log("  ✓ " + m); pass++; };

console.log("safetensors dtype decode");
{
  // F32
  const f32 = new Float32Array([1.5, -2.25, 3.0, 0.0]);
  // F16 bytes for [1, 2] -> 0x3C00, 0x4000
  const f16 = new Uint16Array([0x3c00, 0x4000]);
  // BF16 for 1.0 (0x3F80) and -2.0 (0xC000)
  const bf16 = new Uint16Array([0x3f80, 0xc000]);

  const buf = buildSafetensors({
    a: { dtype: "F32", shape: [2, 2], bytes: new Uint8Array(f32.buffer.slice()) },
    b: { dtype: "F16", shape: [2], bytes: new Uint8Array(f16.buffer.slice()) },
    c: { dtype: "BF16", shape: [2], bytes: new Uint8Array(bf16.buffer.slice()) },
  });
  const st = new SafetensorsFile(buf);
  ok(JSON.stringify([...st.names()].sort()) === JSON.stringify(["a", "b", "c"]), "lists tensor names");
  ok(Array.from(st.tensor("a").data).join() === "1.5,-2.25,3,0", "F32 decodes");
  ok(Array.from(st.tensor("b").data).join() === "1,2", "F16 decodes");
  ok(Array.from(st.tensor("c").data).join() === "1,-2", "BF16 decodes");
  ok(st.meta("a").shape.join() === "2,2", "preserves shape");
}

console.log("int pack/unpack round-trip");
for (const bits of [2, 4, 8]) {
  const lo = -(1 << (bits - 1)), hi = (1 << (bits - 1)) - 1;
  const vals = [];
  for (let i = 0; i < 33; i++) vals.push(lo + (i % (hi - lo + 1)));
  const packed = packInt(vals, bits);
  const back = unpackInt(packed, bits, vals.length);
  ok(Array.from(back).join() === vals.join(), `${bits}-bit pack/unpack exact`);
}

console.log("dequantize");
{
  // 2 rows, 4 cols, per-row scale. value = q * scale[row].
  const q = Int32Array.from([1, -2, 3, 0, 2, 2, -1, 4]);
  const scales = Float32Array.from([0.5, 2.0]);
  const out = dequantize(q, scales, 2, 4, 4);
  ok(Array.from(out).join() === "0.5,-1,1.5,0,4,4,-2,8", "per-row symmetric dequant");

  // via packed bytes (4-bit)
  const bytes = packInt(Array.from(q), 4);
  const out2 = dequantizeWeight(bytes, scales, [2, 4], 4, 4);
  ok(Array.from(out2).join() === Array.from(out).join(), "dequantizeWeight matches");
}

console.log(`\n${pass} checks passed`);
