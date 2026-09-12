// WebGPU device acquisition + capability probing.

export async function requestDevice() {
  if (!("gpu" in navigator)) {
    throw new Error("WebGPU not available. Use Chrome/Edge 113+ or Safari 18+ with WebGPU enabled.");
  }
  const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
  if (!adapter) throw new Error("No suitable GPU adapter found.");

  // Request a generous storage-buffer limit; Gemma-4 weight tensors are large.
  const wantBytes = Math.min(
    adapter.limits.maxStorageBufferBindingSize,
    adapter.limits.maxBufferSize
  );
  const device = await adapter.requestDevice({
    requiredLimits: {
      maxStorageBufferBindingSize: wantBytes,
      maxBufferSize: wantBytes,
      maxComputeWorkgroupStorageSize: adapter.limits.maxComputeWorkgroupStorageSize,
      maxComputeInvocationsPerWorkgroup: adapter.limits.maxComputeInvocationsPerWorkgroup,
    },
  });

  device.lost.then((info) => {
    console.error("WebGPU device lost:", info.message);
  });

  return {
    adapter,
    device,
    info: {
      vendor: adapter.info?.vendor ?? "unknown",
      architecture: adapter.info?.architecture ?? "",
      maxBufferSize: adapter.limits.maxBufferSize,
      maxStorageBufferBindingSize: adapter.limits.maxStorageBufferBindingSize,
    },
  };
}
