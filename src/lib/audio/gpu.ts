export type GpuInfo = {
  available: boolean;
  adapterName: string;
  vendor: string;
  backend: "webgpu" | "cpu";
  note: string;
};

type GpuNavigator = Navigator & {
  gpu?: {
    requestAdapter: () => Promise<{
      info?: { vendor?: string; device?: string; architecture?: string; description?: string };
      requestDevice: () => Promise<GpuDevice>;
    } | null>;
  };
};

type GpuDevice = {
  createShaderModule: (d: { code: string }) => unknown;
  createComputePipeline: (d: unknown) => { getBindGroupLayout: (i: number) => unknown };
  createBuffer: (d: unknown) => GpuBuffer;
  createBindGroup: (d: unknown) => unknown;
  createCommandEncoder: () => {
    beginComputePass: () => {
      setPipeline: (p: unknown) => void;
      setBindGroup: (i: number, g: unknown) => void;
      dispatchWorkgroups: (x: number, y: number) => void;
      end: () => void;
    };
    copyBufferToBuffer: (a: unknown, b: number, c: unknown, d: number, e: number) => void;
    finish: () => unknown;
  };
  queue: {
    writeBuffer: (buf: unknown, offset: number, data: BufferSource) => void;
    submit: (cmds: unknown[]) => void;
  };
  destroy: () => void;
};

type GpuBuffer = {
  mapAsync: (mode: number) => Promise<void>;
  getMappedRange: () => ArrayBuffer;
  unmap: () => void;
};

export async function detectGpu(): Promise<GpuInfo> {
  const cpu: GpuInfo = {
    available: false,
    adapterName: "CPU",
    vendor: "none",
    backend: "cpu",
    note: "WebGPU is not available here. The analyser still runs a full STFT, pitch, and LPC formant pass on the CPU.",
  };
  if (typeof navigator === "undefined") return cpu;
  const gpu = (navigator as GpuNavigator).gpu;
  if (!gpu) return cpu;
  try {
    const adapter = await gpu.requestAdapter();
    if (!adapter) return cpu;
    const info = adapter.info ?? {};
    const vendor = info.vendor || "GPU";
    const name = info.device || info.description || info.architecture || vendor;
    return {
      available: true,
      adapterName: name,
      vendor,
      backend: "webgpu",
      note: "WebGPU over Vulkan — works on AMD Radeon RX 7700S (Mesa RADV on EndeavorOS) and other modern GPUs. No CUDA required.",
    };
  } catch {
    return cpu;
  }
}

/**
 * Parallel DFT spectrogram on WebGPU. Direct DFT is O(N) per bin but the
 * GPU fans it across frames × bins. Returns null when WebGPU is missing.
 */
export async function gpuSpectrogram(
  samples: Float32Array,
  sampleRate: number,
  fftSize: number,
  hop: number,
): Promise<Float32Array | null> {
  void sampleRate;
  if (typeof navigator === "undefined") return null;
  const gpu = (navigator as GpuNavigator).gpu;
  if (!gpu) return null;
  const adapter = await gpu.requestAdapter();
  if (!adapter) return null;
  const device = await adapter.requestDevice();
  const frameCount = Math.max(0, Math.floor((samples.length - fftSize) / hop) + 1);
  const bins = fftSize / 2;
  if (frameCount <= 0) return null;

  const shader = device.createShaderModule({
    code: `
struct Params { fftSize: u32, hop: u32, frames: u32, bins: u32, }
@group(0) @binding(0) var<storage, read> samples: array<f32>;
@group(0) @binding(1) var<uniform> params: Params;
@group(0) @binding(2) var<storage, read_write> mag: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
  let frame = id.x;
  let bin = id.y;
  if (frame >= params.frames || bin >= params.bins) { return; }
  let start = frame * params.hop;
  var re = 0.0;
  var im = 0.0;
  let n = f32(params.fftSize);
  let twoPi = 6.28318530718;
  for (var i: u32 = 0u; i < params.fftSize; i++) {
    let w = 0.54 - 0.46 * cos(twoPi * f32(i) / (n - 1.0));
    let x = samples[start + i] * w;
    let angle = -twoPi * f32(bin) * f32(i) / n;
    re += x * cos(angle);
    im += x * sin(angle);
  }
  mag[frame * params.bins + bin] = sqrt(re * re + im * im);
}
`,
  });

  const pipeline = device.createComputePipeline({
    layout: "auto",
    compute: { module: shader, entryPoint: "main" },
  });

  const STORAGE = 0x80;
  const COPY_DST = 0x08;
  const COPY_SRC = 0x04;
  const UNIFORM = 0x40;
  const MAP_READ = 0x01;

  const sampleBuf = device.createBuffer({
    size: samples.byteLength,
    usage: STORAGE | COPY_DST,
  });
  const sampleCopy = new Float32Array(samples.length);
  sampleCopy.set(samples);
  device.queue.writeBuffer(sampleBuf, 0, sampleCopy.buffer);

  const paramData = new Uint32Array([fftSize, hop, frameCount, bins]);
  const paramBuf = device.createBuffer({
    size: 16,
    usage: UNIFORM | COPY_DST,
  });
  device.queue.writeBuffer(paramBuf, 0, paramData);

  const magBytes = frameCount * bins * 4;
  const magBuf = device.createBuffer({
    size: magBytes,
    usage: STORAGE | COPY_SRC,
  });
  const readBuf = device.createBuffer({
    size: magBytes,
    usage: COPY_DST | MAP_READ,
  });

  const bind = device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: sampleBuf } },
      { binding: 1, resource: { buffer: paramBuf } },
      { binding: 2, resource: { buffer: magBuf } },
    ],
  });

  const encoder = device.createCommandEncoder();
  const pass = encoder.beginComputePass();
  pass.setPipeline(pipeline);
  pass.setBindGroup(0, bind);
  pass.dispatchWorkgroups(Math.ceil(frameCount / 8), Math.ceil(bins / 8));
  pass.end();
  encoder.copyBufferToBuffer(magBuf, 0, readBuf, 0, magBytes);
  device.queue.submit([encoder.finish()]);
  await readBuf.mapAsync(1);
  const copy = new Float32Array(readBuf.getMappedRange().slice(0));
  readBuf.unmap();
  device.destroy();
  return copy;
}
