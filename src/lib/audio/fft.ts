/** In-place radix-2 Cooley–Tukey FFT. Length must be a power of two. */
export function fft(re: Float32Array, im: Float32Array) {
  const n = re.length;
  let j = 0;
  for (let i = 0; i < n; i++) {
    if (i < j) {
      const tr = re[i]!;
      const ti = im[i]!;
      re[i] = re[j]!;
      im[i] = im[j]!;
      re[j] = tr;
      im[j] = ti;
    }
    let m = n >> 1;
    while (m >= 1 && j >= m) {
      j -= m;
      m >>= 1;
    }
    j += m;
  }
  for (let size = 2; size <= n; size <<= 1) {
    const half = size >> 1;
    const step = (-2 * Math.PI) / size;
    for (let i = 0; i < n; i += size) {
      for (let k = 0; k < half; k++) {
        const angle = step * k;
        const wr = Math.cos(angle);
        const wi = Math.sin(angle);
        const even = i + k;
        const odd = even + half;
        const tr = wr * re[odd]! - wi * im[odd]!;
        const ti = wr * im[odd]! + wi * re[odd]!;
        re[odd] = re[even]! - tr;
        im[odd] = im[even]! - ti;
        re[even] += tr;
        im[even] += ti;
      }
    }
  }
}

export function nextPow2(n: number) {
  let p = 1;
  while (p < n) p <<= 1;
  return p;
}

export function hamming(n: number) {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    w[i] = 0.54 - 0.46 * Math.cos((2 * Math.PI * i) / (n - 1));
  }
  return w;
}

export function preEmphasis(samples: Float32Array, coeff = 0.97) {
  const out = new Float32Array(samples.length);
  out[0] = samples[0] ?? 0;
  for (let i = 1; i < samples.length; i++) {
    out[i] = (samples[i] ?? 0) - coeff * (samples[i - 1] ?? 0);
  }
  return out;
}
