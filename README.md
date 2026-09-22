# Phonolab

Voice laboratory: extract **phonemes** from a clip, convert them to text, and score each one for acoustic gender, attributes, and volume — with an expandable spectrogram.

Runs in the browser. On an **AMD Radeon RX 7700S** (EndeavorOS / Titan OS) it uses **WebGPU over Vulkan** (Mesa RADV). No CUDA.

Gender scores are acoustic (F0 + formant scale), not identity.

## What it does

- STFT spectrogram with F1 / F2 / F3 traces
- YIN pitch + LPC formants
- Phoneme timeline (IPA + ARPAbet) with per-glyph confidence
- Subphonemes: **onset** (start) · **nucleus** (center) · **coda** (end), rise / drop / level
- Volume (dB), brightness, breathiness, nasality, roughness, tension, resonance
- Optional Grok Voice transcription and a phonetic reading

## Install on EndeavorOS (Titan OS)

Needs **Node.js 22+**, **npm**, and a Chromium-based browser with WebGPU (Chrome / Chromium / Brave, recent). Firefox 141+ also has WebGPU.

### 1. Packages

```bash
sudo pacman -S --needed git nodejs npm vulkan-radeon vulkan-icd-loader mesa
```

`nodejs` on Arch/EndeavorOS is current enough. If `node -v` is older than 22, use [nvm](https://github.com/nvm-sh/nvm) or `fnm` and install 22 LTS.

Confirm the AMD GPU is visible:

```bash
vulkaninfo --summary | grep -i -E 'deviceName|AMD'
```

You want something like `AMD Radeon RX 7700S`.

### 2. Clone and install

```bash
git clone https://github.com/sera5m/phonolab.git
cd phonolab
npm install
```

### 3. Run

```bash
npm run dev
```

Open **http://localhost:8080** in Chrome or Chromium.

First load synthesizes cardinal vowels so the spectrogram is immediate. Then drop a wav/mp3, record, or pick another demo.

### 4. WebGPU (RX 7700S)

In Chrome: `chrome://gpu` → **WebGPU** should be Hardware accelerated.

If it is not:

- Update Mesa: `sudo pacman -Syu mesa vulkan-radeon`
- On Chrome 113–120 you may need `chrome://flags/#enable-unsafe-webgpu` — current Chrome does not.
- Use a Chromium build, not an ancient Electron wrapper.

Without WebGPU, Phonolab still runs the full STFT / pitch / formant pass on the CPU. The badge in the header tells you which path is active.

### 5. Optional: Grok transcription

Local analysis never leaves the machine. **Transcribe with Grok** and **Phonetic reading** call the xAI API.

```bash
export XAI_API_KEY="xai-..."
npm run dev
```

Get a key at [console.x.ai](https://console.x.ai). Do not commit it.

## Other commands

| Command | What it does |
| --- | --- |
| `npm run dev` | Dev server on port 8080 |
| `npm run build` | Production build |
| `npm run typecheck` | TypeScript check |

## How a phoneme is scored

A phoneme is not a still frame:

1. **Onset** — energy rises; formants glide in (burst + VOT on stops)
2. **Nucleus** — steady target; vowel identity in F1 (height) and F2 (front/back)
3. **Coda** — energy drops; formants head toward the next sound

Gender uses F0 (typical adult-male ~85–180 Hz, adult-female ~165–255 Hz) plus formant scale (shorter vocal tract → higher resonances). Overlap around 150–180 Hz lowers confidence.

## License

MIT
