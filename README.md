# Fusion Horizon UI

> **A hermeneutic interpretation engine with built-in local LLM inference — no API key required, no cloud dependency.**

Built with **Tauri v2 + React 19 + TypeScript + Vite** and powered by **ONNX Runtime Web (WebGPU/WASM)**, this application runs the four-phase hermeneutic circle pipeline (Horizon Mapping, Dialectical Analysis, Iterative Convergence, Horizon Fusion) using a local [Qwen3.5-0.8B](https://huggingface.co/onnx-community/Qwen3.5-0.8B-ONNX) language model that executes entirely on your own hardware. Your data never leaves your machine.

An optional OpenRouter remote provider is available for users who want to use larger cloud models with their own API key.

---

## What It Does

| Capability | Detail |
|---|---|
| **Hermeneutic Circle Pipeline** | Four-agent interpretive system: Horizon Mapping, Dialectic Analysis, Iteration Control, Horizon Fusion |
| **Built-in Local LLM** | Qwen3.5-0.8B Q4 ONNX — runs entirely in-browser via Web Worker |
| **WebGPU Acceleration** | GPU inference via WebGPU API; automatic WASM fallback on unsupported hardware |
| **Zero-Config Default** | Works out of the box — no API key, no server, no Python |
| **OpenRouter Remote Provider** | Optional BYOK remote inference via OpenRouter.ai for larger models |
| **Streaming Inference** | Token-by-token generation with live progress and abort support |
| **Thinking Mode** | Toggle Qwen3.5's chain-of-thought reasoning on or off |
| **Interactive & Single-Query Modes** | One-shot interpretation or multi-turn interactive sessions |
| **Custom Writer Prompts** | Load custom writer prompts to guide the fusion synthesis phase |
| **Session Persistence** | Conversation history saved locally; reload past sessions |
| **Response Export** | Export results as Markdown or JSON |
| **Portable MSI Installer** | Single `.msi` file for Windows distribution via Tauri v2 |

---

## Architecture Overview

```
+-----------------------------------------------------------------+
|  Tauri v2 (Rust shell)                                          |
|  +-----------------------------------------------------------+  |
|  |  Tauri WebView                                             |  |
|  |                                                            |  |
|  |  React UI Thread              Web Worker                   |  |
|  |  +--------------------+       +-----------------------+    |  |
|  |  | Home / QueryInput  |       | qwen.worker.ts        |    |  |
|  |  | SettingsPanel      |       | @hf/transformers v4   |    |  |
|  |  | QwenStatusBar      |       | ORT WebGPU / WASM     |    |  |
|  |  | ProcessTimeline    |       +-----------------------+    |  |
|  |  | ResponseDisplay    |              ^                     |  |
|  |  +--------------------+              |                     |  |
|  |           |                    QwenLocalProvider            |  |
|  |           v                          |                     |  |
|  |  +--------------------+       +------+------+              |  |
|  |  | OrchestratorController |   | ILLMProvider |             |  |
|  |  +--------------------+       +------+------+              |  |
|  |  | HorizonMapperAgent |              |                     |  |
|  |  | DialecticAnalyzer  |    +---------+---------+           |  |
|  |  | IterationController|    | OpenRouterProvider |           |  |
|  |  | HorizonFusionAgent |    +-------------------+           |  |
|  |  +--------------------+      (optional remote)             |  |
|  +-----------------------------------------------------------+  |
+-----------------------------------------------------------------+
```

- **ILLMProvider interface** decouples all agents from any specific LLM backend. Two implementations: `QwenLocalProvider` (default) and `OpenRouterProvider` (optional).
- **Web Worker** runs ONNX inference in isolation, keeping the UI responsive during token generation.
- **Zustand stores** manage hermeneutic state, model lifecycle, UI settings, and session history with localStorage persistence.
- **providerRegistry** singleton manages provider selection and initialization.

---

## Hermeneutic Pipeline

The interpretation pipeline executes four phases regardless of which LLM provider is active:

| Phase | Agent | Purpose |
|---|---|---|
| 1. Horizon Mapping | `HorizonMapperAgent` | Maps pre-understanding, assumptions, biases, and user context |
| 2. Dialectical Analysis | `DialecticAnalyzerAgent` | Part-whole dialectical interpretation with tensions and insights |
| 3. Iteration Control | `IterationControllerAgent` | Evaluates convergence, decides to continue or finalize (2-5 cycles) |
| 4. Horizon Fusion | `HorizonFusionAgent` | Synthesizes all cycle insights and fuses agent/user horizons |

Convergence is measured by text similarity between successive cycle interpretations (default threshold: 0.85). The pipeline runs 2-5 dialectical cycles before producing a final fused response.

---

## Requirements

### To Run the MSI Installer
- Windows 10/11 x64
- ~1 GB free disk space (for model cache)
- Internet connection on first launch only (for ~850 MB model download)
- A WebGPU-capable GPU is recommended (any modern Nvidia/AMD/Intel GPU); falls back to CPU WASM if unavailable

### To Build From Source
- [Node.js](https://nodejs.org/) >= 20
- [Rust](https://rustup.rs/) stable toolchain (MSVC)
- [Tauri CLI v2](https://tauri.app/start/): `cargo install tauri-cli --version "^2"`
- [WiX Toolset v4+](https://github.com/wixtoolset/wix): `dotnet tool install --global wix`
- Windows SDK (included with Visual Studio Build Tools)

---

## Getting Started

### Option A — Install the MSI (End Users)

1. Download the latest `.msi` from the [Releases](https://github.com/MushroomFleet/Fusion-Horizon-QwenORT/releases) page.
2. Run the installer.
3. Launch **Fusion Horizon** from the Start Menu.
4. On first launch, the Qwen 0.8B model (~850 MB) downloads automatically and is cached for all future use.
5. Submit a query — the hermeneutic pipeline runs fully offline from this point.


### Option B — Use OpenRouter Instead of Local Model

1. Open Settings
2. Change the LLM Provider dropdown to **"OpenRouter (Remote)"**
3. Enter your API key from [OpenRouter.ai](https://openrouter.ai/keys)
4. (Optional) Change the model name (default: `x-ai/grok-4-fast:online`)
5. Save — queries now use the remote model

You can switch between local and remote providers at any time.

---

## LLM Provider System

The application supports two LLM providers, selectable in Settings:

| Provider | Label | API Key Required | Default | Offline Capable |
|---|---|---|---|---|
| `qwen-local` | Built-in (Qwen 0.8B) | No | Yes | Yes (after first download) |
| `openrouter` | OpenRouter (Remote) | Yes (BYOK) | No | No |

When the **Built-in** provider is selected:
- The QwenStatusBar shows model loading/download progress
- The model runs in a Web Worker using ONNX Runtime Web
- No API key or account is needed
- All inference happens on your local GPU/CPU

When **OpenRouter** is selected:
- API key and model name fields appear in Settings
- Your key is stored in browser localStorage only (never sent to our servers)
- Any model available on OpenRouter can be used

---

## Configuration

### Hermeneutic Parameters

| Setting | Default | Range | Effect |
|---|---|---|---|
| Min Cycles | 2 | 1-10 | Minimum dialectical cycles before convergence check |
| Max Cycles | 5 | 1-10 | Maximum cycles (forces convergence) |
| Convergence Threshold | 0.85 | 0.5-1.0 | Similarity score to trigger natural convergence |

### Generation Parameters

| Setting | Default | Range | Effect |
|---|---|---|---|
| Temperature | 0.7 | 0.0-2.0 | Randomness; lower = more deterministic |
| Max Tokens | Auto | 100-8000 | Maximum response length per agent call |

### Qwen Local Model Parameters (Internal)

| Parameter | Default | Range | Effect |
|---|---|---|---|
| `maxNewTokens` | 2048 | 1-2048 | Maximum tokens per generation |
| `topP` | 0.9 | 0.1-1.0 | Nucleus sampling threshold |
| `repetitionPenalty` | 1.1 | 1.0-2.0 | Penalises repeated phrases |
| Thinking mode | Off | On/Off | Pipeline uses `/no_think` for structured JSON output |

---

## First-Run Model Download

On first launch with the Built-in provider, the Qwen 0.8B model downloads from the HuggingFace CDN:

- **Download size:** ~850 MB (Q4 quantized weights)
- **Download time:** 30-60 seconds on broadband
- **Cache location:** Browser Cache API (persists across sessions)
- **Subsequent launches:** 5-15 seconds (loaded from cache, no network needed)

The QwenStatusBar shows real-time progress with file name and percentage during download.

---

## Performance

### Local Qwen Inference

| Metric | WebGPU | WASM (CPU) |
|---|---|---|
| First load (cold) | 30-60 seconds | 40-90 seconds |
| Warm load (cached) | 5-15 seconds | 10-20 seconds |
| Token speed | 10-40 tok/s | 5-10 tok/s |
| Memory | ~900 MB VRAM | ~1.2 GB RAM |

### Pipeline Timing (2-cycle run, WebGPU)

| Phase | Estimated Time |
|---|---|
| Horizon Mapping | 10-30 seconds |
| Dialectic Cycle (per cycle) | 15-40 seconds |
| Iteration Control | 5-15 seconds |
| Horizon Fusion | 15-40 seconds |
| **Total (2 cycles)** | **~60-170 seconds** |

---

## Model Information

| Property | Value |
|---|---|
| Model | [Qwen3.5-0.8B](https://huggingface.co/Qwen/Qwen3.5-0.8B) |
| ONNX source | [onnx-community/Qwen3.5-0.8B-ONNX](https://huggingface.co/onnx-community/Qwen3.5-0.8B-ONNX) |
| Quantization | Q4 (4-bit weights) with FP16 vision encoder |
| Parameters | 0.8 billion |
| Context window | 32,768 tokens |
| Model class | `Qwen3_5ForConditionalGeneration` + `AutoProcessor` |
| License | [Apache 2.0](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/LICENSE) |

---

## Tech Stack

| Layer | Technology |
|---|---|
| Desktop shell | [Tauri v2](https://tauri.app/) (Rust) |
| Frontend | React 19 + TypeScript 5.9 |
| Build tool | Vite 7 |
| LLM runtime | [`@huggingface/transformers`](https://github.com/huggingface/transformers.js) v4 (next) |
| Inference engine | [ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript/web.html) |
| Execution providers | WebGPU (primary) + WASM (fallback) |
| State management | [Zustand](https://zustand-demo.pmnd.rs/) 5 + Immer |
| Styling | Tailwind CSS 3 |
| Icons | FontAwesome 7 |
| Installer | MSI via WiX (Tauri bundler) |

---

## Troubleshooting

**QwenStatusBar shows "Model error"**
Check the browser DevTools console (right-click the Tauri window, Inspect) for details. Common causes: insufficient VRAM, WebGPU not available, or interrupted model download. Clear the browser cache and retry.

**The app shows WASM instead of WebGPU**
Your GPU or driver does not support WebGPU. The app still works via WASM but at lower speed (~5-10 tok/s). Update your GPU drivers to the latest version.

**"Qwen provider not ready" error when submitting a query**
The model is still loading. Wait for the QwenStatusBar to show a green dot and "ready" status before submitting.

**JSON parsing errors in pipeline output**
The 0.8B model occasionally produces malformed JSON. The agents have built-in regex fallback parsers that recover gracefully in most cases. If errors persist, try switching to the OpenRouter provider with a larger model.

**SmartScreen warning on MSI install**
The MSI is not code-signed in development builds. Click "More info -> Run anyway" to proceed.

**OpenRouter API errors**
Verify your API key is correct in Settings. Check your OpenRouter account for rate limits or billing issues.

---

## Licence

This project is released under the **MIT Licence**. See [LICENSE](./LICENSE) for details.

The bundled model weights are governed by the [Apache 2.0 licence](https://huggingface.co/Qwen/Qwen3.5-0.8B/blob/main/LICENSE) from Alibaba Cloud / Qwen Team.

---

## Acknowledgements

- [Qwen Team @ Alibaba Cloud](https://github.com/QwenLM) — for the Qwen3.5 model family
- [Xenova / onnx-community](https://huggingface.co/onnx-community) — for the ONNX-converted and quantized model weights
- [Hugging Face Transformers.js](https://github.com/huggingface/transformers.js) — for the ORT-backed JavaScript inference library
- [Tauri](https://tauri.app/) — for making lightweight, secure desktop apps with web frontends practical
- [Fusion Horizon Orchestrator](https://github.com/MushroomFleet/Fusion-Horizon-Orchestrator) — the original Python hermeneutic pipeline

---

## Citation

```bibtex
@software{fusion_horizon_qwenort,
  title = {Fusion-Horizon-QwenORT: Hermeneutic interpretation engine with built-in local LLM inference},
  author = {Drift Johnson},
  year = {2026},
  url = {https://github.com/MushroomFleet/Fusion-Horizon-QwenORT},
  version = {1.0.0}
}
```

### Donate

[![Ko-Fi](https://cdn.ko-fi.com/cdn/kofi3.png?v=3)](https://ko-fi.com/driftjohnson)
