# Qwen3.5-0.8B ONNX Headless Inference Integration

## TINS Metadata

| Field             | Value                                                        |
|-------------------|--------------------------------------------------------------|
| **Document**      | TINS (Technical Implementation & Novelty Specification)      |
| **Version**       | 1.0.0                                                        |
| **Date**          | 2026-03-29                                                   |
| **Model**         | `onnx-community/Qwen3.5-0.8B-ONNX`                          |
| **Runtime**       | ONNX Runtime Web (WebGPU primary, WASM fallback)             |
| **Framework**     | `@huggingface/transformers` v4 (`@next` tag)                 |
| **Host Stack**    | Tauri v2 + React + TypeScript + Vite                         |
| **Type**          | Headless integration (no UI provided)                        |
| **Reference Impl**| `Qwen35-08B-onnx-desktop-dev` (verified working)            |

---

## 1. Description

This document specifies a **headless** integration of local Qwen3.5-0.8B ONNX inference into any existing React/TypeScript/Vite/Tauri v2 desktop application. It is NOT a standalone app. It is a drop-in inference engine.

### What this provides

- A **Web Worker** (`qwen.worker.ts`) that loads the model and runs streaming text generation
- A **React hook** (`useQwen.ts`) that manages the worker lifecycle and exposes a clean API
- A **TypeScript types file** (`qwen.types.ts`) defining the complete worker protocol and hook contract
- A **thinking block parser** (`parseThinking.ts`) for extracting chain-of-thought reasoning
- Required **Tauri configuration changes** (CSP, capabilities, Rust plugins)
- Required **Vite configuration changes** (ORT WASM binaries, worker format, build target)
- Required **npm dependencies**

### What this does NOT provide

- No chat UI components (no message bubbles, no input fields, no sidebars)
- No chat history store (the consumer manages their own conversation state)
- No settings UI (the consumer passes generation parameters directly)
- No model download UI (the consumer reads progress from the hook and renders their own)
- No persistence layer (no localStorage, no Zustand stores for chat)

The consumer builds their own UI. The hook provides state. That is the contract.

---

## 2. Functionality

### Model Loading with WebGPU/WASM Fallback

The model loads via `@huggingface/transformers` v4 using `Qwen3_5ForConditionalGeneration` (a vision-language model class). On first load, the library downloads model files from HuggingFace CDN and caches them via the browser Cache API. On subsequent loads, it reads from cache. The model attempts WebGPU first. If the GPU is unavailable, ONNX Runtime falls back to WASM automatically.

### Streaming Text Generation

Tokens are streamed one at a time via `TextStreamer` with `skip_special_tokens: true`. Each token is posted from the worker to the main thread as a `{ type: "token", payload: string }` message. The hook accumulates tokens into `streamingText` and provides an optional `onToken` callback for per-token processing.

### Thinking Mode (Enable/Disable Chain-of-Thought)

When `thinkingEnabled` is `true`, the system prompt instructs the model to "Think step by step before answering." The model wraps its reasoning in `<think>...</think>` blocks. When `thinkingEnabled` is `false`, the system prompt includes `/no_think` to suppress reasoning output.

### Generation Abort

The consumer can call `abort()` at any time during generation. This triggers `InterruptableStoppingCriteria.interrupt()` in the worker, which halts token generation. Partial text is preserved in `streamingText`.

### Progress Reporting During Model Download

During first-time model download, the worker posts `load-progress` messages containing `{ file, loaded, total, progress, status }`. The hook converts these into a `QwenDownloadProgress` object with a computed `percent` field.

### Configurable Generation Parameters

All generation parameters are exposed:

| Parameter           | Range       | Default | Description                        |
|---------------------|-------------|---------|------------------------------------|
| `maxNewTokens`      | 1-2048      | 512     | Maximum tokens to generate         |
| `temperature`       | 0.0-2.0     | 0.7     | Sampling temperature               |
| `topP`              | 0.1-1.0     | 0.9     | Nucleus sampling threshold         |
| `repetitionPenalty` | 1.0-2.0     | 1.1     | Penalty for repeated tokens        |
| `thinkingEnabled`   | boolean     | true    | Enable chain-of-thought reasoning  |

### Token Throughput Measurement

The hook tracks generation start time and token count, computing a live `tokensPerSecond` value updated with each token.

---

## 3. Technical Implementation

### 3.1 Prerequisites

The host project must already have:

- **Tauri v2** (`@tauri-apps/cli` ^2.0.0, `tauri` crate v2)
- **React** (^18.0.0 or ^19.0.0)
- **TypeScript** (^5.0.0)
- **Vite** (^5.0.0 or ^6.0.0)
- A working `tauri dev` / `tauri build` pipeline
- An existing `src-tauri/` directory with `tauri.conf.json` and `capabilities/default.json`

### 3.2 Dependencies to Add

```bash
npm install @huggingface/transformers@next onnxruntime-web
npm install -D vite-plugin-static-copy
```

The `@next` tag is mandatory. Version 4.x of `@huggingface/transformers` contains `Qwen3_5ForConditionalGeneration`. Version 3.x does not have this class and will fail at import.

Optional (only if the host project does not already have Tailwind v4):

```bash
npm install tailwindcss@^4.0.0
npm install -D @tailwindcss/vite@^4.2.2
```

### 3.3 Vite Configuration Changes

The host project's `vite.config.ts` must be updated with the following additions. This is not a replacement -- merge these into the existing config.

```typescript
import { viteStaticCopy } from "vite-plugin-static-copy";

export default defineConfig({
  plugins: [
    // ... existing plugins ...

    // Copy ORT WASM binaries so the browser can fetch them at runtime
    viteStaticCopy({
      targets: [
        {
          src: "node_modules/onnxruntime-web/dist/*.wasm",
          dest: "ort-wasm",
        },
      ],
    }),
  ],

  // Exclude @huggingface/transformers from Vite's dependency pre-bundling.
  // The library uses top-level await and dynamic imports that break pre-bundling.
  optimizeDeps: {
    exclude: ["@huggingface/transformers"],
  },

  // Required for top-level await support
  build: {
    target: "esnext",
    rollupOptions: {
      output: {
        format: "esm",
      },
    },
  },

  // Web Workers must use ES module format for dynamic imports to work
  worker: {
    format: "es",
  },
});
```

### 3.4 Tauri Configuration Changes

#### 3.4.1 `src-tauri/tauri.conf.json`

Add the following to the `app.security.csp` string. This allows the WebView to fetch model files from HuggingFace CDN:

```
connect-src 'self' https://huggingface.co https://*.huggingface.co https://cdn-lfs.huggingface.co https://*.hf.co blob: asset: http://asset.localhost;
worker-src 'self' blob:;
```

A complete CSP example:

```json
{
  "app": {
    "security": {
      "csp": "default-src 'self'; script-src 'self' 'unsafe-inline' 'unsafe-eval' blob:; style-src 'self' 'unsafe-inline'; connect-src 'self' https://huggingface.co https://*.huggingface.co https://cdn-lfs.huggingface.co https://*.hf.co blob: asset: http://asset.localhost; worker-src 'self' blob:; img-src 'self' asset: http://asset.localhost"
    }
  }
}
```

The `plugins` key must be an empty object:

```json
{
  "plugins": {}
}
```

#### 3.4.2 `src-tauri/capabilities/default.json`

Add HTTP permissions for HuggingFace domains so the WebView can download model files:

```json
{
  "identifier": "default",
  "description": "Default capabilities for the main window",
  "windows": ["main"],
  "permissions": [
    "core:default",
    "core:event:default",
    "core:event:allow-emit",
    "core:event:allow-listen",
    "core:path:default",
    {
      "identifier": "http:default",
      "allow": [
        { "url": "https://huggingface.co/**" },
        { "url": "https://*.huggingface.co/**" },
        { "url": "https://*.hf.co/**" }
      ]
    }
  ]
}
```

Note: If the host project already has a `capabilities/default.json`, merge the `http:default` permission block into the existing `permissions` array. Do not replace the entire file.

#### 3.4.3 `src-tauri/Cargo.toml`

The Rust side requires these Tauri plugins as dependencies:

```toml
[dependencies]
tauri = { version = "2", features = ["protocol-asset"] }
tauri-plugin-http = "2"
```

And in `src-tauri/src/lib.rs` (or `main.rs`), register the plugin:

```rust
fn main() {
    tauri::Builder::default()
        .plugin(tauri_plugin_http::init())
        // ... other plugins ...
        .run(tauri::generate_context!())
        .expect("error while running tauri application");
}
```

### 3.5 ORT Bootstrap (in App Entry Point)

Add this to the host project's entry point (e.g., `main.tsx` or `index.tsx`) BEFORE rendering:

```typescript
import * as ort from "onnxruntime-web";

// Tell ORT where to find the WASM binaries (copied by viteStaticCopy)
ort.env.wasm.wasmPaths = "/ort-wasm/";

// Enable WebGPU with high-performance power preference
(ort.env as any).webgpu = { powerPreference: "high-performance" };
```

This must execute before any component that uses the `useQwen` hook mounts.

### 3.6 CRITICAL WARNINGS

These are hard-won lessons from the reference implementation. Each one caused real build failures or runtime errors.

#### WARNING 1: Tailwind v4 requires the `@tailwindcss/vite` plugin

Tailwind CSS v4 does NOT use `postcss` or `tailwind.config.js`. It requires the `@tailwindcss/vite` plugin registered in `vite.config.ts`. If the host project uses Tailwind v4, ensure the plugin is registered:

```typescript
import tailwindcss from "@tailwindcss/vite";

export default defineConfig({
  plugins: [tailwindcss()],
});
```

#### WARNING 2: `vite-plugin-static-copy` must use RELATIVE paths

On Windows, using `path.resolve()` for the `src` field in `viteStaticCopy` targets produces backslash paths that break glob matching. Always use a forward-slash relative path:

```typescript
// CORRECT
src: "node_modules/onnxruntime-web/dist/*.wasm"

// WRONG - breaks on Windows
src: path.resolve(__dirname, "node_modules/onnxruntime-web/dist/*.wasm")
```

#### WARNING 3: `ort.env.webgpu` is readonly -- must cast to `any`

The TypeScript types for `onnxruntime-web` mark `env.webgpu` as readonly. Direct assignment fails type checking. Cast to `any`:

```typescript
// CORRECT
(ort.env as any).webgpu = { powerPreference: "high-performance" };

// WRONG - TypeScript error
ort.env.webgpu = { powerPreference: "high-performance" };
```

#### WARNING 4: Tauri v2 uses the capabilities system, not plugins config

Tauri v2 replaced the v1 plugin allowlist with a capabilities system. Permissions go in `src-tauri/capabilities/default.json`, not in `tauri.conf.json` under a `plugins` key. The `plugins` key in `tauri.conf.json` must be `{}` (empty object).

#### WARNING 5: Permission naming follows a strict convention

Tauri v2 permission identifiers use a specific format:

- `core:path:default` (not `path:default`)
- `fs:allow-appcache-read-recursive` (not `fs:allow-app-cache-read-recursive`)
- `http:default` with URL allow patterns (not `http:allow-fetch`)

Incorrect permission names cause silent failures or build errors.

#### WARNING 6: There is no `tauri-plugin-path` crate

The `core:path:default` permission is built into Tauri v2 core. Do not add `tauri-plugin-path` to `Cargo.toml` -- it does not exist and will cause a build failure.

#### WARNING 7: CSP must allow HuggingFace CDN wildcards

The model files are served from multiple HuggingFace subdomains. All of these must be in `connect-src`:

- `https://huggingface.co` -- API and model index
- `https://*.huggingface.co` -- CDN subdomains
- `https://cdn-lfs.huggingface.co` -- Large file storage (model weights)
- `https://*.hf.co` -- Short URL domain

Missing any of these causes `fetch()` failures during model download with no clear error message.

#### WARNING 8: `@huggingface/transformers` `@next` tag is mandatory

The stable release (v3.x) does not include `Qwen3_5ForConditionalGeneration`. Only the `@next` tag (v4.x) has this class. Installing without the tag:

```bash
# WRONG - installs v3.x which lacks the model class
npm install @huggingface/transformers

# CORRECT - installs v4.x with Qwen3.5 support
npm install @huggingface/transformers@next
```

#### WARNING 9: `skip_special_tokens: true` is required on TextStreamer

Without this flag, the streamed output includes raw special tokens like `<|im_end|>` at the end of the response. These appear as visible garbage in the UI:

```typescript
// CORRECT
const streamer = new TextStreamer(processor.tokenizer, {
  skip_prompt: true,
  skip_special_tokens: true,
  callback_function: (token: string) => { /* ... */ },
});

// WRONG - <|im_end|> appears in output
const streamer = new TextStreamer(processor.tokenizer, {
  skip_prompt: true,
  callback_function: (token: string) => { /* ... */ },
});
```

#### WARNING 10: This is a vision-language model, not a text-only model

`Qwen3.5-0.8B-ONNX` uses `Qwen3_5ForConditionalGeneration` + `AutoProcessor`, NOT `AutoModelForCausalLM` + `AutoTokenizer`. Using the wrong classes will fail:

```typescript
// CORRECT
import { AutoProcessor, Qwen3_5ForConditionalGeneration } from "@huggingface/transformers";

// WRONG - these are for text-only models
import { AutoModelForCausalLM, AutoTokenizer } from "@huggingface/transformers";
```

The dtype config must include `vision_encoder`:

```typescript
dtype: {
  embed_tokens: "q4",
  vision_encoder: "fp16",
  decoder_model_merged: "q4",
}
```

#### WARNING 11: Message content format uses array-of-objects

When building messages for the processor's `apply_chat_template`, user/assistant message content must be wrapped in `[{ type: "text", text: content }]` format, not passed as a plain string. The system message is an exception and uses a plain string.

```typescript
// CORRECT
const messages = [
  { role: "system", content: systemPrompt },
  ...req.messages.map((m) => ({
    role: m.role,
    content: [{ type: "text" as const, text: m.content }],
  })),
];

// WRONG - processor expects array content for non-system messages
const messages = [
  { role: "system", content: systemPrompt },
  ...req.messages,
];
```

---

### 3.7 File: `qwen.types.ts`

Copy this file into the host project. All other files import types from it.

```typescript
// ─── Model Lifecycle ────────────────────────────────────────────────────────

/** Phases the model goes through from cold start to ready. */
export type QwenPhase = "idle" | "loading" | "downloading" | "ready" | "error";

/** Progress information during model file download. */
export interface QwenDownloadProgress {
  file: string;
  loaded: number;
  total: number;
  percent: number;
}

// ─── Generation Settings ────────────────────────────────────────────────────

/** All configurable generation parameters. */
export interface QwenGenerationSettings {
  /** Maximum new tokens to generate. Range: 1-2048. Default: 512. */
  maxNewTokens: number;
  /** Sampling temperature. Range: 0.0-2.0. Default: 0.7. 0 = greedy. */
  temperature: number;
  /** Nucleus sampling threshold. Range: 0.1-1.0. Default: 0.9. */
  topP: number;
  /** Penalty for repeated tokens. Range: 1.0-2.0. Default: 1.1. */
  repetitionPenalty: number;
  /** Enable chain-of-thought reasoning in <think> blocks. Default: true. */
  thinkingEnabled: boolean;
}

// ─── Message Types ──────────────────────────────────────────────────────────

/** A single message in the conversation history. */
export interface QwenMessage {
  role: "user" | "assistant";
  content: string;
}

/** Parsed response with optional thinking block separated from the answer. */
export interface QwenParsedResponse {
  /** Extracted content from <think>...</think> blocks. Empty string if none. */
  thinking: string;
  /** Main response text with thinking blocks removed. */
  answer: string;
}

// ─── Worker Protocol ────────────────────────────────────────────────────────

/** Messages sent TO the worker from the main thread. */
export type QwenWorkerInMessage =
  | { type: "init" }
  | {
      type: "generate";
      payload: {
        messages: QwenMessage[];
        settings: QwenGenerationSettings;
      };
    }
  | { type: "abort" };

/** Messages sent FROM the worker to the main thread. */
export type QwenWorkerOutMessage =
  | { type: "status"; payload: string }
  | {
      type: "load-progress";
      payload: {
        file: string;
        loaded: number;
        total: number;
        progress: number;
        status: string;
      };
    }
  | { type: "ready"; payload: { device: string } }
  | { type: "token"; payload: string }
  | { type: "done" }
  | { type: "error"; payload: string };

// ─── Hook Return Type ───────────────────────────────────────────────────────

/** Everything the useQwen hook exposes to the consumer. */
export interface UseQwenReturn {
  /** Current model lifecycle phase. */
  phase: QwenPhase;
  /** Detected compute device: "webgpu" | "wasm" | null (before ready). */
  device: string | null;
  /** Download progress during model fetch. Null when not downloading. */
  downloadProgress: QwenDownloadProgress | null;
  /** Human-readable status message from the worker. */
  statusMessage: string;
  /** Error message if phase is "error". Null otherwise. */
  error: string | null;
  /** Live tokens-per-second during generation. */
  tokensPerSecond: number;
  /** True while the model is generating tokens. */
  isGenerating: boolean;

  // Actions

  /** Start generating a response. No-op if model is not ready or already generating. */
  generate: (
    messages: QwenMessage[],
    settings?: Partial<QwenGenerationSettings>
  ) => void;
  /** Abort the current generation. Partial text is preserved. */
  abort: () => void;

  // Streaming state

  /** Accumulated text during generation. Reset on each new generate() call. */
  streamingText: string;
  /** Optional callback invoked for each token. Set this to process tokens individually. */
  onToken?: (token: string) => void;
}
```

---

### 3.8 File: `qwen.worker.ts`

This is the complete Web Worker. It runs in a separate thread and handles all model operations. Copy it into the host project alongside the types file.

```typescript
import {
  env,
  AutoProcessor,
  Qwen3_5ForConditionalGeneration,
  TextStreamer,
  InterruptableStoppingCriteria,
} from "@huggingface/transformers";

// Let the library handle its own downloading and caching via browser Cache API
env.allowLocalModels = false;
env.allowRemoteModels = true;

const MODEL_ID = "onnx-community/Qwen3.5-0.8B-ONNX";

let model: any = null;
let processor: any = null;
let stopCriteria = new InterruptableStoppingCriteria();

// ─── Message Protocol ───────────────────────────────────────────────────────

type WorkerInMessage =
  | { type: "init" }
  | { type: "generate"; payload: GenerateRequest }
  | { type: "abort" };

type GenerateRequest = {
  messages: { role: string; content: string }[];
  settings: {
    maxNewTokens: number;
    temperature: number;
    topP: number;
    repetitionPenalty: number;
    thinkingEnabled: boolean;
  };
};

// ─── Progress reporting ─────────────────────────────────────────────────────

function progressCallback(progress: any) {
  if (progress.status === "download" || progress.status === "progress") {
    postMessage({
      type: "load-progress",
      payload: {
        file: progress.file ?? "",
        loaded: progress.loaded ?? 0,
        total: progress.total ?? 0,
        progress: progress.progress ?? 0,
        status: progress.status,
      },
    });
  } else if (progress.status === "ready") {
    // Individual file ready — no action needed
  } else {
    postMessage({
      type: "status",
      payload: `${progress.status}: ${progress.file ?? ""}`,
    });
  }
}

// ─── Init ───────────────────────────────────────────────────────────────────

async function init() {
  postMessage({ type: "status", payload: "Loading processor\u2026" });

  processor = await AutoProcessor.from_pretrained(MODEL_ID, {
    progress_callback: progressCallback,
  });

  postMessage({
    type: "status",
    payload: "Loading model (this may take a moment)\u2026",
  });

  model = await Qwen3_5ForConditionalGeneration.from_pretrained(MODEL_ID, {
    dtype: {
      embed_tokens: "q4",
      vision_encoder: "fp16",
      decoder_model_merged: "q4",
    },
    device: "webgpu",
    progress_callback: progressCallback,
  });

  postMessage({
    type: "ready",
    payload: { device: (model as any).device ?? "wasm" },
  });
}

// ─── Generate ───────────────────────────────────────────────────────────────

async function generate(req: GenerateRequest) {
  if (!model || !processor) return;

  stopCriteria = new InterruptableStoppingCriteria();

  const systemPrompt = req.settings.thinkingEnabled
    ? "You are a helpful assistant. Think step by step before answering."
    : "You are a helpful assistant. /no_think";

  const messages = [
    { role: "system", content: systemPrompt },
    ...req.messages.map((m) => ({
      role: m.role,
      content: [{ type: "text" as const, text: m.content }],
    })),
  ];

  const text = processor.apply_chat_template(messages, {
    add_generation_prompt: true,
  });

  const inputs = await processor(text);

  const streamer = new TextStreamer(processor.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (token: string) => {
      postMessage({ type: "token", payload: token });
    },
  });

  await model.generate({
    ...inputs,
    max_new_tokens: req.settings.maxNewTokens,
    temperature: req.settings.temperature,
    top_p: req.settings.topP,
    repetition_penalty: req.settings.repetitionPenalty,
    do_sample: req.settings.temperature > 0,
    streamer,
    stopping_criteria: [stopCriteria],
  });

  postMessage({ type: "done" });
}

// ─── Message Handler ────────────────────────────────────────────────────────

self.addEventListener("message", async (e: MessageEvent<WorkerInMessage>) => {
  const msg = e.data;
  switch (msg.type) {
    case "init":
      await init().catch((err) =>
        postMessage({ type: "error", payload: String(err) })
      );
      break;
    case "generate":
      await generate(msg.payload).catch((err) =>
        postMessage({ type: "error", payload: String(err) })
      );
      break;
    case "abort":
      stopCriteria.interrupt();
      break;
  }
});
```

---

### 3.9 File: `parseThinking.ts`

Utility to separate `<think>...</think>` reasoning blocks from the main response text.

```typescript
/**
 * Separates <think>...</think> content from the main response.
 * Returns { thinking, answer } where either may be empty string.
 */
export function parseThinkingBlocks(raw: string): {
  thinking: string;
  answer: string;
} {
  const thinkRegex = /<think>([\s\S]*?)<\/think>/g;
  let thinking = "";
  let answer = raw;

  const match = thinkRegex.exec(raw);
  if (match) {
    thinking = match[1].trim();
    answer = raw.replace(/<think>[\s\S]*?<\/think>/g, "").trim();
  }

  return { thinking, answer };
}
```

---

### 3.10 File: `useQwen.ts`

The React hook that wraps the worker with clean state management. No external state library required (uses React built-in hooks only).

```typescript
import { useState, useEffect, useRef, useCallback } from "react";
import type {
  QwenPhase,
  QwenDownloadProgress,
  QwenMessage,
  QwenGenerationSettings,
  UseQwenReturn,
} from "./qwen.types";

const DEFAULT_SETTINGS: QwenGenerationSettings = {
  maxNewTokens: 512,
  temperature: 0.7,
  topP: 0.9,
  repetitionPenalty: 1.1,
  thinkingEnabled: true,
};

// Singleton worker instance — shared across all components that call useQwen.
// The model loads once and persists for the lifetime of the application.
let workerInstance: Worker | null = null;

function getWorker(): Worker {
  if (!workerInstance) {
    workerInstance = new Worker(
      new URL("./qwen.worker.ts", import.meta.url),
      { type: "module" }
    );
  }
  return workerInstance;
}

/**
 * React hook for Qwen3.5-0.8B local inference.
 *
 * @param autoInit - If true (default), the model starts loading on mount.
 *                   Set to false to defer loading until the consumer is ready.
 *
 * @returns UseQwenReturn - All state and actions needed for inference.
 */
export function useQwen(autoInit = true): UseQwenReturn {
  const [phase, setPhase] = useState<QwenPhase>("idle");
  const [device, setDevice] = useState<string | null>(null);
  const [downloadProgress, setDownloadProgress] =
    useState<QwenDownloadProgress | null>(null);
  const [statusMessage, setStatusMessage] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [tokensPerSecond, setTPS] = useState(0);
  const [isGenerating, setIsGenerating] = useState(false);
  const [streamingText, setStreamingText] = useState("");

  const tokenCountRef = useRef(0);
  const genStartRef = useRef(0);
  const onTokenRef = useRef<((token: string) => void) | undefined>();
  const isGeneratingRef = useRef(false);

  useEffect(() => {
    const worker = getWorker();

    const handler = (e: MessageEvent) => {
      const msg = e.data;
      switch (msg.type) {
        case "status":
          setStatusMessage(msg.payload);
          break;

        case "load-progress":
          if (msg.payload.total > 0) {
            setPhase("downloading");
            setDownloadProgress({
              file: msg.payload.file,
              loaded: msg.payload.loaded,
              total: msg.payload.total,
              percent: Math.round(
                (msg.payload.loaded / msg.payload.total) * 100
              ),
            });
          }
          break;

        case "ready":
          setDevice(msg.payload.device);
          setPhase("ready");
          break;

        case "token":
          setStreamingText((prev) => prev + msg.payload);
          tokenCountRef.current++;
          const elapsed =
            (Date.now() - genStartRef.current) / 1000;
          if (elapsed > 0) {
            setTPS(
              Math.round(tokenCountRef.current / elapsed)
            );
          }
          onTokenRef.current?.(msg.payload);
          break;

        case "done":
          setIsGenerating(false);
          isGeneratingRef.current = false;
          break;

        case "error":
          if (isGeneratingRef.current) {
            setStreamingText(
              (prev) => prev + `\n\n[Error: ${msg.payload}]`
            );
            setIsGenerating(false);
            isGeneratingRef.current = false;
          } else {
            setError(msg.payload);
            setPhase("error");
          }
          break;
      }
    };

    worker.addEventListener("message", handler);

    if (autoInit) {
      setPhase("loading");
      worker.postMessage({ type: "init" });
    }

    return () => worker.removeEventListener("message", handler);
  }, []);

  const generate = useCallback(
    (
      messages: QwenMessage[],
      settings?: Partial<QwenGenerationSettings>
    ) => {
      if (phase !== "ready" || isGenerating) return;

      const merged = { ...DEFAULT_SETTINGS, ...settings };
      setStreamingText("");
      setIsGenerating(true);
      isGeneratingRef.current = true;
      tokenCountRef.current = 0;
      genStartRef.current = Date.now();

      getWorker().postMessage({
        type: "generate",
        payload: { messages, settings: merged },
      });
    },
    [phase, isGenerating]
  );

  const abort = useCallback(() => {
    getWorker().postMessage({ type: "abort" });
  }, []);

  return {
    phase,
    device,
    downloadProgress,
    statusMessage,
    error,
    tokensPerSecond,
    isGenerating,
    streamingText,
    generate,
    abort,
    set onToken(cb: ((token: string) => void) | undefined) {
      onTokenRef.current = cb;
    },
    get onToken() {
      return onTokenRef.current;
    },
  };
}
```

---

## 4. Integration Guide

Step-by-step instructions for adding Qwen inference to an existing Tauri v2 + React + TypeScript + Vite project.

### Step 1: Install Dependencies

```bash
npm install @huggingface/transformers@next onnxruntime-web
npm install -D vite-plugin-static-copy
```

### Step 2: Update `vite.config.ts`

Add the `viteStaticCopy` plugin, exclude `@huggingface/transformers` from `optimizeDeps`, set `build.target` to `"esnext"`, and set `worker.format` to `"es"`. See Section 3.3 for the exact config.

### Step 3: Update Tauri CSP

In `src-tauri/tauri.conf.json`, add the HuggingFace CDN domains to `connect-src` and add `worker-src 'self' blob:`. See Section 3.4.1 for the complete CSP string.

### Step 4: Update Capabilities

In `src-tauri/capabilities/default.json`, add the `http:default` permission with HuggingFace URL patterns. See Section 3.4.2 for the exact JSON.

### Step 5: Add Rust Dependencies

In `src-tauri/Cargo.toml`, ensure `tauri-plugin-http = "2"` is listed and `tauri` has the `"protocol-asset"` feature. Register the HTTP plugin in `lib.rs`. See Section 3.4.3.

### Step 6: Add ORT Bootstrap to Entry Point

In the host project's entry point (e.g., `main.tsx`), add the ORT WASM path and WebGPU configuration BEFORE `ReactDOM.createRoot()`. See Section 3.5.

### Step 7: Copy the 4 Files Into the Project

Copy these files into a directory in the host project (e.g., `src/qwen/` or `src/lib/qwen/`):

```
qwen.types.ts
qwen.worker.ts
parseThinking.ts
useQwen.ts
```

Ensure the import paths in `useQwen.ts` match the file locations:

- `useQwen.ts` imports from `"./qwen.types"`
- `useQwen.ts` references `"./qwen.worker.ts"` via `new URL()`

### Step 8: Use the Hook in Any Component

```typescript
import { useQwen } from "./qwen/useQwen";
import { parseThinkingBlocks } from "./qwen/parseThinking";

function MyComponent() {
  const qwen = useQwen();

  // Show loading state
  if (qwen.phase === "loading") return <div>Loading model...</div>;
  if (qwen.phase === "downloading") {
    return (
      <div>
        Downloading: {qwen.downloadProgress?.file} -{" "}
        {qwen.downloadProgress?.percent}%
      </div>
    );
  }
  if (qwen.phase === "error") return <div>Error: {qwen.error}</div>;

  // Send a message
  function handleSend(text: string) {
    qwen.generate([{ role: "user", content: text }], {
      temperature: 0.7,
      thinkingEnabled: true,
    });
  }

  // Parse thinking blocks from the streaming response
  const { thinking, answer } = parseThinkingBlocks(qwen.streamingText);

  // Abort generation
  function handleStop() {
    qwen.abort();
  }

  // Access metadata
  console.log(qwen.device); // "webgpu" or "wasm"
  console.log(qwen.tokensPerSecond); // live tok/s
  console.log(qwen.isGenerating); // true during inference

  return (
    <div>
      {/* Consumer builds their own UI here */}
    </div>
  );
}
```

---

## 5. Integration Patterns

### Pattern A: Chat Assistant Sidebar

A collapsible sidebar in an existing app that provides a local AI chat interface.

```typescript
import { useState } from "react";
import { useQwen } from "./qwen/useQwen";
import { parseThinkingBlocks } from "./qwen/parseThinking";
import type { QwenMessage } from "./qwen/qwen.types";

interface ChatEntry {
  role: "user" | "assistant";
  content: string;
  thinking?: string;
}

function AISidebar({ isOpen }: { isOpen: boolean }) {
  const qwen = useQwen();
  const [history, setHistory] = useState<ChatEntry[]>([]);
  const [input, setInput] = useState("");

  function handleSend() {
    if (!input.trim() || qwen.phase !== "ready" || qwen.isGenerating) return;

    const userEntry: ChatEntry = { role: "user", content: input.trim() };
    const updatedHistory = [...history, userEntry];
    setHistory(updatedHistory);
    setInput("");

    // Build messages array for the model (full conversation history)
    const messages: QwenMessage[] = updatedHistory.map((e) => ({
      role: e.role,
      content: e.content,
    }));

    qwen.generate(messages, { thinkingEnabled: true });
  }

  // When generation finishes, commit the response to history
  const wasGenerating = useRef(false);
  useEffect(() => {
    if (wasGenerating.current && !qwen.isGenerating && qwen.streamingText) {
      const { thinking, answer } = parseThinkingBlocks(qwen.streamingText);
      setHistory((prev) => [
        ...prev,
        { role: "assistant", content: answer, thinking },
      ]);
    }
    wasGenerating.current = qwen.isGenerating;
  }, [qwen.isGenerating, qwen.streamingText]);

  if (!isOpen) return null;

  return (
    <aside style={{ width: 360, borderLeft: "1px solid #ccc", padding: 16 }}>
      <h3>AI Assistant ({qwen.device ?? "loading..."})</h3>

      {qwen.phase === "downloading" && (
        <div>Downloading model: {qwen.downloadProgress?.percent}%</div>
      )}
      {qwen.phase === "loading" && <div>{qwen.statusMessage}</div>}
      {qwen.phase === "error" && <div style={{ color: "red" }}>{qwen.error}</div>}

      <div style={{ flex: 1, overflowY: "auto" }}>
        {history.map((entry, i) => (
          <div key={i} style={{ marginBottom: 8 }}>
            <strong>{entry.role}:</strong> {entry.content}
            {entry.thinking && (
              <details>
                <summary>Reasoning</summary>
                <pre style={{ fontSize: 12 }}>{entry.thinking}</pre>
              </details>
            )}
          </div>
        ))}

        {qwen.isGenerating && (
          <div style={{ marginBottom: 8 }}>
            <strong>assistant:</strong>{" "}
            {parseThinkingBlocks(qwen.streamingText).answer}
            <span style={{ opacity: 0.5 }}> ({qwen.tokensPerSecond} tok/s)</span>
          </div>
        )}
      </div>

      <div style={{ display: "flex", gap: 8 }}>
        <input
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSend()}
          placeholder="Ask something..."
          disabled={qwen.phase !== "ready"}
          style={{ flex: 1 }}
        />
        {qwen.isGenerating ? (
          <button onClick={() => qwen.abort()}>Stop</button>
        ) : (
          <button onClick={handleSend} disabled={qwen.phase !== "ready"}>
            Send
          </button>
        )}
      </div>
    </aside>
  );
}
```

Note: Add the missing imports (`useRef`, `useEffect`) from React when using this pattern.

### Pattern B: Inline Code Reviewer

An app that sends code to the model and displays analysis inline.

```typescript
import { useQwen } from "./qwen/useQwen";
import { parseThinkingBlocks } from "./qwen/parseThinking";

function CodeReviewer({ code }: { code: string }) {
  const qwen = useQwen();

  function analyzeCode() {
    qwen.generate(
      [
        {
          role: "user",
          content: `Review this code for bugs, performance issues, and improvements. Be concise.\n\n\`\`\`\n${code}\n\`\`\``,
        },
      ],
      {
        maxNewTokens: 1024,
        temperature: 0.3, // Lower temperature for more focused analysis
        thinkingEnabled: true,
      }
    );
  }

  const { thinking, answer } = parseThinkingBlocks(qwen.streamingText);

  return (
    <div>
      <button
        onClick={analyzeCode}
        disabled={qwen.phase !== "ready" || qwen.isGenerating}
      >
        {qwen.isGenerating ? "Analyzing..." : "Review Code"}
      </button>

      {qwen.isGenerating && (
        <div style={{ fontSize: 12, color: "#888" }}>
          {qwen.tokensPerSecond} tok/s
        </div>
      )}

      {thinking && (
        <details style={{ marginTop: 8 }}>
          <summary>Model reasoning</summary>
          <pre style={{ fontSize: 12, background: "#f5f5f5", padding: 8 }}>
            {thinking}
          </pre>
        </details>
      )}

      {answer && (
        <div style={{ marginTop: 8, whiteSpace: "pre-wrap" }}>{answer}</div>
      )}
    </div>
  );
}
```

### Pattern C: Document Summarizer

An app that processes document text through the model for summaries.

```typescript
import { useState } from "react";
import { useQwen } from "./qwen/useQwen";
import { parseThinkingBlocks } from "./qwen/parseThinking";

function DocumentSummarizer() {
  const qwen = useQwen();
  const [documentText, setDocumentText] = useState("");

  function summarize() {
    if (!documentText.trim()) return;

    // Truncate to roughly 2000 characters to stay within context limits
    const truncated = documentText.slice(0, 2000);

    qwen.generate(
      [
        {
          role: "user",
          content: `Summarize the following document in 3-5 bullet points. Be concise and capture the key points.\n\n---\n${truncated}\n---`,
        },
      ],
      {
        maxNewTokens: 512,
        temperature: 0.3,
        thinkingEnabled: false, // No reasoning needed for summaries
      }
    );
  }

  const { answer } = parseThinkingBlocks(qwen.streamingText);

  return (
    <div>
      <textarea
        value={documentText}
        onChange={(e) => setDocumentText(e.target.value)}
        placeholder="Paste document text here..."
        rows={10}
        style={{ width: "100%" }}
      />

      <button
        onClick={summarize}
        disabled={qwen.phase !== "ready" || qwen.isGenerating || !documentText.trim()}
      >
        {qwen.isGenerating ? `Summarizing... (${qwen.tokensPerSecond} tok/s)` : "Summarize"}
      </button>

      {qwen.isGenerating && (
        <button onClick={() => qwen.abort()} style={{ marginLeft: 8 }}>
          Cancel
        </button>
      )}

      {answer && (
        <div style={{ marginTop: 16, whiteSpace: "pre-wrap" }}>
          <h4>Summary</h4>
          {answer}
        </div>
      )}
    </div>
  );
}
```

### Pattern D: Command Palette AI

A keyboard-triggered command palette that sends queries to the local model.

```typescript
import { useState, useEffect, useRef } from "react";
import { useQwen } from "./qwen/useQwen";
import { parseThinkingBlocks } from "./qwen/parseThinking";

function AICommandPalette() {
  const qwen = useQwen();
  const [isOpen, setIsOpen] = useState(false);
  const [query, setQuery] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);

  // Toggle with Ctrl+K
  useEffect(() => {
    function handleKeyDown(e: KeyboardEvent) {
      if (e.ctrlKey && e.key === "k") {
        e.preventDefault();
        setIsOpen((prev) => !prev);
      }
      if (e.key === "Escape") {
        setIsOpen(false);
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, []);

  // Auto-focus input when opened
  useEffect(() => {
    if (isOpen) inputRef.current?.focus();
  }, [isOpen]);

  function handleSubmit() {
    if (!query.trim() || qwen.phase !== "ready" || qwen.isGenerating) return;

    qwen.generate(
      [
        {
          role: "user",
          content: query.trim(),
        },
      ],
      {
        maxNewTokens: 256, // Keep command palette responses short
        temperature: 0.5,
        thinkingEnabled: false,
      }
    );
  }

  const { answer } = parseThinkingBlocks(qwen.streamingText);

  if (!isOpen) return null;

  return (
    <div
      style={{
        position: "fixed",
        top: 0,
        left: 0,
        right: 0,
        bottom: 0,
        background: "rgba(0,0,0,0.5)",
        display: "flex",
        justifyContent: "center",
        paddingTop: 120,
        zIndex: 9999,
      }}
      onClick={() => setIsOpen(false)}
    >
      <div
        style={{
          background: "white",
          borderRadius: 8,
          width: 560,
          maxHeight: 400,
          overflow: "auto",
          padding: 16,
          boxShadow: "0 8px 32px rgba(0,0,0,0.2)",
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <input
          ref={inputRef}
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && handleSubmit()}
          placeholder={
            qwen.phase === "ready"
              ? "Ask the AI anything... (Enter to send)"
              : qwen.phase === "downloading"
              ? `Downloading model (${qwen.downloadProgress?.percent}%)...`
              : "Loading model..."
          }
          disabled={qwen.phase !== "ready"}
          style={{
            width: "100%",
            padding: 12,
            fontSize: 16,
            border: "1px solid #ddd",
            borderRadius: 4,
            outline: "none",
          }}
        />

        {answer && (
          <div
            style={{
              marginTop: 12,
              padding: 12,
              background: "#f9f9f9",
              borderRadius: 4,
              whiteSpace: "pre-wrap",
              fontSize: 14,
              lineHeight: 1.5,
            }}
          >
            {answer}
          </div>
        )}

        {qwen.isGenerating && (
          <div
            style={{
              marginTop: 8,
              fontSize: 12,
              color: "#888",
              display: "flex",
              justifyContent: "space-between",
            }}
          >
            <span>{qwen.tokensPerSecond} tok/s</span>
            <button
              onClick={() => qwen.abort()}
              style={{ fontSize: 12, cursor: "pointer" }}
            >
              Stop
            </button>
          </div>
        )}

        <div style={{ marginTop: 8, fontSize: 11, color: "#aaa" }}>
          Press Ctrl+K to toggle | Esc to close | Device: {qwen.device ?? "..."}
        </div>
      </div>
    </div>
  );
}
```

---

## 6. Multi-turn Conversation Management

The `useQwen` hook is **stateless** regarding conversation history. It does not track messages. The consumer is responsible for maintaining conversation history and passing the full array each time `generate()` is called.

### How it works

1. The consumer maintains an array of `QwenMessage` objects.
2. When the user sends a message, the consumer appends it to the array.
3. The consumer calls `generate(fullHistory)` with the entire array.
4. When generation completes (`isGenerating` transitions from `true` to `false`), the consumer parses `streamingText` and appends the assistant response to the array.
5. On the next user message, the consumer passes the full array again (including all previous messages).

### Example: Conversation History Manager

```typescript
import { useState, useEffect, useRef } from "react";
import { useQwen } from "./qwen/useQwen";
import { parseThinkingBlocks } from "./qwen/parseThinking";
import type { QwenMessage } from "./qwen/qwen.types";

interface ConversationEntry {
  role: "user" | "assistant";
  content: string;
  thinking?: string;
}

function useConversation() {
  const qwen = useQwen();
  const [history, setHistory] = useState<ConversationEntry[]>([]);
  const wasGeneratingRef = useRef(false);

  // Commit assistant response to history when generation finishes
  useEffect(() => {
    if (wasGeneratingRef.current && !qwen.isGenerating && qwen.streamingText) {
      const { thinking, answer } = parseThinkingBlocks(qwen.streamingText);
      setHistory((prev) => [
        ...prev,
        {
          role: "assistant",
          content: answer,
          thinking: thinking || undefined,
        },
      ]);
    }
    wasGeneratingRef.current = qwen.isGenerating;
  }, [qwen.isGenerating, qwen.streamingText]);

  function sendMessage(text: string, settings?: Parameters<typeof qwen.generate>[1]) {
    if (qwen.phase !== "ready" || qwen.isGenerating) return;

    const userEntry: ConversationEntry = { role: "user", content: text };
    const newHistory = [...history, userEntry];
    setHistory(newHistory);

    // Build the messages array for the model
    const messages: QwenMessage[] = newHistory.map((e) => ({
      role: e.role,
      content: e.content,
    }));

    qwen.generate(messages, settings);
  }

  function clearHistory() {
    setHistory([]);
  }

  return {
    ...qwen,
    history,
    sendMessage,
    clearHistory,
  };
}
```

### Context Window Considerations

The Qwen3.5-0.8B model has a limited context window. For long conversations, the consumer should implement a sliding window or summarization strategy:

```typescript
function trimHistory(history: QwenMessage[], maxMessages: number): QwenMessage[] {
  if (history.length <= maxMessages) return history;
  // Keep the most recent messages
  return history.slice(-maxMessages);
}

// Usage: pass trimmed history to generate()
qwen.generate(trimHistory(messages, 20), settings);
```

---

## 7. Performance Notes

### First Load (Cold Start -- No Cache)

- **Duration**: ~30-60 seconds on WebGPU
- **Download size**: ~850 MB total across all model files
- The `@huggingface/transformers` library downloads from HuggingFace CDN
- Progress is reported per-file via `load-progress` messages
- WebGPU shader compilation adds ~5-10 seconds after download

### Subsequent Loads (Warm Start -- Cached)

- **Duration**: ~5-15 seconds
- Model files are served from the browser Cache API
- WebGPU shaders may need recompilation on some browsers/drivers

### Generation Speed

| Backend | Tokens/second | Notes                          |
|---------|---------------|--------------------------------|
| WebGPU  | ~20-40 tok/s  | Modern discrete GPU            |
| WebGPU  | ~10-20 tok/s  | Integrated GPU (Intel/AMD APU) |
| WASM    | ~5-10 tok/s   | CPU fallback                   |

### Memory Usage

| Backend | Usage         | Location |
|---------|---------------|----------|
| WebGPU  | ~900 MB       | VRAM     |
| WASM    | ~1.2 GB       | RAM      |

### Threading

The model runs entirely in a Web Worker. The UI thread is never blocked during loading or generation. All communication is via `postMessage`.

### Quantization

The model uses mixed quantization to balance speed and quality:

- `embed_tokens`: q4 (4-bit, reduces memory for embeddings)
- `vision_encoder`: fp16 (16-bit, preserves encoder quality)
- `decoder_model_merged`: q4 (4-bit, largest component, biggest memory savings)

---

## 8. Testing Scenarios

### Scenario 1: Cold Start (No Cache)

**Steps**: Clear browser cache, launch the application.
**Expected**: Phase transitions `idle` -> `loading` -> `downloading` -> `ready`. Download progress is reported with file names, byte counts, and percentage. Total download is ~850 MB. After download, device is reported as `"webgpu"` or `"wasm"`.

### Scenario 2: Warm Start (Cached)

**Steps**: Launch the application after a previous successful load.
**Expected**: Phase transitions `idle` -> `loading` -> `ready` within 5-15 seconds. Download progress may briefly appear but with instant completion (served from cache). No network traffic for model files.

### Scenario 3: Generate with Thinking ON

**Steps**: Send a message with `thinkingEnabled: true`.
**Expected**: `streamingText` contains `<think>...</think>` blocks followed by the answer. `parseThinkingBlocks()` correctly separates the reasoning from the response. The `<think>` block appears first in the stream.

### Scenario 4: Generate with Thinking OFF

**Steps**: Send a message with `thinkingEnabled: false`.
**Expected**: `streamingText` contains only the answer. No `<think>` blocks. Response is shorter and faster. The system prompt includes `/no_think`.

### Scenario 5: Abort Mid-Generation

**Steps**: Call `abort()` while `isGenerating` is `true`.
**Expected**: Generation stops. `isGenerating` becomes `false`. `streamingText` contains the partial response generated up to the abort point. No `done` message is posted (the generate promise resolves after interruption, then `done` is posted).

### Scenario 6: Multi-turn Conversation

**Steps**: Send multiple messages in sequence, passing the full history each time.
**Expected**: The model's responses reflect awareness of the full conversation history. Each `generate()` call receives the entire message array. The consumer accumulates responses between calls.

### Scenario 7: WebGPU Fallback to WASM

**Steps**: Run on a machine without WebGPU support (e.g., older GPU, software renderer).
**Expected**: The model loads successfully on WASM. `device` reports `"wasm"`. Generation works but at lower speed (~5-10 tok/s). No errors or crashes.

### Scenario 8: Error Recovery

**Steps**: Disconnect network during model download, then reconnect and retry.
**Expected**: `phase` transitions to `"error"` with a descriptive error message. The consumer can reload the page or re-initialize the worker to retry. Partial cache entries may allow faster retry.

### Scenario 9: No `<|im_end|>` Tokens in Output

**Steps**: Generate any response and inspect `streamingText`.
**Expected**: No raw special tokens (`<|im_end|>`, `<|im_start|>`, etc.) appear in the output. The `skip_special_tokens: true` flag on `TextStreamer` strips them.

### Scenario 10: Concurrent Hook Instances

**Steps**: Mount two components that both call `useQwen()`.
**Expected**: Both share the same singleton worker. The model loads once. Both receive the same state updates. Only one can generate at a time (second `generate()` call is a no-op if `isGenerating` is true).

---

## 9. File Checklist

| File                | Size (approx) | Purpose                              |
|---------------------|---------------|---------------------------------------|
| `qwen.types.ts`     | ~2 KB         | All TypeScript types                  |
| `qwen.worker.ts`    | ~3 KB         | Web Worker: model load + generation   |
| `parseThinking.ts`  | ~0.5 KB       | Think block parser utility            |
| `useQwen.ts`        | ~4 KB         | React hook wrapping the worker        |

Total integration footprint: ~10 KB of source code (excluding dependencies).

---

## 10. Dependency Summary

### Runtime Dependencies (added to host project)

| Package                        | Version     | Purpose                                  |
|--------------------------------|-------------|------------------------------------------|
| `@huggingface/transformers`    | ^4.0.0-next | Model loading, tokenization, generation  |
| `onnxruntime-web`              | ^1.21.0     | ONNX inference runtime (WebGPU + WASM)   |

### Dev Dependencies (added to host project)

| Package                   | Version | Purpose                          |
|---------------------------|---------|----------------------------------|
| `vite-plugin-static-copy` | ^2.0.0  | Copy ORT WASM binaries to public |

### Rust Dependencies (in `src-tauri/Cargo.toml`)

| Crate                | Version | Purpose                                    |
|----------------------|---------|--------------------------------------------|
| `tauri-plugin-http`  | 2       | Allow WebView to fetch from HuggingFace    |

Note: `tauri` must have the `"protocol-asset"` feature enabled if the host project uses asset protocol for other purposes.
