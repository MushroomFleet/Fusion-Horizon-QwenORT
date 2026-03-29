# Qwen3.5-0.8B ONNX Local Inference — Fusion-Horizon-UI Integration Plan

## TINS Metadata

| Field             | Value                                                                |
|-------------------|----------------------------------------------------------------------|
| **Document**      | TINS (Technical Implementation & Novelty Specification)              |
| **Version**       | 1.0.0                                                                |
| **Date**          | 2026-03-29                                                           |
| **Model**         | `onnx-community/Qwen3.5-0.8B-ONNX`                                  |
| **Runtime**       | ONNX Runtime Web (WebGPU primary, WASM fallback)                     |
| **Framework**     | `@huggingface/transformers` v4 (`@next` tag)                         |
| **Host Stack**    | React 19 + TypeScript 5.9 + Vite 7 + Zustand 5 (pure web app)       |
| **Type**          | Built-in LLM provider for hermeneutic pipeline integration           |
| **Reference Impl**| `Qwen35-08-onnx-headless-post-TINS.md` (verified working pattern)    |

---

## 1. Description

This document specifies the integration of local Qwen3.5-0.8B ONNX inference into the **Fusion-Horizon-UI** hermeneutic interpretation application as a first-class LLM provider. The goal is to prove whether a 0.8B parameter model can function within the fusion-horizon hermeneutic circle pipeline (horizon mapping, dialectical analysis, iteration control, and horizon fusion).

### What this provides

- An **LLM Provider abstraction layer** (`ILLMProvider` interface) that decouples all agents from any specific API client
- A **Qwen Local Provider** (`QwenLocalProvider`) that adapts the Web Worker inference engine to the provider interface
- A **refactored OpenRouter Provider** (`OpenRouterProvider`) that wraps the existing client behind the same interface
- A **provider selection system** in Settings UI with three modes: `qwen-local` (default), `openrouter`
- A **model status indicator** showing Qwen download/loading/ready state
- **Zero-configuration default** — the app works out of the box with local inference, no API key required

### What this does NOT change

- The hermeneutic pipeline logic (agents, orchestrator, state layers) — only the LLM call path changes
- The system prompts in `src/config/prompts.ts` — identical prompts are sent to Qwen
- Session history, convergence algorithm, or UI layout — all remain as-is
- The OpenRouter provider continues to work exactly as before when selected

### Architectural decision: Why a provider interface, not a direct hook swap

The existing codebase has 4 agents that all call `this.apiClient.createCompletion()` via `BaseAgent`. Rather than rewriting each agent, we introduce `ILLMProvider` with a single `createCompletion()` method. Both OpenRouter and Qwen implement it. The orchestrator passes the active provider to agents at construction time. This is the minimum viable abstraction — one interface, two implementations, one injection point.

---

## 2. Functionality

### 2.1 Provider Selection

The user selects their LLM provider in Settings. The choice persists in localStorage via the existing Zustand `uiStore`. Available providers:

| Provider       | Label in UI           | Requires API Key | Default |
|----------------|-----------------------|-------------------|---------|
| `qwen-local`  | "Built-in (Qwen 0.8B)"| No               | Yes     |
| `openrouter`  | "OpenRouter (Remote)" | Yes               | No      |

When `qwen-local` is selected, the API Key and Model Name fields are hidden. When `openrouter` is selected, those fields appear as they do today.

### 2.2 Model Lifecycle (Qwen Local Only)

When the provider is `qwen-local`, the model must be loaded before queries can be processed. The lifecycle phases are:

| Phase         | Description                                      | UI Indication                    |
|---------------|--------------------------------------------------|----------------------------------|
| `idle`        | Worker not yet initialized                       | None (pre-mount)                 |
| `loading`     | Worker spawned, processor/model loading           | Status bar: "Loading model..."   |
| `downloading` | First-time model download from HuggingFace CDN   | Progress bar with file/percent   |
| `ready`       | Model loaded, inference available                 | Green indicator, device shown    |
| `error`       | Load failed                                       | Red indicator with error message |

The model loads eagerly when `qwen-local` is the selected provider. It does NOT load when `openrouter` is selected (no wasted resources).

### 2.3 Hermeneutic Pipeline with Local Inference

The pipeline executes identically regardless of provider. Each agent calls `provider.createCompletion(messages, options)` and receives a `CompletionResponse`. The only differences with local inference:

- **No network latency** for LLM calls (but initial model download requires network)
- **No API key required** — works offline after first model download
- **Token streaming** happens internally in the worker; the provider collects the full response before returning (non-streaming adapter pattern)
- **Thinking mode** is disabled (`/no_think`) for pipeline calls — the hermeneutic system needs structured JSON output, not chain-of-thought reasoning
- **Token limits** are lower (max 2048 new tokens per generation vs. effectively unlimited on remote models)

### 2.4 Generation Parameters for Pipeline Agents

Each agent in the pipeline uses specific parameters. These are mapped to Qwen generation settings:

| Agent                  | Temperature | Max Tokens (OpenRouter) | Max Tokens (Qwen) | Notes                         |
|------------------------|-------------|-------------------------|--------------------|-------------------------------|
| HorizonMapperAgent     | 0.7         | 1500                    | 1500               | JSON output required          |
| DialecticAnalyzerAgent | 0.7         | 2000                    | 2000               | JSON output, largest payload  |
| IterationControllerAgent| 0.7        | 1500                    | 1024               | JSON output, decision only    |
| HorizonFusionAgent     | 0.7         | 2500                    | 2048               | Capped at Qwen's max          |

### 2.5 Qwen-Specific System Prompt Wrapper

The Qwen model requires specific formatting for reliable JSON output from a 0.8B model. A thin prompt wrapper is applied ONLY when the active provider is `qwen-local`:

```
[Original system prompt from prompts.ts]

CRITICAL: You MUST respond with valid JSON only. No markdown, no explanation, no text outside the JSON object. Start your response with { and end with }.
```

This wrapper is applied in `QwenLocalProvider.createCompletion()`, not in the agent code.

---

## 3. Technical Implementation

### 3.1 New Dependencies

```bash
npm install @huggingface/transformers@next onnxruntime-web
npm install -D vite-plugin-static-copy
```

**`@huggingface/transformers@next`** — v4.x required. Contains `Qwen3_5ForConditionalGeneration`. The stable v3.x does NOT have this class and will fail at import.

**`onnxruntime-web`** — ONNX Runtime for WebGPU and WASM inference backends.

**`vite-plugin-static-copy`** — Copies ORT WASM binaries to the build output so the browser can fetch them at runtime.

### 3.2 Vite Configuration Changes

**File:** `vite.config.ts`

Merge these changes into the existing config. Do not replace the file.

```typescript
// ADD this import at the top
import { viteStaticCopy } from "vite-plugin-static-copy";

export default defineConfig({
  plugins: [
    react(),

    // ADD: Copy ORT WASM binaries so the browser can fetch them at runtime
    viteStaticCopy({
      targets: [
        {
          // MUST use forward-slash relative path — path.resolve() breaks on Windows
          src: "node_modules/onnxruntime-web/dist/*.wasm",
          dest: "ort-wasm",
        },
      ],
    }),
  ],

  // KEEP existing resolve.alias block unchanged

  // ADD: Exclude @huggingface/transformers from Vite's dependency pre-bundling.
  // The library uses top-level await and dynamic imports that break pre-bundling.
  optimizeDeps: {
    exclude: ["@huggingface/transformers"],
  },

  server: {
    port: 5173,
    host: true,
  },

  build: {
    outDir: 'dist',
    sourcemap: false,
    minify: 'terser',

    // CHANGE: target must be esnext for top-level await support
    target: "esnext",

    rollupOptions: {
      output: {
        // CHANGE: format must be esm for dynamic imports in workers
        format: "esm",
        // KEEP existing manualChunks
        manualChunks: {
          'react-vendor': ['react', 'react-dom'],
          'store-vendor': ['zustand', 'immer'],
          'ui-vendor': ['@fortawesome/react-fontawesome', '@fortawesome/fontawesome-svg-core'],
        },
      },
    },
    chunkSizeWarningLimit: 1000,
  },

  // ADD: Web Workers must use ES module format for dynamic imports to work
  worker: {
    format: "es",
  },
})
```

### 3.3 ORT Bootstrap in Entry Point

**File:** `src/main.tsx`

Add ORT configuration BEFORE `createRoot()`. This tells ONNX Runtime where to find WASM binaries and enables WebGPU.

```typescript
import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import './index.css'
import App from './App.tsx'
import { ErrorBoundary } from './components/common/ErrorBoundary'
import { ToastContainer } from './components/common/Toast'

// ─── ORT Bootstrap ─────────────────────────────────────────────────────────
// Must execute before any component that uses the Qwen provider mounts.
import * as ort from "onnxruntime-web";

// Tell ORT where to find the WASM binaries (copied by viteStaticCopy)
ort.env.wasm.wasmPaths = "/ort-wasm/";

// Enable WebGPU with high-performance power preference
// TypeScript types mark env.webgpu as readonly — cast to any
(ort.env as any).webgpu = { powerPreference: "high-performance" };
// ────────────────────────────────────────────────────────────────────────────

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ErrorBoundary>
      <App />
      <ToastContainer />
    </ErrorBoundary>
  </StrictMode>,
)
```

### 3.4 CRITICAL WARNINGS (from reference implementation)

These are hard-won lessons. Each one caused real build failures or runtime errors.

#### WARNING 1: `vite-plugin-static-copy` must use RELATIVE forward-slash paths

On Windows, `path.resolve()` produces backslash paths that break glob matching.

```typescript
// CORRECT
src: "node_modules/onnxruntime-web/dist/*.wasm"

// WRONG — breaks on Windows
src: path.resolve(__dirname, "node_modules/onnxruntime-web/dist/*.wasm")
```

#### WARNING 2: `ort.env.webgpu` is readonly — must cast to `any`

```typescript
// CORRECT
(ort.env as any).webgpu = { powerPreference: "high-performance" };

// WRONG — TypeScript error
ort.env.webgpu = { powerPreference: "high-performance" };
```

#### WARNING 3: `@huggingface/transformers` `@next` tag is mandatory

The stable release (v3.x) does not include `Qwen3_5ForConditionalGeneration`. Only the `@next` tag (v4.x) has this class.

```bash
# WRONG — installs v3.x which lacks the model class
npm install @huggingface/transformers

# CORRECT — installs v4.x with Qwen3.5 support
npm install @huggingface/transformers@next
```

#### WARNING 4: `skip_special_tokens: true` is required on TextStreamer

Without this flag, raw special tokens like `<|im_end|>` appear in streamed output.

#### WARNING 5: This is a vision-language model, not a text-only model

Uses `Qwen3_5ForConditionalGeneration` + `AutoProcessor`, NOT `AutoModelForCausalLM` + `AutoTokenizer`. The dtype config must include `vision_encoder`:

```typescript
dtype: {
  embed_tokens: "q4",
  vision_encoder: "fp16",
  decoder_model_merged: "q4",
}
```

#### WARNING 6: Message content format uses array-of-objects for non-system messages

```typescript
// CORRECT
const messages = [
  { role: "system", content: systemPrompt },
  ...req.messages.map((m) => ({
    role: m.role,
    content: [{ type: "text" as const, text: m.content }],
  })),
];

// WRONG — processor expects array content for non-system messages
const messages = [
  { role: "system", content: systemPrompt },
  ...req.messages,
];
```

#### WARNING 7: No Tauri in this project

Fusion-Horizon-UI is a pure web app (Vite + React). There is no `src-tauri/` directory. All Tauri-specific configuration (CSP, capabilities, Cargo.toml, Rust plugins) from the reference TINS plan does NOT apply. The browser handles CORS for HuggingFace CDN natively. No CSP headers need to be set for dev mode.

---

## 4. File-by-File Implementation

### 4.1 File: `src/core/api/types.ts` (NEW)

The provider abstraction interface. This is the single contract that all LLM providers must implement.

```typescript
import type { ChatMessage, CompletionRequest, CompletionResponse } from '@/types/api';

/**
 * LLM Provider type identifier.
 * 'qwen-local' = built-in Qwen 0.8B ONNX inference (default)
 * 'openrouter' = remote OpenRouter API (BYOK)
 */
export type LLMProviderType = 'qwen-local' | 'openrouter';

/**
 * Lifecycle phase for providers that require loading (e.g., local models).
 * Remote providers are always 'ready'.
 */
export type ProviderPhase = 'idle' | 'loading' | 'downloading' | 'ready' | 'error';

/**
 * Download progress for local model providers.
 */
export interface ProviderDownloadProgress {
  file: string;
  loaded: number;
  total: number;
  percent: number;
}

/**
 * Provider status information for UI display.
 */
export interface ProviderStatus {
  phase: ProviderPhase;
  device: string | null;
  downloadProgress: ProviderDownloadProgress | null;
  statusMessage: string;
  error: string | null;
  isConfigured: boolean;
}

/**
 * The single interface that all LLM providers must implement.
 * Agents call createCompletion() — they never know which provider is active.
 */
export interface ILLMProvider {
  /** Unique type identifier for this provider. */
  readonly type: LLMProviderType;

  /** Human-readable label for UI display. */
  readonly label: string;

  /**
   * Initialize the provider. For local models, this starts model loading.
   * For remote providers, this validates configuration.
   * Must be called before createCompletion().
   */
  initialize(): Promise<void>;

  /**
   * Create a chat completion. This is the only method agents call.
   * Returns a CompletionResponse matching the existing API types.
   */
  createCompletion(
    messages: ChatMessage[],
    options?: Partial<CompletionRequest>
  ): Promise<CompletionResponse>;

  /**
   * Get current provider status for UI display.
   */
  getStatus(): ProviderStatus;

  /**
   * Check if the provider is ready for inference.
   */
  isReady(): boolean;

  /**
   * Clean up resources (e.g., terminate worker).
   */
  dispose(): void;
}
```

---

### 4.2 File: `src/core/api/OpenRouterProvider.ts` (NEW)

Wraps the existing `OpenRouterClient` behind the `ILLMProvider` interface. Minimal changes — the client code stays as-is.

```typescript
import { openRouterClient, OpenRouterClient } from './OpenRouterClient';
import type { ChatMessage, CompletionRequest, CompletionResponse } from '@/types/api';
import type { ILLMProvider, LLMProviderType, ProviderStatus } from './types';

/**
 * OpenRouter LLM Provider.
 * Wraps the existing OpenRouterClient behind the ILLMProvider interface.
 */
export class OpenRouterProvider implements ILLMProvider {
  readonly type: LLMProviderType = 'openrouter';
  readonly label = 'OpenRouter (Remote)';

  private client: OpenRouterClient;

  constructor() {
    this.client = openRouterClient;
  }

  async initialize(): Promise<void> {
    // Refresh from localStorage in case the user updated API key/model
    this.client.refreshFromLocalStorage();
  }

  async createCompletion(
    messages: ChatMessage[],
    options?: Partial<CompletionRequest>
  ): Promise<CompletionResponse> {
    // Refresh before each call to pick up settings changes
    this.client.refreshFromLocalStorage();
    return this.client.createCompletion(messages, options);
  }

  getStatus(): ProviderStatus {
    return {
      phase: this.client.isConfigured() ? 'ready' : 'error',
      device: null,
      downloadProgress: null,
      statusMessage: this.client.isConfigured() ? 'Connected' : 'No API key configured',
      error: this.client.isConfigured() ? null : 'Please add your OpenRouter API key in Settings',
      isConfigured: this.client.isConfigured(),
    };
  }

  isReady(): boolean {
    return this.client.isConfigured();
  }

  dispose(): void {
    // No resources to clean up for remote provider
  }
}
```

---

### 4.3 File: `src/core/qwen/qwen.types.ts` (NEW)

All TypeScript types for the Qwen worker protocol. Copied from the reference TINS with no changes.

```typescript
// ─── Model Lifecycle ────────────────────────────────────────────────────────

export type QwenPhase = "idle" | "loading" | "downloading" | "ready" | "error";

export interface QwenDownloadProgress {
  file: string;
  loaded: number;
  total: number;
  percent: number;
}

// ─── Generation Settings ────────────────────────────────────────────────────

export interface QwenGenerationSettings {
  maxNewTokens: number;
  temperature: number;
  topP: number;
  repetitionPenalty: number;
  thinkingEnabled: boolean;
}

// ─── Message Types ──────────────────────────────────────────────────────────

export interface QwenMessage {
  role: "user" | "assistant";
  content: string;
}

export interface QwenParsedResponse {
  thinking: string;
  answer: string;
}

// ─── Worker Protocol ────────────────────────────────────────────────────────

export type QwenWorkerInMessage =
  | { type: "init" }
  | {
      type: "generate";
      payload: {
        messages: QwenMessage[];
        settings: QwenGenerationSettings;
        systemPrompt?: string;
      };
    }
  | { type: "abort" };

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
  | { type: "done"; payload: { totalTokens: number } }
  | { type: "error"; payload: string };
```

Note: Two differences from the reference TINS:
1. `generate` payload includes optional `systemPrompt` — the provider injects the agent's system prompt with the JSON reinforcement wrapper.
2. `done` payload includes `totalTokens` for usage tracking in `CompletionResponse`.

---

### 4.4 File: `src/core/qwen/qwen.worker.ts` (NEW)

The Web Worker. Runs model loading and generation in a separate thread. Adapted from reference TINS with the `systemPrompt` passthrough.

```typescript
import {
  env,
  AutoProcessor,
  Qwen3_5ForConditionalGeneration,
  TextStreamer,
  InterruptableStoppingCriteria,
} from "@huggingface/transformers";

env.allowLocalModels = false;
env.allowRemoteModels = true;

const MODEL_ID = "onnx-community/Qwen3.5-0.8B-ONNX";

let model: any = null;
let processor: any = null;
let stopCriteria = new InterruptableStoppingCriteria();

// ─── Worker Protocol ────────────────────────────────────────────────────────

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
  systemPrompt?: string;
};

// ─── Progress Reporting ─────────────────────────────────────────────────────

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

  // Use provided system prompt (from the hermeneutic agent) or default.
  // /no_think is always appended for pipeline use — we need structured JSON, not CoT.
  const systemPrompt = req.systemPrompt
    ? req.systemPrompt + " /no_think"
    : req.settings.thinkingEnabled
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

  let tokenCount = 0;

  const streamer = new TextStreamer(processor.tokenizer, {
    skip_prompt: true,
    skip_special_tokens: true,
    callback_function: (token: string) => {
      tokenCount++;
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

  postMessage({ type: "done", payload: { totalTokens: tokenCount } });
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

### 4.5 File: `src/core/qwen/parseThinking.ts` (NEW)

Utility to separate `<think>...</think>` reasoning blocks from the main response. Copied from reference TINS unchanged.

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

### 4.6 File: `src/core/api/QwenLocalProvider.ts` (NEW)

The critical adapter. Implements `ILLMProvider` using the Web Worker. Collects streamed tokens into a full `CompletionResponse` that agents consume. This is the bridge between the worker protocol and the existing agent contract.

```typescript
import type { ChatMessage, CompletionRequest, CompletionResponse } from '@/types/api';
import type { ILLMProvider, LLMProviderType, ProviderPhase, ProviderStatus, ProviderDownloadProgress } from './types';
import type { QwenGenerationSettings } from '@/core/qwen/qwen.types';
import { parseThinkingBlocks } from '@/core/qwen/parseThinking';

// Default generation settings for hermeneutic pipeline use
const PIPELINE_DEFAULTS: QwenGenerationSettings = {
  maxNewTokens: 2048,
  temperature: 0.7,
  topP: 0.9,
  repetitionPenalty: 1.1,
  thinkingEnabled: false, // Pipeline needs JSON, not CoT
};

/**
 * JSON reinforcement suffix appended to agent system prompts.
 * Helps the 0.8B model stay on track for structured output.
 */
const JSON_REINFORCEMENT = `\n\nCRITICAL: You MUST respond with valid JSON only. No markdown, no explanation, no text outside the JSON object. Start your response with { and end with }.`;

/**
 * Qwen Local LLM Provider.
 *
 * Manages a singleton Web Worker running Qwen3.5-0.8B ONNX inference.
 * Adapts the streaming worker protocol into the synchronous CompletionResponse
 * contract that BaseAgent.callLLM() expects.
 */
export class QwenLocalProvider implements ILLMProvider {
  readonly type: LLMProviderType = 'qwen-local';
  readonly label = 'Built-in (Qwen 0.8B)';

  private worker: Worker | null = null;
  private phase: ProviderPhase = 'idle';
  private device: string | null = null;
  private downloadProgress: ProviderDownloadProgress | null = null;
  private statusMessage = '';
  private error: string | null = null;

  // Listeners for status updates (UI subscribes to these)
  private statusListeners: Set<(status: ProviderStatus) => void> = new Set();

  /**
   * Subscribe to status changes for UI updates.
   * Returns an unsubscribe function.
   */
  onStatusChange(listener: (status: ProviderStatus) => void): () => void {
    this.statusListeners.add(listener);
    return () => this.statusListeners.delete(listener);
  }

  private notifyStatusChange(): void {
    const status = this.getStatus();
    this.statusListeners.forEach((listener) => listener(status));
  }

  /**
   * Initialize the provider: spawn the worker and start model loading.
   * Resolves when the model is ready for inference.
   */
  async initialize(): Promise<void> {
    if (this.phase === 'ready') return;
    if (this.phase === 'loading' || this.phase === 'downloading') return;

    this.phase = 'loading';
    this.error = null;
    this.notifyStatusChange();

    return new Promise<void>((resolve, reject) => {
      this.worker = new Worker(
        new URL('../qwen/qwen.worker.ts', import.meta.url),
        { type: 'module' }
      );

      const handler = (e: MessageEvent) => {
        const msg = e.data;
        switch (msg.type) {
          case 'status':
            this.statusMessage = msg.payload;
            this.notifyStatusChange();
            break;

          case 'load-progress':
            if (msg.payload.total > 0) {
              this.phase = 'downloading';
              this.downloadProgress = {
                file: msg.payload.file,
                loaded: msg.payload.loaded,
                total: msg.payload.total,
                percent: Math.round((msg.payload.loaded / msg.payload.total) * 100),
              };
              this.notifyStatusChange();
            }
            break;

          case 'ready':
            this.device = msg.payload.device;
            this.phase = 'ready';
            this.downloadProgress = null;
            this.statusMessage = `Model ready (${msg.payload.device})`;
            this.notifyStatusChange();
            // Keep the handler attached for generate responses
            resolve();
            break;

          case 'error':
            this.phase = 'error';
            this.error = msg.payload;
            this.notifyStatusChange();
            reject(new Error(msg.payload));
            break;
        }
      };

      this.worker.addEventListener('message', handler);
      this.worker.postMessage({ type: 'init' });
    });
  }

  /**
   * Create a chat completion by running inference on the local model.
   *
   * Collects all streamed tokens into a single CompletionResponse.
   * The system message in the messages array is intercepted, wrapped with
   * JSON reinforcement, and passed to the worker as `systemPrompt`.
   */
  async createCompletion(
    messages: ChatMessage[],
    options?: Partial<CompletionRequest>
  ): Promise<CompletionResponse> {
    if (!this.worker || this.phase !== 'ready') {
      throw new Error(`Qwen provider not ready (phase: ${this.phase})`);
    }

    // Separate system prompt from conversation messages
    let systemPrompt: string | undefined;
    const conversationMessages: { role: string; content: string }[] = [];

    for (const msg of messages) {
      if (msg.role === 'system') {
        systemPrompt = msg.content + JSON_REINFORCEMENT;
      } else {
        conversationMessages.push({ role: msg.role, content: msg.content });
      }
    }

    // Map CompletionRequest options to Qwen generation settings
    const generationSettings: QwenGenerationSettings = {
      ...PIPELINE_DEFAULTS,
      temperature: options?.temperature ?? PIPELINE_DEFAULTS.temperature,
      maxNewTokens: options?.maxTokens ?? PIPELINE_DEFAULTS.maxNewTokens,
    };

    return new Promise<CompletionResponse>((resolve, reject) => {
      let collectedText = '';
      let totalTokens = 0;

      const handler = (e: MessageEvent) => {
        const msg = e.data;
        switch (msg.type) {
          case 'token':
            collectedText += msg.payload;
            break;

          case 'done':
            totalTokens = msg.payload?.totalTokens ?? 0;
            this.worker!.removeEventListener('message', handler);

            // Strip any thinking blocks from the response (pipeline doesn't need them)
            const { answer } = parseThinkingBlocks(collectedText);
            const finalContent = answer || collectedText;

            resolve({
              content: finalContent,
              model: 'qwen3.5-0.8b-onnx-local',
              usage: {
                promptTokens: 0, // Worker doesn't report prompt tokens
                completionTokens: totalTokens,
                totalTokens: totalTokens,
              },
              finishReason: 'stop',
              rawResponse: { local: true, device: this.device },
            });
            break;

          case 'error':
            this.worker!.removeEventListener('message', handler);
            reject(new Error(`Qwen inference error: ${msg.payload}`));
            break;
        }
      };

      this.worker!.addEventListener('message', handler);

      this.worker!.postMessage({
        type: 'generate',
        payload: {
          messages: conversationMessages,
          settings: generationSettings,
          systemPrompt,
        },
      });
    });
  }

  getStatus(): ProviderStatus {
    return {
      phase: this.phase,
      device: this.device,
      downloadProgress: this.downloadProgress,
      statusMessage: this.statusMessage,
      error: this.error,
      isConfigured: true, // Always configured — no API key needed
    };
  }

  isReady(): boolean {
    return this.phase === 'ready';
  }

  dispose(): void {
    if (this.worker) {
      this.worker.terminate();
      this.worker = null;
    }
    this.phase = 'idle';
    this.device = null;
    this.notifyStatusChange();
  }
}
```

---

### 4.7 File: `src/core/api/providerRegistry.ts` (NEW)

Singleton registry that manages provider instances and selection.

```typescript
import type { ILLMProvider, LLMProviderType } from './types';
import { QwenLocalProvider } from './QwenLocalProvider';
import { OpenRouterProvider } from './OpenRouterProvider';

/**
 * Provider Registry — manages LLM provider instances.
 *
 * The active provider is determined by the user's settings.
 * Providers are lazily initialized (only the active one loads).
 */
class ProviderRegistry {
  private providers: Map<LLMProviderType, ILLMProvider> = new Map();
  private activeType: LLMProviderType = 'qwen-local';

  constructor() {
    // Register all available providers
    this.providers.set('qwen-local', new QwenLocalProvider());
    this.providers.set('openrouter', new OpenRouterProvider());
  }

  /**
   * Get the currently active provider.
   */
  getActive(): ILLMProvider {
    const provider = this.providers.get(this.activeType);
    if (!provider) {
      throw new Error(`No provider registered for type: ${this.activeType}`);
    }
    return provider;
  }

  /**
   * Get a specific provider by type.
   */
  get(type: LLMProviderType): ILLMProvider | undefined {
    return this.providers.get(type);
  }

  /**
   * Switch the active provider.
   * Does NOT initialize the new provider — call initialize() separately.
   */
  async setActive(type: LLMProviderType): Promise<void> {
    if (!this.providers.has(type)) {
      throw new Error(`Unknown provider type: ${type}`);
    }

    // Dispose the old provider if switching away from a local model
    if (this.activeType !== type) {
      const oldProvider = this.providers.get(this.activeType);
      if (oldProvider && this.activeType === 'qwen-local') {
        oldProvider.dispose();
        // Re-create so it can be initialized fresh later
        this.providers.set('qwen-local', new QwenLocalProvider());
      }
    }

    this.activeType = type;
  }

  /**
   * Initialize the active provider.
   */
  async initializeActive(): Promise<void> {
    const provider = this.getActive();
    await provider.initialize();
  }

  /**
   * Get all available provider types and their labels.
   */
  getAvailableProviders(): Array<{ type: LLMProviderType; label: string }> {
    return Array.from(this.providers.entries()).map(([type, provider]) => ({
      type,
      label: provider.label,
    }));
  }

  /**
   * Get the active provider type.
   */
  getActiveType(): LLMProviderType {
    return this.activeType;
  }
}

/** Singleton provider registry instance. */
export const providerRegistry = new ProviderRegistry();
```

---

### 4.8 Changes to `src/core/agents/BaseAgent.ts` (MODIFY)

Replace the hardcoded `OpenRouterClient` dependency with `ILLMProvider`. This is the key decoupling point.

**Before:**

```typescript
import { openRouterClient } from '@/core/api/OpenRouterClient';
import type { OpenRouterClient } from '@/core/api/OpenRouterClient';
// ...
export abstract class BaseAgent implements IAgent {
  // ...
  protected apiClient: OpenRouterClient;
  // ...
  constructor(config: AgentConfig, apiClient?: OpenRouterClient) {
    // ...
    this.apiClient = apiClient || openRouterClient;
    // ...
  }
  // ...
  protected async callLLM(userMessage: string, options?: LLMCallOptions): Promise<string> {
    // ...
    const response = await this.apiClient.createCompletion(
      this.conversationHistory,
      {
        temperature: options?.temperature ?? this.config.temperature ?? 0.7,
        maxTokens: options?.maxTokens ?? this.config.maxTokens,
      }
    );
    // ...
  }
}
```

**After:**

```typescript
import { providerRegistry } from '@/core/api/providerRegistry';
import type { ILLMProvider } from '@/core/api/types';
import type { IMessage } from '@/types/messages';
import type { ChatMessage } from '@/types/api';
import type { AgentConfig, AgentStatus, AgentState, LLMCallOptions, IAgent } from './types';
import type { AgentRole } from '@/types/messages';

export abstract class BaseAgent implements IAgent {
  config: AgentConfig;
  status: AgentStatus;
  role: AgentRole;
  name: string;

  protected provider: ILLMProvider;
  protected conversationHistory: ChatMessage[];
  protected messageHistory: IMessage[];
  protected isInitialized: boolean;

  constructor(config: AgentConfig, provider?: ILLMProvider) {
    this.config = config;
    this.role = config.role;
    this.name = config.name;
    this.status = 'uninitialized';

    // Use injected provider or fall back to the active provider from the registry
    this.provider = provider || providerRegistry.getActive();

    this.conversationHistory = [
      {
        role: 'system',
        content: config.systemPrompt,
      },
    ];
    this.messageHistory = [];
    this.isInitialized = false;

    if (config.enableLogging !== false) {
      console.info(`Agent '${this.name}' created with role ${this.role} using provider ${this.provider.type}`);
    }
  }

  abstract process(message: IMessage): Promise<IMessage>;

  async initialize(): Promise<void> {
    if (this.isInitialized) {
      console.warn(`Agent '${this.name}' already initialized`);
      return;
    }
    console.info(`Initializing agent '${this.name}'`);
    this.isInitialized = true;
    this.status = 'initialized';
  }

  async cleanup(): Promise<void> {
    console.info(`Cleaning up agent '${this.name}'`);
    this.conversationHistory = [this.conversationHistory[0]];
    this.messageHistory = [];
    this.status = 'cleaned_up';
  }

  getStatus(): AgentStatus {
    return this.status;
  }

  getState(): AgentState {
    return {
      role: this.role,
      name: this.name,
      isInitialized: this.isInitialized,
      messageCount: this.messageHistory.length,
      conversationTurns: this.conversationHistory.length - 1,
      status: this.status,
    };
  }

  protected async callLLM(
    userMessage: string,
    options?: LLMCallOptions
  ): Promise<string> {
    const userChatMessage: ChatMessage = {
      role: 'user',
      content: userMessage,
    };
    this.conversationHistory.push(userChatMessage);

    console.debug(`[${this.name}] Calling LLM (${this.provider.type}) with message: ${userMessage.substring(0, 100)}...`);

    try {
      const response = await this.provider.createCompletion(
        this.conversationHistory,
        {
          temperature: options?.temperature ?? this.config.temperature ?? 0.7,
          maxTokens: options?.maxTokens ?? this.config.maxTokens,
        }
      );

      const assistantMessage: ChatMessage = {
        role: 'assistant',
        content: response.content,
      };
      this.conversationHistory.push(assistantMessage);

      console.debug(`[${this.name}] LLM response: ${response.content.substring(0, 100)}...`);

      return response.content;
    } catch (error) {
      console.error(`[${this.name}] LLM call failed:`, error);
      throw error;
    }
  }

  protected clearConversationHistory(): void {
    const systemPrompt = this.conversationHistory[0];
    this.conversationHistory = [systemPrompt];
    console.debug(`[${this.name}] Conversation history cleared`);
  }

  protected addMessageToHistory(message: IMessage): void {
    this.messageHistory.push(message);
  }

  toString(): string {
    return `${this.constructor.name}(name='${this.name}', role=${this.role}, status=${this.status})`;
  }
}
```

---

### 4.9 Changes to `src/stores/uiStore.ts` (MODIFY)

Add `llmProvider` field to `UISettings`.

**Add to the `UISettings` interface:**

```typescript
export interface UISettings {
  // LLM Provider selection
  llmProvider: 'qwen-local' | 'openrouter';

  // ... all existing fields unchanged ...
}
```

**Update `defaultSettings`:**

```typescript
const defaultSettings: UISettings = {
  llmProvider: 'qwen-local',  // ADD: Default to built-in local model

  // ... all existing defaults unchanged ...
  showDetailedProcess: true,
  autoScroll: true,
  animationsEnabled: true,
  defaultMode: 'single',
  saveOutputByDefault: true,
  apiKey: '',
  modelName: 'x-ai/grok-4-fast:online',
  temperature: 0.7,
  maxTokens: undefined,
  minCycles: 2,
  maxCycles: 5,
  convergenceThreshold: 0.85,
};
```

---

### 4.10 Changes to `src/components/features/SettingsPanel.tsx` (MODIFY)

Add provider selection dropdown and conditionally show/hide API key fields.

**Add provider selector at the top of the API Configuration section, REPLACING the existing section:**

```tsx
{/* LLM Provider Selection */}
<section>
  <h3 className="text-lg font-semibold mb-4">LLM Provider</h3>
  <div className="space-y-4">
    <div>
      <label className="block text-sm font-medium mb-2">Provider</label>
      <select
        className="w-full rounded-lg border border-border bg-background px-3 py-2 text-sm"
        value={localSettings.llmProvider}
        onChange={(e) => setLocalSettings({
          ...localSettings,
          llmProvider: e.target.value as 'qwen-local' | 'openrouter'
        })}
      >
        <option value="qwen-local">Built-in (Qwen 0.8B) — No API key required</option>
        <option value="openrouter">OpenRouter (Remote) — Bring Your Own Key</option>
      </select>
    </div>

    {localSettings.llmProvider === 'qwen-local' && (
      <div className="bg-green-500/10 border border-green-500/30 rounded-lg p-4">
        <p className="text-sm text-muted-foreground">
          <strong>Built-in Model:</strong> Qwen3.5 0.8B runs locally in your browser using WebGPU.
          No API key needed. The model (~850 MB) downloads on first use and is cached for subsequent sessions.
        </p>
      </div>
    )}

    {localSettings.llmProvider === 'openrouter' && (
      <>
        <div className="bg-primary/10 border border-primary/30 rounded-lg p-4">
          <p className="text-sm text-muted-foreground">
            <strong>Bring Your Own Key:</strong> Your API key is stored in your browser's localStorage and never sent to our servers.
          </p>
          <p className="text-sm text-muted-foreground mt-2">
            Get your API key from <a href="https://openrouter.ai/keys" target="_blank" rel="noopener noreferrer" className="text-primary hover:underline">OpenRouter.ai</a>
          </p>
        </div>
        <Input
          type="password"
          label="OpenRouter API Key"
          value={localSettings.apiKey}
          onChange={(e) => setLocalSettings({ ...localSettings, apiKey: e.target.value })}
          placeholder="sk-or-v1-..."
          helperText="Your API key is stored locally in your browser"
        />
        <Input
          type="text"
          label="Model Name"
          value={localSettings.modelName}
          onChange={(e) => setLocalSettings({ ...localSettings, modelName: e.target.value })}
          placeholder="x-ai/grok-4-fast:online"
          helperText="Default model has been tested and works well"
        />
      </>
    )}
  </div>
</section>
```

---

### 4.11 File: `src/stores/qwenStore.ts` (NEW)

Zustand store for Qwen model status. The UI subscribes to this for model loading indicators.

```typescript
import { create } from 'zustand';
import type { ProviderPhase, ProviderDownloadProgress, ProviderStatus } from '@/core/api/types';

interface QwenStoreState {
  phase: ProviderPhase;
  device: string | null;
  downloadProgress: ProviderDownloadProgress | null;
  statusMessage: string;
  error: string | null;

  // Actions
  updateStatus: (status: ProviderStatus) => void;
  reset: () => void;
}

export const useQwenStore = create<QwenStoreState>()((set) => ({
  phase: 'idle',
  device: null,
  downloadProgress: null,
  statusMessage: '',
  error: null,

  updateStatus: (status: ProviderStatus) => set({
    phase: status.phase,
    device: status.device,
    downloadProgress: status.downloadProgress,
    statusMessage: status.statusMessage,
    error: status.error,
  }),

  reset: () => set({
    phase: 'idle',
    device: null,
    downloadProgress: null,
    statusMessage: '',
    error: null,
  }),
}));
```

---

### 4.12 File: `src/hooks/useProvider.ts` (NEW)

React hook that manages provider lifecycle. Replaces `useOpenRouter` as the primary hook for LLM operations.

```typescript
import { useEffect, useCallback, useRef } from 'react';
import { providerRegistry } from '@/core/api/providerRegistry';
import { QwenLocalProvider } from '@/core/api/QwenLocalProvider';
import { useQwenStore } from '@/stores/qwenStore';
import { useUI } from './useUI';
import type { LLMProviderType } from '@/core/api/types';

/**
 * Hook that manages the active LLM provider lifecycle.
 *
 * - Initializes the active provider on mount or when the selection changes
 * - Subscribes to Qwen status updates for UI display
 * - Provides the active provider to the rest of the app
 */
export function useProvider() {
  const { settings, showError } = useUI();
  const qwenStore = useQwenStore();
  const prevProviderRef = useRef<LLMProviderType | null>(null);

  // Sync provider selection with settings and initialize
  useEffect(() => {
    const selectedProvider = settings.llmProvider || 'qwen-local';

    // Skip if already on this provider
    if (prevProviderRef.current === selectedProvider) return;
    prevProviderRef.current = selectedProvider;

    async function activateProvider() {
      try {
        await providerRegistry.setActive(selectedProvider);

        // Subscribe to Qwen status updates if local provider
        if (selectedProvider === 'qwen-local') {
          const qwenProvider = providerRegistry.get('qwen-local') as QwenLocalProvider;
          const unsub = qwenProvider.onStatusChange((status) => {
            qwenStore.updateStatus(status);
          });

          await providerRegistry.initializeActive();
          return unsub;
        } else {
          qwenStore.reset();
          await providerRegistry.initializeActive();
        }
      } catch (err) {
        console.error('Failed to initialize provider:', err);
        showError(`Failed to initialize ${selectedProvider}: ${err}`);
      }
    }

    const cleanupPromise = activateProvider();

    return () => {
      cleanupPromise.then((unsub) => {
        if (typeof unsub === 'function') unsub();
      });
    };
  }, [settings.llmProvider]);

  const isReady = useCallback(() => {
    return providerRegistry.getActive().isReady();
  }, []);

  return {
    provider: providerRegistry.getActive(),
    providerType: providerRegistry.getActiveType(),
    isReady,
    qwenStatus: qwenStore,
  };
}
```

---

### 4.13 Changes to `src/hooks/useOpenRouter.ts` (MODIFY — KEEP FOR BACKWARD COMPAT)

The existing `useOpenRouter` hook is kept but modified to delegate to the provider registry. Any code still importing it continues to work.

```typescript
import { useState, useCallback } from 'react';
import { providerRegistry } from '@/core/api/providerRegistry';
import type { ChatMessage, CompletionResponse } from '@/types/api';
import { APIError } from '@/core/api/errors';
import { useUI } from './useUI';

/**
 * Custom hook for LLM API operations.
 * Now delegates to the active provider from providerRegistry.
 */
export function useOpenRouter() {
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<APIError | null>(null);
  const { showError } = useUI();

  const createCompletion = useCallback(
    async (
      messages: ChatMessage[],
      options?: { temperature?: number; maxTokens?: number }
    ): Promise<CompletionResponse | null> => {
      setIsLoading(true);
      setError(null);

      try {
        const provider = providerRegistry.getActive();
        const response = await provider.createCompletion(messages, options);
        return response;
      } catch (err) {
        const apiError = err instanceof APIError ? err : new APIError(String(err));
        setError(apiError);
        showError(apiError.getUserMessage());
        console.error('API error:', apiError);
        return null;
      } finally {
        setIsLoading(false);
      }
    },
    [showError]
  );

  const validateApiKey = useCallback(async (): Promise<boolean> => {
    // Only meaningful for OpenRouter
    const provider = providerRegistry.getActive();
    if (provider.type !== 'openrouter') return true;

    setIsLoading(true);
    setError(null);
    try {
      // Test with a minimal completion
      await provider.createCompletion([{ role: 'user', content: 'test' }]);
      return true;
    } catch (err) {
      return false;
    } finally {
      setIsLoading(false);
    }
  }, []);

  const clearError = useCallback(() => {
    setError(null);
  }, []);

  return {
    createCompletion,
    validateApiKey,
    clearError,
    isLoading,
    error,
    client: providerRegistry.getActive(),
  };
}
```

---

### 4.14 Changes to `src/core/agents/OrchestratorController.ts` (MODIFY)

The orchestrator must ensure agents use the active provider. Since `BaseAgent` now reads from `providerRegistry.getActive()` by default, the agent constructors need no changes. However, the orchestrator should verify the provider is ready before starting the pipeline.

**Add this check at the start of `processQuery()`:**

```typescript
// At the top of processQuery(), BEFORE existing logic:
import { providerRegistry } from '@/core/api/providerRegistry';

// Inside processQuery():
async processQuery(options: ProcessingOptions): Promise<ProcessingResult> {
  // Verify provider is ready before starting pipeline
  const activeProvider = providerRegistry.getActive();
  if (!activeProvider.isReady()) {
    throw new OrchestrationError(
      `LLM provider "${activeProvider.type}" is not ready. ` +
      (activeProvider.type === 'qwen-local'
        ? 'The local model is still loading. Please wait for the model to finish loading.'
        : 'Please configure your API key in Settings.')
    );
  }

  // ... rest of existing processQuery() code unchanged ...
}
```

Also, the agent constructors in the `OrchestratorController` constructor need no changes because `BaseAgent` now defaults to `providerRegistry.getActive()`. However, to ensure agents always get the latest provider when the orchestrator is reused across provider switches, add a `refreshAgents()` method:

```typescript
/**
 * Recreate all agents with the current active provider.
 * Call this when the user switches providers.
 */
refreshAgents(): void {
  this.registry.cleanup();

  this.horizonMapper = new HorizonMapperAgent();
  this.dialecticAnalyzer = new DialecticAnalyzerAgent();
  this.iterationController = new IterationControllerAgent();
  this.horizonFusion = new HorizonFusionAgent();

  this.registry = new AgentRegistry();
  this.registry.register(this.horizonMapper);
  this.registry.register(this.dialecticAnalyzer);
  this.registry.register(this.iterationController);
  this.registry.register(this.horizonFusion);

  console.info(`Orchestrator agents refreshed with provider: ${providerRegistry.getActiveType()}`);
}
```

---

### 4.15 File: `src/components/features/QwenStatusBar.tsx` (NEW)

A compact status indicator for the Qwen model state. Shown in the main UI when `qwen-local` is the active provider.

```tsx
import React from 'react';
import { useQwenStore } from '@/stores/qwenStore';
import { useUI } from '@/hooks/useUI';

export const QwenStatusBar: React.FC = () => {
  const { settings } = useUI();
  const { phase, device, downloadProgress, statusMessage, error } = useQwenStore();

  // Only show when Qwen is the active provider
  if (settings.llmProvider !== 'qwen-local') return null;

  return (
    <div className="flex items-center gap-2 text-xs text-muted-foreground px-4 py-1 border-b border-border">
      {/* Status dot */}
      <span
        className={`w-2 h-2 rounded-full ${
          phase === 'ready'
            ? 'bg-green-500'
            : phase === 'error'
            ? 'bg-red-500'
            : phase === 'loading' || phase === 'downloading'
            ? 'bg-yellow-500 animate-pulse'
            : 'bg-gray-400'
        }`}
      />

      {/* Status text */}
      {phase === 'ready' && (
        <span>Qwen 0.8B ready ({device})</span>
      )}
      {phase === 'loading' && (
        <span>{statusMessage || 'Loading model...'}</span>
      )}
      {phase === 'downloading' && downloadProgress && (
        <span>
          Downloading: {downloadProgress.file} — {downloadProgress.percent}%
        </span>
      )}
      {phase === 'error' && (
        <span className="text-red-400">Model error: {error}</span>
      )}
      {phase === 'idle' && (
        <span>Model not loaded</span>
      )}
    </div>
  );
};
```

---

## 5. Integration Sequence (Build Order)

Execute these steps in order. Each step must pass before proceeding.

### Step 1: Install Dependencies

```bash
cd Fusion-Horizon-UI
npm install @huggingface/transformers@next onnxruntime-web
npm install -D vite-plugin-static-copy
```

Verify in `package.json` that `@huggingface/transformers` is `^4.x.x` (not `^3.x.x`).

### Step 2: Update `vite.config.ts`

Apply changes from Section 3.2. Run `npm run dev` to verify Vite starts without errors.

### Step 3: Add ORT Bootstrap to `src/main.tsx`

Apply changes from Section 3.3. Run `npm run dev` — the app should load as before (no visual change yet).

### Step 4: Create Provider Abstraction Layer

Create these files in order:
1. `src/core/api/types.ts` (Section 4.1)
2. `src/core/api/OpenRouterProvider.ts` (Section 4.2)

Run `npx tsc --noEmit` to verify types compile.

### Step 5: Create Qwen Worker and Types

Create these files:
1. `src/core/qwen/qwen.types.ts` (Section 4.3)
2. `src/core/qwen/qwen.worker.ts` (Section 4.4)
3. `src/core/qwen/parseThinking.ts` (Section 4.5)

### Step 6: Create QwenLocalProvider

Create `src/core/api/QwenLocalProvider.ts` (Section 4.6).

### Step 7: Create Provider Registry

Create `src/core/api/providerRegistry.ts` (Section 4.7).

### Step 8: Modify BaseAgent

Apply changes from Section 4.8. This is the critical refactoring step. After this change:
- `npx tsc --noEmit` should compile cleanly
- The existing OpenRouter flow should still work (because `providerRegistry.getActive()` defaults to `qwen-local` but `BaseAgent` constructor still runs)

### Step 9: Add UI Settings for Provider Selection

1. Modify `src/stores/uiStore.ts` (Section 4.9)
2. Modify `src/components/features/SettingsPanel.tsx` (Section 4.10)
3. Create `src/stores/qwenStore.ts` (Section 4.11)

### Step 10: Create Provider Hook

Create `src/hooks/useProvider.ts` (Section 4.12).

### Step 11: Update useOpenRouter Hook

Modify `src/hooks/useOpenRouter.ts` (Section 4.13).

### Step 12: Update Orchestrator

Modify `src/core/agents/OrchestratorController.ts` (Section 4.14).

### Step 13: Add QwenStatusBar Component

1. Create `src/components/features/QwenStatusBar.tsx` (Section 4.15)
2. Add `<QwenStatusBar />` to `src/App.tsx` or `src/pages/Home.tsx` above the main content area.

### Step 14: Wire Provider Hook into App

Add `useProvider()` call to the top-level App component or Home page so the provider initializes on mount:

```tsx
// In App.tsx or Home.tsx
import { useProvider } from '@/hooks/useProvider';

function App() {
  useProvider(); // Initializes the active provider on mount
  // ... rest of existing app code
}
```

### Step 15: Test

Run the full test sequence from Section 6.

---

## 6. Testing Scenarios

### Scenario 1: First Launch with Default Provider (Qwen Local)

**Steps:** Clear localStorage, launch the app with `npm run dev`.
**Expected:**
- Settings show "Built-in (Qwen 0.8B)" selected
- QwenStatusBar shows loading/downloading state
- Model downloads (~850 MB, ~30-60 seconds on broadband)
- Status transitions: idle -> loading -> downloading -> ready
- Device shows "webgpu" or "wasm"

### Scenario 2: Submit Query with Qwen Local

**Steps:** After model is ready, submit a hermeneutic query.
**Expected:**
- Pipeline executes all 4 phases (horizon mapping, dialectic analysis, iteration, fusion)
- Each agent receives structured JSON responses from Qwen
- Convergence detection works across cycles
- Final fusion result is displayed

### Scenario 3: Switch to OpenRouter

**Steps:** Open Settings, change provider to OpenRouter, enter API key, save.
**Expected:**
- QwenStatusBar disappears
- API key and model name fields appear
- Submit a query — it uses OpenRouter API
- Pipeline executes identically to before the integration

### Scenario 4: Switch Back to Qwen Local

**Steps:** Open Settings, change provider back to "Built-in (Qwen 0.8B)", save.
**Expected:**
- QwenStatusBar reappears
- Model loads from cache (5-15 seconds, no network download)
- Submit a query — it uses local inference

### Scenario 5: Qwen JSON Output Quality

**Steps:** Submit a complex query and inspect the pipeline's intermediate outputs.
**Expected:**
- HorizonMapperAgent receives valid JSON with assumptions, biases, background_knowledge, user_context
- DialecticAnalyzerAgent receives valid JSON with provisional_whole, part_analysis, tensions, etc.
- If JSON is malformed, the agent's existing fallback parsing (regex extraction) recovers gracefully

### Scenario 6: Offline After First Download

**Steps:** Disconnect from the internet after model has been cached. Submit a query.
**Expected:**
- Model loads from browser cache
- Local inference works without network
- Pipeline completes fully offline

### Scenario 7: WebGPU Fallback to WASM

**Steps:** Run on a machine without WebGPU support.
**Expected:**
- Model loads successfully on WASM backend
- Device reports "wasm"
- Generation works but slower (~5-10 tok/s)

### Scenario 8: Provider Ready Guard

**Steps:** Submit a query before the model finishes loading.
**Expected:**
- OrchestratorController throws `OrchestrationError` with a clear message
- Error is displayed as a toast notification
- No crash, no partial pipeline execution

---

## 7. Performance Expectations

### Local Qwen Inference

| Metric                        | WebGPU            | WASM              |
|-------------------------------|-------------------|--------------------|
| First load (cold, no cache)   | 30-60 seconds     | 40-90 seconds      |
| Subsequent load (warm cache)  | 5-15 seconds      | 10-20 seconds      |
| Token generation speed        | 10-40 tok/s       | 5-10 tok/s         |
| Memory usage                  | ~900 MB VRAM      | ~1.2 GB RAM        |

### Pipeline Timing (per complete hermeneutic circle)

| Phase                | Est. time (WebGPU)  | Est. time (WASM)    |
|----------------------|---------------------|---------------------|
| Horizon mapping      | 10-30 seconds       | 30-60 seconds       |
| Dialectic cycle (x2-5)| 15-40 sec/cycle    | 45-90 sec/cycle     |
| Iteration control    | 5-15 seconds        | 15-30 seconds       |
| Horizon fusion       | 15-40 seconds       | 45-90 seconds       |
| **Total (2 cycles)** | **~60-170 seconds** | **~180-360 seconds**|

The 0.8B model will be significantly slower and produce lower quality output than remote models. This integration is a proof-of-concept to test whether the hermeneutic pipeline functions correctly with a small local model. Quality assessment is a separate research question.

---

## 8. File Checklist

| File                                          | Action  | Size (approx) | Purpose                                    |
|-----------------------------------------------|---------|---------------|--------------------------------------------|
| `src/core/api/types.ts`                       | CREATE  | ~2 KB         | ILLMProvider interface + types              |
| `src/core/api/OpenRouterProvider.ts`          | CREATE  | ~2 KB         | OpenRouter adapter for ILLMProvider         |
| `src/core/api/QwenLocalProvider.ts`           | CREATE  | ~6 KB         | Qwen worker adapter for ILLMProvider        |
| `src/core/api/providerRegistry.ts`            | CREATE  | ~3 KB         | Singleton provider manager                  |
| `src/core/qwen/qwen.types.ts`                | CREATE  | ~2 KB         | Worker protocol types                       |
| `src/core/qwen/qwen.worker.ts`               | CREATE  | ~3 KB         | Web Worker: model load + generation         |
| `src/core/qwen/parseThinking.ts`             | CREATE  | ~0.5 KB       | Think block parser                          |
| `src/stores/qwenStore.ts`                     | CREATE  | ~1 KB         | Zustand store for model status              |
| `src/hooks/useProvider.ts`                    | CREATE  | ~2 KB         | Provider lifecycle hook                     |
| `src/components/features/QwenStatusBar.tsx`   | CREATE  | ~2 KB         | Model status indicator                      |
| `vite.config.ts`                              | MODIFY  | —             | Add WASM copy, esnext target, worker format |
| `src/main.tsx`                                | MODIFY  | —             | Add ORT bootstrap                           |
| `src/core/agents/BaseAgent.ts`                | MODIFY  | —             | Replace OpenRouterClient with ILLMProvider  |
| `src/stores/uiStore.ts`                       | MODIFY  | —             | Add llmProvider to UISettings               |
| `src/components/features/SettingsPanel.tsx`   | MODIFY  | —             | Add provider selection dropdown             |
| `src/hooks/useOpenRouter.ts`                  | MODIFY  | —             | Delegate to providerRegistry                |
| `src/core/agents/OrchestratorController.ts`   | MODIFY  | —             | Add provider ready check + refreshAgents()  |
| `src/App.tsx` or `src/pages/Home.tsx`         | MODIFY  | —             | Add useProvider() + QwenStatusBar           |

**New files:** 10 | **Modified files:** 8 | **Total integration footprint:** ~24 KB new source code

---

## 9. Dependency Summary

### New Runtime Dependencies

| Package                        | Version     | Purpose                                  |
|--------------------------------|-------------|------------------------------------------|
| `@huggingface/transformers`    | ^4.0.0-next | Model loading, tokenization, generation  |
| `onnxruntime-web`              | ^1.21.0     | ONNX inference runtime (WebGPU + WASM)   |

### New Dev Dependencies

| Package                   | Version | Purpose                          |
|---------------------------|---------|----------------------------------|
| `vite-plugin-static-copy` | ^2.0.0  | Copy ORT WASM binaries to public |

### Existing Dependencies (unchanged)

| Package    | Version | Relevance                              |
|------------|---------|----------------------------------------|
| `react`    | ^19.1   | UI framework (already installed)       |
| `zustand`  | ^5.0    | State management (already installed)   |
| `immer`    | ^10.2   | Immutable updates (already installed)  |
| `vite`     | ^7.1    | Build tool (already installed)         |

---

## 10. Known Limitations and Future Considerations

### 0.8B Model Quality

The Qwen3.5-0.8B model has significantly lower capability than large remote models. Expected challenges:
- **JSON reliability** — The model may produce malformed JSON. The existing agents have regex-based fallback parsers that mitigate this.
- **Instruction following** — The model may not follow complex system prompts consistently. The JSON reinforcement wrapper helps but is not guaranteed.
- **Context window** — Limited context means long dialectical cycles may lose earlier context. The existing `trimHistory` pattern applies.

### Not Included in This Plan

- **Model selection UI** — Only one local model (Qwen 0.8B) is supported. Future work could add a model picker.
- **Streaming display** — The pipeline agents collect full responses. Streaming progress per-agent could be added later.
- **Ollama provider** — Could be added as a third `ILLMProvider` implementation using the same abstraction layer.
- **Concurrent generation** — The singleton worker allows only one generation at a time. Pipeline phases are sequential, so this is not a problem for the hermeneutic circle.
