export type Provider = "anthropic" | "openai" | "gemini" | "ollama";

export type OllamaStatus = { running: boolean; models: string[] };

export const VISION_MODEL_PULL = "ollama pull llama3.2-vision";

// All models below are vision-capable text-output models — that's the minimum
// the app needs (we send a screenshot, expect streamed text back). Embedding,
// audio, image-generation, and text-only-no-vision models are intentionally
// omitted because they will fail at the first request.

export const ANTHROPIC_MODELS = [
  { id: "claude-sonnet-4-6", label: "Claude Sonnet 4.6 — recommended" },
  { id: "claude-opus-4-7", label: "Claude Opus 4.7 — most capable" },
  { id: "claude-haiku-4-5-20251001", label: "Claude Haiku 4.5 — fast" },
  { id: "claude-sonnet-4-5-20250929", label: "Claude Sonnet 4.5" },
  { id: "claude-opus-4-1-20250805", label: "Claude Opus 4.1 — deep reasoning" },
  { id: "claude-sonnet-4-20250514", label: "Claude Sonnet 4" },
  { id: "claude-3-7-sonnet-20250219", label: "Claude Sonnet 3.7" },
  { id: "claude-3-5-haiku-20241022", label: "Claude Haiku 3.5" },
  { id: "claude-3-haiku-20240307", label: "Claude 3 Haiku" },
];

// Vision-capable models only. o4-mini was removed because it doesn't accept
// image input; o3 is kept because it does. The Rust client switches to
// `max_completion_tokens` for o-series and GPT-5-family model ids.
export const OPENAI_MODELS = [
  { id: "gpt-5.5", label: "GPT-5.5 — latest flagship" },
  { id: "gpt-5.4", label: "GPT-5.4" },
  { id: "gpt-5.4-mini", label: "GPT-5.4 mini — fast" },
  { id: "gpt-5.4-nano", label: "GPT-5.4 nano — cheapest" },
  { id: "gpt-5.2", label: "GPT-5.2" },
  { id: "gpt-5.2-chat-latest", label: "GPT-5.2 Chat latest" },
  { id: "gpt-5-mini", label: "GPT-5 mini" },
  { id: "gpt-5-nano", label: "GPT-5 nano" },
  { id: "gpt-5", label: "GPT-5" },
  { id: "gpt-4.1", label: "GPT-4.1 — non-reasoning" },
  { id: "gpt-4.1-mini", label: "GPT-4.1 mini — fast" },
  { id: "gpt-4.1-nano", label: "GPT-4.1 nano — cheapest" },
  { id: "gpt-4o", label: "GPT-4o" },
  { id: "gpt-4o-mini", label: "GPT-4o mini — fast + cheap" },
  { id: "gpt-4o-2024-11-20", label: "GPT-4o (2024-11-20)" },
  { id: "gpt-4o-2024-08-06", label: "GPT-4o (2024-08-06)" },
  { id: "gpt-4o-2024-05-13", label: "GPT-4o (2024-05-13)" },
  { id: "gpt-4o-mini-2024-07-18", label: "GPT-4o mini (2024-07-18)" },
  { id: "chatgpt-4o-latest", label: "ChatGPT-4o latest" },
  { id: "gpt-4-turbo", label: "GPT-4 Turbo" },
  { id: "gpt-4-turbo-2024-04-09", label: "GPT-4 Turbo (2024-04-09)" },
  { id: "o3", label: "o3 — reasoning + vision" },
];

export const GEMINI_MODELS = [
  { id: "gemini-3-flash-preview", label: "Gemini 3 Flash Preview — latest" },
  { id: "gemini-3.1-pro-preview", label: "Gemini 3.1 Pro Preview" },
  { id: "gemini-3.1-flash-lite", label: "Gemini 3.1 Flash-Lite — stable" },
  { id: "gemini-2.5-flash", label: "Gemini 2.5 Flash — fast + capable" },
  { id: "gemini-2.5-pro", label: "Gemini 2.5 Pro — best reasoning" },
  { id: "gemini-2.5-flash-lite", label: "Gemini 2.5 Flash Lite — cheapest" },
  { id: "gemini-2.0-flash", label: "Gemini 2.0 Flash" },
  { id: "gemini-2.0-flash-lite", label: "Gemini 2.0 Flash Lite" },
  { id: "gemini-1.5-pro", label: "Gemini 1.5 Pro" },
  { id: "gemini-1.5-pro-002", label: "Gemini 1.5 Pro 002" },
  { id: "gemini-1.5-flash", label: "Gemini 1.5 Flash" },
  { id: "gemini-1.5-flash-002", label: "Gemini 1.5 Flash 002" },
  { id: "gemini-1.5-flash-8b", label: "Gemini 1.5 Flash 8B" },
];

export function looksLikeVisionModel(name: string): boolean {
  const n = name.toLowerCase();
  return /llava|vision|bakllava|qwen.*-vl|moondream|cogvlm/i.test(n);
}
