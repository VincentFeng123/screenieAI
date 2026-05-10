import { invoke } from "@tauri-apps/api/core";

export type HistoryEntry = {
  id: string;
  created_at_ms: number;
  provider: string;
  model: string;
  prompt: string;
  response: string;
  width: number;
  height: number;
};

export async function listHistory(): Promise<HistoryEntry[]> {
  return await invoke<HistoryEntry[]>("list_history");
}

export async function deleteHistoryEntry(id: string): Promise<void> {
  await invoke("delete_history_entry", { id });
}

export async function clearHistory(): Promise<void> {
  await invoke("clear_history");
}

export async function loadHistoryImage(id: string): Promise<string> {
  return await invoke<string>("load_history_image", { id });
}

export async function saveHistoryEntry(args: {
  pngB64: string;
  width: number;
  height: number;
  provider: string;
  model: string;
  prompt: string;
  response: string;
}): Promise<HistoryEntry | null> {
  try {
    return await invoke<HistoryEntry>("add_history_entry", {
      pngB64: args.pngB64,
      width: args.width,
      height: args.height,
      provider: args.provider,
      model: args.model,
      prompt: args.prompt,
      response: args.response,
    });
  } catch (e) {
    console.error("add_history_entry failed:", e);
    return null;
  }
}

/// Derive a single-line summary from a history entry by joining a truncated
/// prompt with a truncated start-of-response. Total budget is ~10 words:
/// prompt takes up to 4, response gets whatever's left (6 in the typical
/// case, more if the prompt was shorter). Each side is cropped at the word
/// level so we never chop mid-word, and trailing "…" signals truncation.
/// Markdown noise (code fences, inline code, headings, bold/italic, links,
/// math delimiters) is stripped first so the summary reads as plain prose.
export function deriveHistoryTitle(entry: HistoryEntry): string {
  const stripMd = (s: string) =>
    s
      .replace(/```[\s\S]*?```/g, " ")
      .replace(/`[^`]+`/g, "")
      .replace(/^#{1,6}\s+/gm, "")
      .replace(/\*\*([^*]+)\*\*/g, "$1")
      .replace(/(?<!\\)\*([^*]+)\*/g, "$1")
      .replace(/\[([^\]]+)\]\([^)]+\)/g, "$1")
      .replace(/\$\$[^$]+\$\$/g, " ")
      .replace(/\$[^$]+\$/g, "")
      .replace(/\s+/g, " ")
      .trim();

  const PROMPT_MAX = 4;
  const TOTAL_MAX = 10;

  const promptWords = stripMd(entry.prompt).split(/\s+/).filter(Boolean);
  const responseWords = stripMd(entry.response).split(/\s+/).filter(Boolean);

  const promptUsed = promptWords.slice(0, PROMPT_MAX);
  const responseBudget = TOTAL_MAX - promptUsed.length;
  const responseUsed = responseWords.slice(0, Math.max(0, responseBudget));

  const format = (used: string[], total: number) =>
    used.length === 0 ? "" : used.join(" ") + (total > used.length ? "…" : "");

  const prompt = format(promptUsed, promptWords.length);
  const response = format(responseUsed, responseWords.length);

  if (prompt && response) return `${prompt} — ${response}`;
  return prompt || response || "Untitled capture";
}
