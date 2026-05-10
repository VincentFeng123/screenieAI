/// User-defined prompt templates. Persisted to localStorage as a JSON array
/// under the key below. The toolbar's preset chips read this list (with
/// the built-in defaults serving as the bootstrap content on first run).

const STORAGE_KEY = "screenie.templates";
const EVENT = "screenie-templates-changed";

export type PromptTemplate = {
  id: string;
  label: string;
  prompt: string;
};

const DEFAULTS: PromptTemplate[] = [
  { id: "explain", label: "Explain", prompt: "Explain what's shown in this image clearly and concisely." },
  { id: "translate", label: "Translate", prompt: "Translate any text in this image to English. Output only the translation." },
  { id: "ocr", label: "OCR", prompt: "Extract all text visible in this image verbatim. Preserve formatting." },
  { id: "summarize", label: "Summarize", prompt: "Summarize the content of this image in 2–3 sentences." },
];

function nextId(): string {
  return `t_${Date.now().toString(36)}_${Math.random().toString(36).slice(2, 7)}`;
}

export function readTemplates(): PromptTemplate[] {
  if (typeof window === "undefined") return DEFAULTS.slice();
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return DEFAULTS.slice();
    const parsed = JSON.parse(raw);
    if (!Array.isArray(parsed)) return DEFAULTS.slice();
    return parsed
      .filter(
        (t): t is PromptTemplate =>
          t &&
          typeof t.id === "string" &&
          typeof t.label === "string" &&
          typeof t.prompt === "string",
      );
  } catch {
    return DEFAULTS.slice();
  }
}

export function writeTemplates(next: PromptTemplate[]): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(next));
  window.dispatchEvent(new CustomEvent(EVENT, { detail: next }));
}

export function subscribeTemplates(
  listener: (next: PromptTemplate[]) => void,
): () => void {
  if (typeof window === "undefined") return () => {};
  const onEvent = (event: Event) => {
    const detail = (event as CustomEvent<PromptTemplate[]>).detail;
    listener(detail ?? readTemplates());
  };
  const onStorage = (event: StorageEvent) => {
    if (event.key === STORAGE_KEY) listener(readTemplates());
  };
  window.addEventListener(EVENT, onEvent);
  window.addEventListener("storage", onStorage);
  return () => {
    window.removeEventListener(EVENT, onEvent);
    window.removeEventListener("storage", onStorage);
  };
}

export function addTemplate(label: string, prompt: string): PromptTemplate {
  const list = readTemplates();
  const t: PromptTemplate = { id: nextId(), label, prompt };
  writeTemplates([...list, t]);
  return t;
}

export function updateTemplate(
  id: string,
  patch: Partial<Omit<PromptTemplate, "id">>,
): void {
  const list = readTemplates();
  writeTemplates(
    list.map((t) =>
      t.id === id ? { ...t, ...patch } : t,
    ),
  );
}

export function deleteTemplate(id: string): void {
  writeTemplates(readTemplates().filter((t) => t.id !== id));
}

export function resetTemplates(): void {
  writeTemplates(DEFAULTS.slice());
}

/// A special template id understood by the toolbar: when its `id` matches
/// this value, the toolbar treats sending as "OCR → clipboard" (bypasses
/// the AI chat — runs the prompt, copies the response text to clipboard,
/// fires a transient toast).
export const OCR_CLIPBOARD_TEMPLATE_ID = "ocr-clipboard";

/// A special template that's always available even if the user clears
/// their list. Adds the OCR-to-clipboard chip to the row.
export const OCR_CLIPBOARD_TEMPLATE: PromptTemplate = {
  id: OCR_CLIPBOARD_TEMPLATE_ID,
  label: "OCR → Clipboard",
  prompt:
    "Extract every piece of visible text from this image, in reading order. " +
    "Output only the extracted text — no commentary, no markdown formatting, " +
    "no labels. Preserve original line breaks where reasonable.",
};
