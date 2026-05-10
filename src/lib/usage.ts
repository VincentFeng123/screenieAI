/// Token + cost tracking for the AI providers. Aggregates per-month
/// running totals in localStorage so the Settings → Overview can show
/// "$X.XX this month" without a separate database. Each ask_ai call
/// produces ONE Usage event; the renderer calls `recordUsage` once per
/// finished assistant turn.

export type AskEvent =
  | { type: "chunk"; text: string }
  | {
      type: "usage";
      inputTokens?: number;
      outputTokens?: number;
      input_tokens?: number;
      output_tokens?: number;
    };

export type ProviderId = "anthropic" | "openai" | "gemini" | "ollama";

const STORAGE_KEY = "screenie.usage";
const EVENT = "screenie-usage-changed";

/// Per-thousand-tokens prices (USD). Values are approximate as of 2026
/// and may drift; the user can override per-model later if needed. The
/// keys are matched by checking if the model id starts with the prefix
/// (so "claude-3-5-sonnet-20241022" matches "claude-3-5-sonnet").
type RateRow = {
  modelPrefix: string;
  /// Cents per 1k input tokens.
  inCents: number;
  /// Cents per 1k output tokens.
  outCents: number;
};

const RATES: Record<ProviderId, RateRow[]> = {
  anthropic: [
    { modelPrefix: "claude-opus-4-7", inCents: 0.5, outCents: 2.5 },
    { modelPrefix: "claude-sonnet-4-6", inCents: 0.3, outCents: 1.5 },
    { modelPrefix: "claude-opus-4", inCents: 1.5, outCents: 7.5 },
    { modelPrefix: "claude-sonnet-4", inCents: 0.3, outCents: 1.5 },
    { modelPrefix: "claude-haiku-4", inCents: 0.1, outCents: 0.5 },
    { modelPrefix: "claude-3-7-sonnet", inCents: 0.3, outCents: 1.5 },
    { modelPrefix: "claude-3-5-sonnet", inCents: 0.3, outCents: 1.5 },
    { modelPrefix: "claude-3-5-haiku", inCents: 0.08, outCents: 0.4 },
    { modelPrefix: "claude-3-opus", inCents: 1.5, outCents: 7.5 },
    { modelPrefix: "claude-3-sonnet", inCents: 0.3, outCents: 1.5 },
    { modelPrefix: "claude-3-haiku", inCents: 0.025, outCents: 0.125 },
    { modelPrefix: "claude-", inCents: 0.3, outCents: 1.5 }, // catch-all
  ],
  openai: [
    { modelPrefix: "gpt-5.5", inCents: 0.5, outCents: 3.0 },
    { modelPrefix: "gpt-5.4-mini", inCents: 0.075, outCents: 0.45 },
    { modelPrefix: "gpt-5.4-nano", inCents: 0.02, outCents: 0.125 },
    { modelPrefix: "gpt-5.4", inCents: 0.25, outCents: 1.5 },
    { modelPrefix: "gpt-5.2", inCents: 0.175, outCents: 1.4 },
    { modelPrefix: "gpt-5-mini", inCents: 0.025, outCents: 0.2 },
    { modelPrefix: "gpt-5-nano", inCents: 0.005, outCents: 0.04 },
    { modelPrefix: "gpt-5", inCents: 0.125, outCents: 1.0 },
    { modelPrefix: "gpt-4o-mini", inCents: 0.015, outCents: 0.06 },
    { modelPrefix: "gpt-4o", inCents: 0.25, outCents: 1.0 },
    { modelPrefix: "gpt-4.1-nano", inCents: 0.01, outCents: 0.04 },
    { modelPrefix: "gpt-4.1-mini", inCents: 0.04, outCents: 0.16 },
    { modelPrefix: "gpt-4.1", inCents: 0.2, outCents: 0.8 },
    { modelPrefix: "gpt-4-turbo", inCents: 1.0, outCents: 3.0 },
    { modelPrefix: "chatgpt-4o", inCents: 0.5, outCents: 1.5 },
    { modelPrefix: "o3", inCents: 0.2, outCents: 0.8 },
    { modelPrefix: "gpt-", inCents: 0.25, outCents: 1.0 }, // catch-all
  ],
  gemini: [
    { modelPrefix: "gemini-2.5-pro", inCents: 0.125, outCents: 0.5 },
    { modelPrefix: "gemini-2.5-flash-lite", inCents: 0.0075, outCents: 0.03 },
    { modelPrefix: "gemini-2.5-flash", inCents: 0.03, outCents: 0.25 },
    { modelPrefix: "gemini-2.0-flash-lite", inCents: 0.0075, outCents: 0.03 },
    { modelPrefix: "gemini-2.0-flash", inCents: 0.01, outCents: 0.04 },
    { modelPrefix: "gemini-1.5-pro", inCents: 0.125, outCents: 0.5 },
    { modelPrefix: "gemini-1.5-flash-8b", inCents: 0.00375, outCents: 0.015 },
    { modelPrefix: "gemini-1.5-flash", inCents: 0.0075, outCents: 0.03 },
    { modelPrefix: "gemini-", inCents: 0.0075, outCents: 0.03 },
  ],
  ollama: [{ modelPrefix: "", inCents: 0, outCents: 0 }],
};

function rateFor(provider: ProviderId, model: string): RateRow {
  const candidates = RATES[provider] ?? RATES.ollama;
  for (const r of candidates) {
    if (model.startsWith(r.modelPrefix)) return r;
  }
  return { modelPrefix: "", inCents: 0, outCents: 0 };
}

/// Compute the cost in cents for a given (provider, model, tokens) tuple.
/// Returns a fractional value (rounded to 4 decimal places) so small
/// queries don't round to zero in the running total.
export function estimateCostCents(
  provider: ProviderId,
  model: string,
  inputTokens: number,
  outputTokens: number,
): number {
  const r = rateFor(provider, model);
  const cents =
    (inputTokens / 1000) * r.inCents + (outputTokens / 1000) * r.outCents;
  return Math.round(cents * 10000) / 10000;
}

export function formatTokens(n: number): string {
  if (n < 1000) return `${n}`;
  if (n < 10000) return `${(n / 1000).toFixed(1)}k`;
  return `${Math.round(n / 1000)}k`;
}

export function formatCostCents(cents: number): string {
  if (cents <= 0) return "—";
  if (cents < 1) {
    // Sub-cent: show four sig figs.
    return `< 1¢`;
  }
  if (cents < 100) return `${cents.toFixed(cents < 10 ? 2 : 1)}¢`;
  return `$${(cents / 100).toFixed(2)}`;
}

export function usageTokensFromEvent(
  event: AskEvent,
): { inputTokens: number; outputTokens: number } | null {
  if (event.type !== "usage") return null;
  const inputTokens = event.inputTokens ?? event.input_tokens;
  const outputTokens = event.outputTokens ?? event.output_tokens;
  if (
    typeof inputTokens !== "number" ||
    typeof outputTokens !== "number" ||
    !Number.isFinite(inputTokens) ||
    !Number.isFinite(outputTokens)
  ) {
    return null;
  }
  return { inputTokens, outputTokens };
}

type ContextLimitRow = {
  provider: ProviderId;
  modelPrefix: string;
  tokens: number;
};

const CONTEXT_LIMITS: ContextLimitRow[] = [
  { provider: "anthropic", modelPrefix: "claude-opus-4-7", tokens: 1_000_000 },
  { provider: "anthropic", modelPrefix: "claude-sonnet-4-6", tokens: 1_000_000 },
  { provider: "anthropic", modelPrefix: "claude-opus-4", tokens: 200_000 },
  { provider: "anthropic", modelPrefix: "claude-sonnet-4", tokens: 200_000 },
  { provider: "anthropic", modelPrefix: "claude-haiku-4", tokens: 200_000 },
  { provider: "anthropic", modelPrefix: "claude-3", tokens: 200_000 },
  { provider: "openai", modelPrefix: "gpt-5.5", tokens: 1_000_000 },
  { provider: "openai", modelPrefix: "gpt-5.4-mini", tokens: 400_000 },
  { provider: "openai", modelPrefix: "gpt-5.4-nano", tokens: 400_000 },
  { provider: "openai", modelPrefix: "gpt-5.4", tokens: 1_000_000 },
  { provider: "openai", modelPrefix: "gpt-5.2-chat", tokens: 128_000 },
  { provider: "openai", modelPrefix: "gpt-5", tokens: 400_000 },
  { provider: "openai", modelPrefix: "gpt-4o", tokens: 128_000 },
  { provider: "openai", modelPrefix: "gpt-4.1", tokens: 1_000_000 },
  { provider: "openai", modelPrefix: "gpt-4-turbo", tokens: 128_000 },
  { provider: "openai", modelPrefix: "chatgpt-4o", tokens: 128_000 },
  { provider: "openai", modelPrefix: "o3", tokens: 200_000 },
  { provider: "gemini", modelPrefix: "gemini-3", tokens: 1_000_000 },
  { provider: "gemini", modelPrefix: "gemini-2.5", tokens: 1_000_000 },
  { provider: "gemini", modelPrefix: "gemini-2.0", tokens: 1_000_000 },
  { provider: "gemini", modelPrefix: "gemini-1.5", tokens: 1_000_000 },
];

function contextLimitFor(provider: ProviderId, model: string): number | null {
  const row = CONTEXT_LIMITS.find(
    (candidate) =>
      candidate.provider === provider && model.startsWith(candidate.modelPrefix),
  );
  return row?.tokens ?? null;
}

export function formatUsageSummary(args: {
  provider?: string;
  model?: string;
  inputTokens: number;
  outputTokens: number;
  costCents: number;
}): string {
  const used = args.inputTokens + args.outputTokens;
  const provider = args.provider as ProviderId | undefined;
  const limit =
    provider && args.model ? contextLimitFor(provider, args.model) : null;
  const left =
    limit && limit > used ? `${formatTokens(limit - used)} left` : null;
  const cost = formatCostCents(args.costCents);

  return [
    `${formatTokens(used)} used`,
    left,
    `${formatTokens(args.inputTokens)} in`,
    `${formatTokens(args.outputTokens)} out`,
    cost,
  ]
    .filter(Boolean)
    .join(" · ");
}

export type ProviderTotals = {
  inputTokens: number;
  outputTokens: number;
  costCents: number;
  calls: number;
};

export type UsageStore = Record<string, Record<string, ProviderTotals>>;
//                          ^ "YYYY-MM"  ^ provider id

function readStore(): UsageStore {
  if (typeof window === "undefined") return {};
  try {
    const raw = window.localStorage.getItem(STORAGE_KEY);
    if (!raw) return {};
    const parsed = JSON.parse(raw);
    return parsed && typeof parsed === "object" ? parsed : {};
  } catch {
    return {};
  }
}

function writeStore(store: UsageStore): void {
  if (typeof window === "undefined") return;
  window.localStorage.setItem(STORAGE_KEY, JSON.stringify(store));
  window.dispatchEvent(new CustomEvent(EVENT, { detail: store }));
}

export function currentMonthKey(): string {
  const d = new Date();
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}`;
}

export function recordUsage(args: {
  provider: ProviderId;
  model: string;
  inputTokens: number;
  outputTokens: number;
}): void {
  const cents = estimateCostCents(
    args.provider,
    args.model,
    args.inputTokens,
    args.outputTokens,
  );
  const store = readStore();
  const month = currentMonthKey();
  const monthMap = (store[month] = store[month] ?? {});
  const entry: ProviderTotals = monthMap[args.provider] ?? {
    inputTokens: 0,
    outputTokens: 0,
    costCents: 0,
    calls: 0,
  };
  entry.inputTokens += args.inputTokens;
  entry.outputTokens += args.outputTokens;
  entry.costCents = Math.round((entry.costCents + cents) * 10000) / 10000;
  entry.calls += 1;
  monthMap[args.provider] = entry;
  writeStore(store);
}

export function readMonth(monthKey: string = currentMonthKey()):
  | Record<string, ProviderTotals>
  | null {
  return readStore()[monthKey] ?? null;
}

export function readAllMonths(): UsageStore {
  return readStore();
}

export function subscribeUsage(listener: (store: UsageStore) => void): () => void {
  if (typeof window === "undefined") return () => {};
  const onEvent = (event: Event) => {
    const detail = (event as CustomEvent<UsageStore>).detail;
    listener(detail ?? readStore());
  };
  const onStorage = (event: StorageEvent) => {
    if (event.key === STORAGE_KEY) listener(readStore());
  };
  window.addEventListener(EVENT, onEvent);
  window.addEventListener("storage", onStorage);
  return () => {
    window.removeEventListener(EVENT, onEvent);
    window.removeEventListener("storage", onStorage);
  };
}

export function clearMonth(monthKey: string): void {
  const store = readStore();
  if (store[monthKey]) {
    delete store[monthKey];
    writeStore(store);
  }
}

export function clearAll(): void {
  writeStore({});
}
