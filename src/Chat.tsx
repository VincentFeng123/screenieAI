import { memo, useDeferredValue, useEffect, useMemo, useRef, useState } from "react";
import { Channel, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import "highlight.js/styles/github-dark.css";
import "./markdown.css";
import "./overlay.css";
import {
  ArrowUp,
  Check,
  Clock,
  Copy,
  MessageSquarePlus,
  RotateCcw,
  X,
} from "lucide-react";
import {
  formatAiMarkdown,
  SCREENIE_KATEX_OPTIONS,
} from "./lib/formatAiMarkdown";
import { SvgInsetBorder } from "./components/Frosted";
import CustomDropdown, {
  type CustomDropdownOption,
} from "./components/CustomDropdown";
import {
  ANTHROPIC_MODELS,
  GEMINI_MODELS,
  OPENAI_MODELS,
  type Provider,
} from "./settings/constants";
import { type HistoryEntry } from "./lib/history";
import HistoryList from "./components/HistoryList";
import {
  estimateCostCents,
  formatUsageSummary,
  recordUsage,
  type AskEvent,
  type ProviderId,
  usageTokensFromEvent,
} from "./lib/usage";

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  usage?: {
    inputTokens: number;
    outputTokens: number;
    provider?: string;
    model?: string;
    costCents: number;
  };
};

type ChatSeed = {
  png_b64: string;
  width: number;
  height: number;
  provider: string;
  model: string;
  messages_json: string;
};

const PANEL_RADIUS = 24;
const CHAT_SESSION_KEY = "screenie.detached_chat_session";

const PROVIDER_LABEL: Record<Provider, { label: string; cloud: boolean }> = {
  anthropic: { label: "Claude", cloud: true },
  openai: { label: "OpenAI", cloud: true },
  gemini: { label: "Gemini", cloud: true },
  ollama: { label: "Ollama", cloud: false },
};

function modelOptionsForProvider(
  provider: Provider,
  currentModel: string,
): CustomDropdownOption[] {
  const withCurrent = (options: CustomDropdownOption[]) => {
    if (!currentModel || options.some((o) => o.value === currentModel)) return options;
    return [{ value: currentModel, label: currentModel }, ...options];
  };
  if (provider === "anthropic") {
    return withCurrent(ANTHROPIC_MODELS.map((m) => ({ value: m.id, label: m.label })));
  }
  if (provider === "openai") {
    return withCurrent(OPENAI_MODELS.map((m) => ({ value: m.id, label: m.label })));
  }
  if (provider === "gemini") {
    return withCurrent(GEMINI_MODELS.map((m) => ({ value: m.id, label: m.label })));
  }
  return withCurrent([
    { value: currentModel || "llama3.2-vision", label: currentModel || "llama3.2-vision" },
  ]);
}

/// Detached chat overlay — borderless, transparent, always-on-top.
/// Visually modeled after the Apple Intelligence / ChatGPT compact chat
/// panel: close button top-left, history/copy/new chat top-right, message
/// stream in the middle (image as a chat bubble at the top), prompt
/// textfield in a rounded rect at the bottom.
export default function Chat() {
  const [seed, setSeed] = useState<ChatSeed | null>(null);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streaming, setStreaming] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [statusToast, setStatusToast] = useState<string | null>(null);
  const [loadStalled, setLoadStalled] = useState(false);
  const [prompt, setPrompt] = useState("");
  const [chatView, setChatView] = useState<"chat" | "history">("chat");
  const [lightbox, setLightbox] = useState<string | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const runSeqRef = useRef(0);
  // Mirror of `streaming` for closures (hydrate/listen) that capture state
  // once at mount.
  const streamingRef = useRef<string | null>(null);
  const stickToBottomRef = useRef(true);

  // Hydrate from Rust state (set by `open_chat_window`). The chat window is
  // reused, so also listen for reseeds after the first mount.
  useEffect(() => {
    let cancelled = false;
    let unlistenFn: (() => void) | null = null;
    const hydrate = async (isReseed: boolean) => {
      // P4-C-reseed: a reseed (new pin from the overlay) replaces the
      // conversation, so any in-flight stream from the previous chat would
      // otherwise keep burning tokens server-side and (per the per-window
      // cancel slot in lib.rs) would still occupy this window's
      // chat_ai_cancel slot. Tear it down before bumping runSeq.
      if (isReseed && streamingRef.current !== null) {
        invoke("cancel_ai").catch((e) => {
          console.error("cancel_ai (reseed) failed:", e);
        });
      }
      const stored = readStoredChatSession();
      let fresh: ChatSeed | null = null;
      try {
        fresh = await invoke<ChatSeed | null>("take_chat_seed");
      } catch {
        // Browser previews and interrupted Tauri sessions can miss the IPC
        // command. The sessionStorage fallback below still lets a reload
        // recover the last detached chat.
      }
      // P-A-R2: on a reseed (chat-seed-changed event), only the freshly-
      // taken Rust seed is trustworthy — the sessionStorage copy is the
      // previous chat's data. Falling back to stored there would silently
      // show the user a stale image with a new conversation thread.
      const s = isReseed ? fresh : (fresh ?? stored?.seed ?? null);
      if (cancelled || !s) return;
      runSeqRef.current += 1;
      setSeed(s);
      setStreaming(null);
      setError(null);
      setChatView("chat");
      try {
        const parsed = JSON.parse(s.messages_json) as ChatMessage[];
        const nextMessages = Array.isArray(parsed) ? parsed : [];
        setMessages(nextMessages);
        writeStoredChatSession(s, nextMessages);
      } catch {
        setMessages([]);
        writeStoredChatSession(s, []);
      }
    };
    void hydrate(false);
    // P4-C-listen: stash the resolved unlisten in a closure-local so the
    // cleanup tears down deterministically even if the listener Promise
    // resolves after the component already unmounted.
    listen("chat-seed-changed", () => {
      if (cancelled) return;
      void hydrate(true);
    })
      .then((fn) => {
        if (cancelled) {
          fn();
          return;
        }
        unlistenFn = fn;
      })
      .catch((e) => {
        console.error("listen(chat-seed-changed) failed:", e);
      });
    return () => {
      cancelled = true;
      if (unlistenFn) unlistenFn();
    };
  }, []);

  useEffect(() => {
    if (seed) {
      setLoadStalled(false);
      return;
    }
    // P-D: chat window is reused — `take_chat_seed` resolves nearly
    // instantly under normal use. The prior 2500 ms gate read as "stuck"
    // when the user reopened the window via Mission Control without a
    // fresh pin. 800 ms still hides the placeholder for the IPC happy
    // path while surfacing the "No chat data" recovery state quickly.
    const t = window.setTimeout(() => setLoadStalled(true), 800);
    return () => window.clearTimeout(t);
  }, [seed]);

  useEffect(() => {
    if (!seed) return;
    writeStoredChatSession(seed, messages);
  }, [seed, messages]);

  // Mirror streaming into a ref so closures captured at mount (the
  // chat-seed-changed listener) can see the current value.
  useEffect(() => {
    streamingRef.current = streaming;
  }, [streaming]);

  // P4-C-scroll: only auto-scroll when the user is already pinned to the
  // bottom. If they scrolled up to read prior context, leave them alone
  // until they return to within ~24px of the bottom.
  //
  // P-A-B2: the `seed` dep is load-bearing. On first paint `seed` is null
  // and the JSX renders the "Loading chat…" placeholder — scrollRef.current
  // is null and the effect bails. Once the seed hydrates and the chat
  // container mounts, the effect must re-run to actually attach the
  // listener; otherwise stickToBottomRef stays true forever and the user
  // can never scroll up mid-stream.
  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    const onScroll = () => {
      const distance = el.scrollHeight - el.scrollTop - el.clientHeight;
      stickToBottomRef.current = distance < 24;
    };
    el.addEventListener("scroll", onScroll, { passive: true });
    return () => el.removeEventListener("scroll", onScroll);
  }, [seed]);

  useEffect(() => {
    const el = scrollRef.current;
    if (!el) return;
    if (!stickToBottomRef.current) return;
    el.scrollTop = el.scrollHeight;
  }, [messages, streaming]);

  // When a brand-new conversation seeds in, force-stick to bottom so the
  // first chunk doesn't leave the user mid-scroll on whatever the previous
  // chat's position was.
  useEffect(() => {
    stickToBottomRef.current = true;
  }, [seed?.png_b64]);


  const closeWindow = () => {
    if (streaming !== null) {
      runSeqRef.current += 1;
      invoke("cancel_ai").catch((e) => {
        console.error("cancel_ai failed:", e);
      });
    }
    getCurrentWindow().close().catch((e) => {
      console.error("close window failed:", e);
    });
  };

  const newChat = () => {
    if (streaming !== null) {
      runSeqRef.current += 1;
      invoke("cancel_ai").catch((e) => {
        console.error("cancel_ai failed:", e);
      });
    }
    setMessages([]);
    setStreaming(null);
    setError(null);
    setStatusToast("Chat cleared");
  };

  const copyConversation = () => {
    if (!seed) return;
    const text = messages
      .map((m) => `${m.role === "user" ? "You:" : "AI:"} ${m.content}`)
      .join("\n\n");
    if (!text) {
      setStatusToast("Nothing to copy yet");
      return;
    }
    navigator.clipboard.writeText(text).then(
      () => setStatusToast("Conversation copied"),
      () => setStatusToast("Copy failed"),
    );
  };

  const updateModel = (next: string) => {
    setSeed((prev) => (prev ? { ...prev, model: next } : prev));
  };

  const provider = (seed?.provider as Provider) ?? "anthropic";
  const providerMeta = PROVIDER_LABEL[provider] ?? PROVIDER_LABEL.anthropic;
  const modelOptions = useMemo(
    () => modelOptionsForProvider(provider, seed?.model ?? ""),
    [provider, seed?.model],
  );

  const runAi = async (history: ChatMessage[]) => {
    if (!seed) return;
    const runId = ++runSeqRef.current;
    setError(null);
    setStreaming("");
    const channel = new Channel<AskEvent>();
    let acc = "";
    const usageBox: { value: { inputTokens: number; outputTokens: number } | null } = {
      value: null,
    };
    channel.onmessage = (event) => {
      if (runSeqRef.current !== runId) return;
      if (event.type === "chunk") {
        acc += event.text;
        setStreaming(acc);
      } else if (event.type === "usage") {
        usageBox.value = usageTokensFromEvent(event);
      }
    };
    try {
      await invoke("ask_ai", {
        provider: seed.provider,
        model: seed.model,
        responseProfile: "concise",
        messages: history,
        imageB64: seed.png_b64,
        onChunk: channel,
      });
      if (runSeqRef.current !== runId) return;
      const usage = usageBox.value;
      const assistant: ChatMessage = {
        role: "assistant",
        content: acc,
        usage: usage
          ? {
              ...usage,
              provider: seed.provider,
              model: seed.model,
              costCents: estimateCostCents(
                seed.provider as ProviderId,
                seed.model,
                usage.inputTokens,
                usage.outputTokens,
              ),
            }
          : undefined,
      };
      setMessages((prev) => [...prev, assistant]);
      setStreaming(null);
      if (usage) {
        recordUsage({
          provider: seed.provider as ProviderId,
          model: seed.model,
          inputTokens: usage.inputTokens,
          outputTokens: usage.outputTokens,
        });
      }
    } catch (e) {
      if (runSeqRef.current !== runId) return;
      const msg = typeof e === "string" ? e : (e as Error).message ?? String(e);
      setError(msg);
      if (acc) {
        setMessages((prev) => [...prev, { role: "assistant", content: acc }]);
      }
      setStreaming(null);
    }
  };

  const sendUser = (text: string) => {
    if (streaming !== null) return;
    if (!text.trim()) return;
    const next: ChatMessage[] = [...messages, { role: "user", content: text }];
    setMessages(next);
    setPrompt("");
    runAi(next);
  };

  /// Re-run the AI from the LAST user turn — drops the most recent assistant
  /// message and asks again. Useful when the previous answer was off.
  const regenerate = () => {
    if (streaming !== null) return;
    let cutoff = -1;
    for (let i = messages.length - 1; i >= 0; i--) {
      if (messages[i].role === "user") {
        cutoff = i;
        break;
      }
    }
    if (cutoff < 0) return;
    const trimmed = messages.slice(0, cutoff + 1);
    setMessages(trimmed);
    runAi(trimmed);
  };

  const openHistoryEntry = (entry: HistoryEntry, fullPng: string) => {
    setSeed((prev) =>
      prev
        ? {
            ...prev,
            png_b64: fullPng,
            width: entry.width,
            height: entry.height,
            provider: entry.provider,
            model: entry.model,
          }
        : {
            png_b64: fullPng,
            width: entry.width,
            height: entry.height,
            provider: entry.provider,
            model: entry.model,
            messages_json: "",
          },
    );
    setMessages([
      { role: "user", content: entry.prompt || "Describe what's shown in this image." },
      { role: "assistant", content: entry.response },
    ]);
    setStreaming(null);
    setError(null);
    setChatView("chat");
  };

  const autoGrow = () => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(140, Math.max(22, ta.scrollHeight)) + "px";
  };
  useEffect(autoGrow, [prompt]);
  useEffect(() => {
    if (!statusToast) return;
    const t = window.setTimeout(() => setStatusToast(null), 1600);
    return () => window.clearTimeout(t);
  }, [statusToast]);

  if (!seed) {
    return (
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          height: "100vh",
          color: "rgba(255,255,255,0.6)",
          fontSize: 13,
          fontFamily: "-apple-system, BlinkMacSystemFont, sans-serif",
          flexDirection: "column",
          gap: 12,
          padding: 24,
          textAlign: "center",
        }}
      >
        <span>
          {loadStalled
            ? "No chat data found. Pin a capture again to reload this window."
            : "Loading chat…"}
        </span>
        {loadStalled && (
          <button
            type="button"
            className="screenie-chat-action"
            onClick={closeWindow}
            aria-label="Close chat window"
          >
            Close
          </button>
        )}
      </div>
    );
  }

  return (
    <div
      style={{
        position: "relative",
        height: "100vh",
        width: "100vw",
        padding: 0,
        margin: 0,
        fontFamily: "-apple-system, BlinkMacSystemFont, sans-serif",
        color: "rgba(255, 255, 255, 0.92)",
        overflow: "hidden",
      }}
    >
      <div
        className="screenie-chat-panel"
        style={{
          width: "100%",
          height: "100%",
          borderRadius: PANEL_RADIUS,
          overflow: "hidden",
          display: "flex",
          flexDirection: "column",
          position: "relative",
        }}
      >
        {/* No BlurredBackdrop bitmap here — the captured-screenshot
            recipe used in the embedded overlay panel is a stale image
            for a detached window. The chat panel's CSS already applies
            `backdrop-filter: blur(22px)` (overlay.css), so what's visible
            through the transparent WebView is blurred live as windows
            move around behind it. */}

        {/* Header — small circular close on the left, history/copy/new
            on the right with flat (no-fill) icon buttons. The strip is a
            drag region; buttons opt out via data-tauri-drag-region="false"
            so clicks don't get hijacked by the drag handler. */}
        <div
          data-tauri-drag-region
          style={{
            position: "relative",
            zIndex: 1,
            height: 38,
            padding: "0 10px",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
          }}
        >
          <button
            type="button"
            className="screenie-close-btn"
            data-tauri-drag-region="false"
            onClick={closeWindow}
            aria-label="Close chat window"
            title="Close chat window"
            onMouseDown={(e) => e.stopPropagation()}
            style={{ transform: "translate(0, 0)" }}
          >
            <X size={10} strokeWidth={2} aria-hidden />
          </button>
          <div style={{ display: "flex", gap: 4 }} data-tauri-drag-region="false">
            <button
              type="button"
              className="screenie-chat-flat-btn"
              data-tauri-drag-region="false"
              onClick={() => setChatView((v) => (v === "history" ? "chat" : "history"))}
              aria-label={chatView === "history" ? "Back to chat" : "Open history"}
              title={chatView === "history" ? "Back to chat" : "Open history"}
              data-active={chatView === "history"}
              onMouseDown={(e) => e.stopPropagation()}
            >
              <Clock size={13} strokeWidth={1.85} aria-hidden />
            </button>
            <button
              type="button"
              className="screenie-chat-flat-btn"
              data-tauri-drag-region="false"
              onClick={copyConversation}
              aria-label="Copy conversation"
              title="Copy conversation"
              onMouseDown={(e) => e.stopPropagation()}
            >
              <Copy size={13} strokeWidth={1.85} aria-hidden />
            </button>
            <button
              type="button"
              className="screenie-chat-flat-btn"
              data-tauri-drag-region="false"
              onClick={newChat}
              aria-label="Start new chat"
              title="Start new chat"
              onMouseDown={(e) => e.stopPropagation()}
            >
              <MessageSquarePlus size={13} strokeWidth={1.85} aria-hidden />
            </button>
          </div>
        </div>

        {chatView === "history" ? (
          <HistoryList
            onOpen={openHistoryEntry}
            emptyState={
              <div className="screenie-history-status">
                No captures yet — your history will show up here.
                <button
                  onClick={() => setChatView("chat")}
                  className="screenie-action"
                  style={{ marginTop: 12, fontSize: 12 }}
                >
                  Back to chat
                </button>
              </div>
            }
          />
        ) : (
          <div
            ref={scrollRef}
            style={{
              flex: 1,
              minHeight: 0,
              overflow: "auto",
              padding: "4px 14px 8px",
              display: "flex",
              flexDirection: "column",
              gap: 14,
              position: "relative",
              zIndex: 1,
            }}
          >
            <ImageMessage b64={seed.png_b64} onOpen={() => setLightbox(seed.png_b64)} />
            {messages.map((m, i) => (
              <ChatBubble
                key={i}
                message={m}
                isLastAssistant={
                  m.role === "assistant" &&
                  i === messages.length - 1 &&
                  streaming === null
                }
                onRegenerate={regenerate}
              />
            ))}
            {streaming !== null && (
              <ChatBubble
                key={messages.length}
                message={{ role: "assistant", content: streaming }}
                streaming
              />
            )}
            {error && (
              <div
                style={{
                  color: "#ff8a8a",
                  fontSize: 12.5,
                  background: "rgba(255, 80, 80, 0.08)",
                  border: "1px solid rgba(255, 80, 80, 0.2)",
                  padding: "8px 10px",
                  borderRadius: 8,
                  lineHeight: 1.45,
                }}
              >
                {error}
              </div>
            )}
          </div>
        )}

        {/* Bottom prompt — outer rounded rect that wraps a textarea +
            send button. Mirrors the screenshot's "Ask anything" pill. */}
        <div
          style={{
            position: "relative",
            zIndex: 1,
            padding: "8px 12px 12px",
          }}
        >
          {statusToast && (
            <div
              role="status"
              style={{
                margin: "0 4px 8px",
                color: "rgba(255,255,255,0.72)",
                fontSize: 12,
                textAlign: "center",
              }}
            >
              {statusToast}
            </div>
          )}
          <div
            className="screenie-chat-prompt"
            onMouseDown={(e) => e.stopPropagation()}
          >
            <textarea
              ref={taRef}
              rows={1}
              value={prompt}
              onChange={(e) => {
                setPrompt(e.target.value);
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault();
                  sendUser(prompt);
                }
              }}
              placeholder="Ask anything"
              style={{
                width: "100%",
                minHeight: 22,
                maxHeight: 140,
                background: "transparent",
                border: "none",
                color: "rgba(255, 255, 255, 0.95)",
                fontSize: 14,
                padding: "10px 14px 4px",
                outline: "none",
                resize: "none",
                lineHeight: 1.4,
                fontFamily: "inherit",
                display: "block",
              }}
            />
            <div
              style={{
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                padding: "4px 10px 8px 14px",
                gap: 8,
              }}
            >
              <div className="screenie-chat-model-select">
                <CustomDropdown
                  value={seed.model}
                  options={modelOptions}
                  onChange={updateModel}
                  ariaLabel={`${providerMeta.label} model`}
                  variant="ghost"
                  disabled={streaming !== null}
                  triggerLabel={
                    <span className="screenie-model-label">
                      <span
                        className={`screenie-model-dot ${
                          providerMeta.cloud ? "cloud" : "local"
                        }`}
                      />
                      <span>{providerMeta.label}</span>
                    </span>
                  }
                />
              </div>
              <button
                className="screenie-send"
                onClick={() => sendUser(prompt)}
                disabled={streaming !== null || !prompt.trim()}
                aria-label="Send"
              >
                <ArrowUp size={15} strokeWidth={2} aria-hidden />
              </button>
            </div>
          </div>
        </div>

        <SvgInsetBorder radius={PANEL_RADIUS} />
      </div>

      {lightbox && <ImageLightbox b64={lightbox} onClose={() => setLightbox(null)} />}
    </div>
  );
}

function readStoredChatSession(): { seed: ChatSeed; messages: ChatMessage[] } | null {
  try {
    const raw = window.sessionStorage.getItem(CHAT_SESSION_KEY);
    if (!raw) return null;
    const parsed = JSON.parse(raw) as {
      seed?: Partial<ChatSeed>;
      messages?: ChatMessage[];
    };
    const seed = parsed.seed;
    if (
      !seed ||
      typeof seed.png_b64 !== "string" ||
      typeof seed.width !== "number" ||
      typeof seed.height !== "number" ||
      typeof seed.provider !== "string" ||
      typeof seed.model !== "string"
    ) {
      return null;
    }
    const messages = Array.isArray(parsed.messages) ? parsed.messages : [];
    return {
      seed: {
        png_b64: seed.png_b64,
        width: seed.width,
        height: seed.height,
        provider: seed.provider,
        model: seed.model,
        messages_json: JSON.stringify(messages),
      },
      messages,
    };
  } catch {
    return null;
  }
}

function writeStoredChatSession(seed: ChatSeed, messages: ChatMessage[]): void {
  try {
    window.sessionStorage.setItem(
      CHAT_SESSION_KEY,
      JSON.stringify({
        seed: { ...seed, messages_json: JSON.stringify(messages) },
        messages,
      }),
    );
  } catch {
    /* storage can be full or unavailable; detached chat still works in-memory */
  }
}

function ImageMessage({ b64, onOpen }: { b64: string; onOpen: () => void }) {
  return (
    <button
      type="button"
      onClick={onOpen}
      aria-label="Open image at full size"
      style={{
        alignSelf: "flex-end",
        maxWidth: "85%",
        background: "transparent",
        border: "none",
        padding: 0,
        margin: 0,
        cursor: "pointer",
        display: "block",
      }}
    >
      <img
        src={`data:image/png;base64,${b64}`}
        alt="Captured region"
        draggable={false}
        style={{
          display: "block",
          maxWidth: "100%",
          maxHeight: 200,
          borderRadius: 10,
          objectFit: "contain",
        }}
      />
    </button>
  );
}

function ImageLightbox({ b64, onClose }: { b64: string; onClose: () => void }) {
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [onClose]);
  return (
    <div
      role="button"
      aria-label="Close image preview"
      onClick={onClose}
      style={{
        position: "fixed",
        inset: 0,
        zIndex: 100,
        background: "rgba(0,0,0,0.78)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
        padding: 24,
        cursor: "zoom-out",
      }}
    >
      <img
        src={`data:image/png;base64,${b64}`}
        alt="Captured region"
        draggable={false}
        style={{ maxWidth: "100%", maxHeight: "100%", borderRadius: 8 }}
        onClick={(e) => e.stopPropagation()}
      />
    </div>
  );
}

// P-E: `ChatBubble` is rendered for every message in the conversation. When
// the latest assistant message is streaming, React used to re-render EVERY
// bubble in the list per chunk (often 200+ Hz on a fast network), since the
// `messages` prop array identity changed when the streaming buffer was
// appended. The custom comparator skips re-render for any bubble whose
// content + usage + role + streaming flag haven't changed. The active
// streaming bubble still re-renders correctly because its `message.content`
// changes each chunk; older bubbles are stable.
function ChatBubbleImpl({
  message,
  streaming,
  isLastAssistant,
  onRegenerate,
}: {
  message: ChatMessage;
  streaming?: boolean;
  isLastAssistant?: boolean;
  onRegenerate?: () => void;
}) {
  const [copied, setCopied] = useState(false);
  const isAssistant = message.role === "assistant";
  const formatted = isAssistant ? formatAiMarkdown(message.content) : message.content;
  // P-E: deprioritize the markdown render during streaming. Same rationale
  // as Overlay.tsx — the ReactMarkdown + remark + rehype-katex + rehype-
  // highlight chain re-parses the entire accumulated stream on every chunk
  // (often 200+ Hz), and useDeferredValue lets React schedule that work at
  // low priority so the UI stays responsive.
  const deferredFormatted = useDeferredValue(formatted);

  if (!isAssistant) {
    // User bubble — round white pill, right-aligned, like the screenshot.
    return (
      <div
        style={{
          alignSelf: "flex-end",
          maxWidth: "85%",
          background: "rgba(253, 252, 248, 0.97)",
          padding: "10px 16px",
          borderRadius: 18,
          fontSize: 13.5,
          lineHeight: 1.4,
          color: "rgba(18, 18, 16, 0.96)",
          whiteSpace: "pre-wrap",
        }}
      >
        {message.content}
      </div>
    );
  }

  // Assistant — text directly on the panel background (no bubble), with a
  // small action-icon row below.
  return (
    <div style={{ alignSelf: "flex-start", width: "100%" }}>
      <div
        className="screenie-md"
        data-density="comfortable"
        style={{
          fontSize: 13.5,
          lineHeight: 1.5,
          color: "rgba(255,255,255,0.92)",
        }}
      >
        {message.content ? (
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[[rehypeKatex, SCREENIE_KATEX_OPTIONS], rehypeHighlight]}
          >
            {deferredFormatted}
          </ReactMarkdown>
        ) : (
          streaming && (
            <span style={{ opacity: 0.55, fontStyle: "italic" }}>Thinking…</span>
          )
        )}
      </div>
      {!streaming && message.content && (
        <div
          style={{
            marginTop: 8,
            display: "flex",
            alignItems: "center",
            gap: 4,
          }}
        >
          <button
            type="button"
            className="screenie-chat-action"
            onClick={() => {
              navigator.clipboard.writeText(message.content).then(
                () => {
                  setCopied(true);
                  setTimeout(() => setCopied(false), 1400);
                },
                () => {},
              );
            }}
            aria-label="Copy response"
          >
            {copied ? (
              <Check size={13} strokeWidth={2} aria-hidden />
            ) : (
              <Copy size={13} strokeWidth={1.85} aria-hidden />
            )}
          </button>
          {isLastAssistant && onRegenerate && (
            <button
              type="button"
              className="screenie-chat-action"
              onClick={onRegenerate}
              aria-label="Regenerate response"
            >
              <RotateCcw size={13} strokeWidth={1.85} aria-hidden />
            </button>
          )}
          {message.usage && (
            <span className="screenie-usage-chip" style={{ marginLeft: 4 }}>
              {formatUsageSummary(message.usage)}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

const ChatBubble = memo(ChatBubbleImpl, (prev, next) => {
  // Stable when role + content + streaming flag + isLastAssistant + usage
  // + onRegenerate identity all match. Streaming bubble re-renders correctly
  // because its `message.content` mutates each chunk.
  return (
    prev.message.role === next.message.role &&
    prev.message.content === next.message.content &&
    prev.message.usage === next.message.usage &&
    prev.streaming === next.streaming &&
    prev.isLastAssistant === next.isLastAssistant &&
    prev.onRegenerate === next.onRegenerate
  );
});
