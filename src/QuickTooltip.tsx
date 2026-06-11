import {
  useCallback,
  useDeferredValue,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type MouseEvent as ReactMouseEvent,
} from "react";
import { Channel, invoke } from "@tauri-apps/api/core";
import { getCurrentWindow } from "@tauri-apps/api/window";
import {
  AlertTriangle,
  ArrowUp,
  Bot,
  Camera,
  MessageCircle,
  Plus,
  Settings,
  ShieldCheck,
  Square,
  Video,
  X,
} from "lucide-react";
import { revealItemInDir } from "@tauri-apps/plugin-opener";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import "highlight.js/styles/github-dark.css";
import "./markdown.css";
import "./overlay.css";
import "./quick-tooltip.css";
import screenieLogoUrl from "../src-tauri/icons/tray-icon.svg";
import {
  ANTHROPIC_MODELS,
  GEMINI_MODELS,
  OPENAI_MODELS,
  type Provider,
} from "./settings/constants";
import { readPreferences } from "./settings/preferences";
import {
  formatAiMarkdown,
  SCREENIE_KATEX_OPTIONS,
} from "./lib/formatAiMarkdown";
import {
  recordUsage,
  type AskEvent,
  type ProviderId,
  usageTokensFromEvent,
} from "./lib/usage";
import { SvgInsetBorder } from "./components/Frosted";
import CustomDropdown, {
  type CustomDropdownOption,
} from "./components/CustomDropdown";
import { useVoiceSession } from "./voice/useVoiceSession";
import VoiceMicButton, { type VoiceMicState } from "./voice/VoiceMicButton";
import VoiceLevelMeter from "./voice/VoiceLevelMeter";
import VoiceTranscriptFeed from "./voice/VoiceTranscriptFeed";
import VoiceModelDownloadCard from "./voice/VoiceModelDownloadCard";

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

type AgentAction = {
  action: string;
  app?: string;
  id?: number;
  text?: string;
  combo?: string;
  path?: string[];
  question?: string;
  script?: string;
  name?: string;
  file?: string;
  dx?: number;
  dy?: number;
  ms?: number;
  url?: string;
  query?: string;
  reason?: string;
};

type AgentTarget = {
  id: number;
  role: string;
  name: string;
  value?: string | null;
};

type AgentConfirmationRequest = {
  requestId: string;
  action: AgentAction;
  target?: AgentTarget | null;
  reason: string;
};

type AgentQuestionRequest = {
  requestId: string;
  question: string;
  options: string[];
};

type AgentAutonomy = "ask" | "confirm" | "auto";

const AGENT_AUTONOMY_STORAGE_KEY = "agent_autonomy";

const AGENT_AUTONOMY_OPTIONS: CustomDropdownOption[] = [
  { value: "ask", label: "Ask everything" },
  { value: "confirm", label: "Confirm risky" },
  { value: "auto", label: "Full auto" },
];

function readAgentAutonomy(): AgentAutonomy {
  const saved = localStorage.getItem(AGENT_AUTONOMY_STORAGE_KEY);
  return saved === "ask" || saved === "auto" ? saved : "confirm";
}

function readAgentScriptingEnabled(): boolean {
  return localStorage.getItem("agent_scripting_enabled") === "true";
}

// Default ON, unlike scripting: the lookup sends only app + feature names.
function readAgentWebLookupEnabled(): boolean {
  return localStorage.getItem("agent_web_lookup_enabled") !== "false";
}

type AgentTaskFinished = {
  status?: string;
  failureReason?: string | null;
};

type AgentStepUpdate = {
  step: number;
  phase?: "planned" | "completed";
  reason?: string | null;
  action: AgentAction;
  mechanism?: string | null;
  durationMs?: number | null;
  target?: string | null;
  verification?: string | null;
};

const AGENT_MECHANISM_LABELS: Record<string, string> = {
  axPress: "ax",
  axSetValue: "ax",
  menuPress: "menu",
  syntheticClick: "click",
  clipboardPaste: "paste",
  syntheticInput: "keys",
  uiSearch: "search",
  webLookup: "web",
  capture: "capture",
  record: "record",
};

type AgentRecordingState = {
  active: boolean;
  scope?: string;
  startedAtMs?: number;
};

type AgentSavedClip = {
  path: string;
  format: string;
  durationMs: number;
  bytes?: number;
};

function clipFileName(path: string): string {
  return path.split("/").pop() ?? path;
}

function agentStepDetail(update: AgentStepUpdate): string {
  const parts: string[] = [];
  if (update.phase === "completed") {
    const mechanism = update.mechanism
      ? AGENT_MECHANISM_LABELS[update.mechanism] ?? update.mechanism
      : null;
    if (mechanism) parts.push(mechanism);
    if (typeof update.durationMs === "number") {
      parts.push(`${update.durationMs}ms`);
    }
  }
  return parts.length > 0 ? ` (${parts.join(" · ")})` : "";
}

type ProviderInfo = {
  provider: Provider;
  label: string;
  cloud: boolean;
  model: string;
};

type TooltipFrostRegion = {
  x: number;
  y: number;
  w: number;
  h: number;
  radius: number;
};

const QUICK_TOOLTIP_FROST_REGION_SELECTOR = [
  ".quick-tooltip-shell",
  ".quick-tooltip-status-card",
  ".quick-tooltip-chat-panel",
  ".quick-tooltip-agent-composer",
  ".quick-tooltip-voice-feed",
  ".screenie-select-menu-portal",
].join(", ");
const QUICK_TOOLTIP_STATUS_CONTENT_SELECTOR =
  ".quick-tooltip-error, .quick-tooltip-confirmation, .quick-tooltip-question, .quick-tooltip-agent-progress, .quick-tooltip-clip";
const QUICK_TOOLTIP_STATUS_MIN_HEIGHT = 82;
const QUICK_TOOLTIP_STATUS_MAX_HEIGHT = 420;
// Must match QUICK_TOOLTIP_AGENT_CARD_H / _MAX_H in src-tauri/src/lib.rs.
const QUICK_TOOLTIP_AGENT_CARD_MIN_HEIGHT = 106;
const QUICK_TOOLTIP_AGENT_CARD_MAX_HEIGHT = 340;

const QUICK_TOOLTIP_DRAG_BLOCKERS =
  'button, input, textarea, select, a, [role="button"], [role="listbox"], .screenie-select, .screenie-select-menu-portal, .quick-tooltip-agent-card';

const PROVIDER_LABELS: Record<
  Provider,
  { label: string; cloud: boolean; defaultModel: string }
> = {
  anthropic: { label: "Claude", cloud: true, defaultModel: "claude-sonnet-4-6" },
  openai: { label: "OpenAI", cloud: true, defaultModel: "gpt-4o" },
  gemini: { label: "Gemini", cloud: true, defaultModel: "gemini-2.5-flash" },
  ollama: { label: "Ollama", cloud: false, defaultModel: "llama3.2-vision" },
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

function modelStorageKey(provider: Provider): string {
  if (provider === "ollama") return "ollama_model";
  if (provider === "openai") return "openai_model";
  if (provider === "gemini") return "gemini_model";
  return "anthropic_model";
}

function storedCloudModel(
  key: string,
  options: Array<{ id: string }>,
  fallback: string,
): string {
  const saved = localStorage.getItem(key);
  return saved && options.some((option) => option.id === saved) ? saved : fallback;
}

function readProviderInfo(): ProviderInfo {
  const savedProvider = localStorage.getItem("provider");
  const provider: Provider =
    savedProvider && savedProvider in PROVIDER_LABELS
      ? (savedProvider as Provider)
      : "anthropic";
  const meta = PROVIDER_LABELS[provider];
  let model = meta.defaultModel;
  if (provider === "ollama") {
    model = localStorage.getItem("ollama_model") || meta.defaultModel;
  } else if (provider === "openai") {
    model = storedCloudModel("openai_model", OPENAI_MODELS, meta.defaultModel);
  } else if (provider === "gemini") {
    model = storedCloudModel("gemini_model", GEMINI_MODELS, meta.defaultModel);
  } else {
    model = storedCloudModel("anthropic_model", ANTHROPIC_MODELS, meta.defaultModel);
  }
  return { provider, label: meta.label, cloud: meta.cloud, model };
}

function errorText(error: unknown): string {
  if (typeof error === "string") return error;
  if (error instanceof Error) return error.message;
  return String(error);
}

function formatAgentAction(action: AgentAction): string {
  if (action.action === "activateApp" && action.app) return `Open ${action.app}`;
  if (action.action === "key" && action.combo) return `Press ${action.combo}`;
  if (action.action === "type") return "Type text";
  if (action.action === "doubleClick") return "Double-click";
  if (action.action === "click") return "Click";
  if (action.action === "scroll") return "Scroll";
  if (action.action === "wait") return "Wait";
  if (action.action === "openUrl") return action.url ? `Open ${action.url}` : "Open URL";
  if (action.action === "webSearch")
    return action.query ? `Search "${action.query}"` : "Search the web";
  if (action.action === "readPage") return "Read page";
  if (action.action === "menu" && action.path) return `Menu: ${action.path.join(" → ")}`;
  if (action.action === "ask") return "Ask you a question";
  if (action.action === "applescript") return "Run AppleScript";
  if (action.action === "shortcut")
    return action.name ? `Run shortcut "${action.name}"` : "Run a shortcut";
  if (action.action === "moveToTrash")
    return action.file ? `Move to Trash: ${action.file}` : "Move a file to Trash";
  if (action.action === "done") return "Finish task";
  return action.action || "Action";
}

function formatAgentTarget(target?: AgentTarget | null): string {
  if (!target) return "Current app";
  const name = target.name?.trim();
  if (name) return name;
  return target.role || `Element ${target.id}`;
}

function selectedModelLabel(options: CustomDropdownOption[], value: string): string {
  return options.find((option) => option.value === value)?.label ?? value;
}

function clamp(value: number, min: number, max: number): number {
  return Math.min(Math.max(value, min), max);
}

function parseRadius(raw: string, w: number, h: number): number {
  const trimmed = raw.trim();
  if (trimmed.endsWith("%")) {
    const pct = parseFloat(trimmed);
    if (!Number.isFinite(pct)) return 0;
    return (pct / 100) * Math.min(w, h);
  }
  const n = parseFloat(trimmed);
  return Number.isFinite(n) ? n : 0;
}

function collectTooltipFrostRegions(): TooltipFrostRegion[] {
  const regions: TooltipFrostRegion[] = [];
  const viewportW = window.innerWidth;
  const viewportH = window.innerHeight;

  document
    .querySelectorAll<HTMLElement>(QUICK_TOOLTIP_FROST_REGION_SELECTOR)
    .forEach((el) => {
      const style = window.getComputedStyle(el);
      if (style.display === "none" || style.visibility === "hidden") return;
      if (parseFloat(style.opacity || "1") < 0.05) return;

      const rect = el.getBoundingClientRect();
      const x1 = clamp(rect.left, 0, viewportW);
      const y1 = clamp(rect.top, 0, viewportH);
      const x2 = clamp(rect.right, 0, viewportW);
      const y2 = clamp(rect.bottom, 0, viewportH);
      const w = x2 - x1;
      const h = y2 - y1;
      if (w < 1 || h < 1) return;

      const radius = parseRadius(style.borderTopLeftRadius, w, h);
      regions.push({ x: x1, y: y1, w, h, radius });
    });

  return regions;
}

function frostSignature(regions: TooltipFrostRegion[]): string {
  return JSON.stringify(
    {
      viewportW: Math.round(window.innerWidth),
      viewportH: Math.round(window.innerHeight),
      regions: regions.map((r) => ({
        x: Math.round(r.x),
        y: Math.round(r.y),
        w: Math.round(r.w),
        h: Math.round(r.h),
        radius: Math.round(r.radius),
      })),
    },
  );
}

function useQuickTooltipFrostRegions(enabled: boolean) {
  const lastSignatureRef = useRef("");
  const enabledRef = useRef(enabled);
  const rafRef = useRef<number | null>(null);
  const frameSyncedRef = useRef(false);
  enabledRef.current = enabled;

  const cancelScheduledSync = useCallback(() => {
    if (rafRef.current === null) return;
    window.cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
  }, []);

  const syncNow = useCallback(() => {
    const regions = enabledRef.current ? collectTooltipFrostRegions() : [];
    const signature = frostSignature(regions);
    if (signature === lastSignatureRef.current) return;
    lastSignatureRef.current = signature;
    invoke("set_quick_tooltip_vibrancy_regions", { regions }).catch((e) => {
      console.error("set_quick_tooltip_vibrancy_regions failed:", e);
      // The signature was stored optimistically; a lost send must not
      // dedupe-suppress the retry that would repair the panes.
      lastSignatureRef.current = "";
    });
  }, []);

  const scheduleSync = useCallback(() => {
    cancelScheduledSync();
    rafRef.current = window.requestAnimationFrame(() => {
      rafRef.current = null;
      syncNow();
    });
  }, [cancelScheduledSync, syncNow]);

  useLayoutEffect(() => {
    if (frameSyncedRef.current) return;
    frameSyncedRef.current = true;
    syncNow();
    window.requestAnimationFrame(() => {
      frameSyncedRef.current = false;
    });
  });

  useEffect(() => {
    const ro = new ResizeObserver(scheduleSync);
    document
      .querySelectorAll(QUICK_TOOLTIP_FROST_REGION_SELECTOR)
      .forEach((el) => ro.observe(el));

    const observer = new MutationObserver((mutations) => {
      let meaningful = false;
      for (const m of mutations) {
        m.addedNodes.forEach((n) => {
          if (!(n instanceof HTMLElement)) return;
          if (n.matches?.(QUICK_TOOLTIP_FROST_REGION_SELECTOR)) ro.observe(n);
          n.querySelectorAll?.(QUICK_TOOLTIP_FROST_REGION_SELECTOR).forEach((el) =>
            ro.observe(el),
          );
        });
        if (meaningful) continue;
        const target = m.target as Element | null;
        if (!target?.closest?.(".screenie-md")) {
          meaningful = true;
        }
      }
      if (meaningful) scheduleSync();
    });

    observer.observe(document.body, {
      attributes: true,
      childList: true,
      subtree: true,
      attributeFilter: ["class", "style", "data-expanded"],
    });
    window.addEventListener("resize", scheduleSync);
    return () => {
      cancelScheduledSync();
      ro.disconnect();
      observer.disconnect();
      window.removeEventListener("resize", scheduleSync);
      invoke("set_quick_tooltip_vibrancy_regions", { regions: [] }).catch(() => {});
      lastSignatureRef.current = "";
    };
  }, [cancelScheduledSync, scheduleSync]);
}

export default function QuickTooltip() {
  const [expanded, setExpanded] = useState(false);
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [streaming, setStreaming] = useState<string | null>(null);
  const [prompt, setPrompt] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [agentInputOpen, setAgentInputOpen] = useState(false);
  const [agentModelMenuOpen, setAgentModelMenuOpen] = useState(false);
  const [agentGoal, setAgentGoal] = useState("");
  const [agentRunning, setAgentRunning] = useState(false);
  const [agentStatus, setAgentStatus] = useState<AgentStepUpdate | null>(null);
  const [recordingActive, setRecordingActive] = useState(false);
  const [savedClip, setSavedClip] = useState<AgentSavedClip | null>(null);
  const [confirmation, setConfirmation] =
    useState<AgentConfirmationRequest | null>(null);
  const [question, setQuestion] = useState<AgentQuestionRequest | null>(null);
  const [questionAnswer, setQuestionAnswer] = useState("");
  const [autonomy, setAutonomy] = useState<AgentAutonomy>(() =>
    readAgentAutonomy(),
  );
  const [providerInfo, setProviderInfo] = useState<ProviderInfo>(() =>
    readProviderInfo(),
  );
  const questionInputRef = useRef<HTMLInputElement>(null);
  const runSeqRef = useRef(0);
  const scrollRef = useRef<HTMLDivElement>(null);
  const taRef = useRef<HTMLTextAreaElement>(null);
  const agentInputRef = useRef<HTMLTextAreaElement>(null);
  const statusCardRef = useRef<HTMLElement>(null);
  const agentInputOpenRef = useRef(false);
  const agentFormRef = useRef<HTMLDivElement>(null);
  const [statusHeight, setStatusHeight] = useState(
    QUICK_TOOLTIP_STATUS_MIN_HEIGHT,
  );
  const [agentCardHeight, setAgentCardHeight] = useState(
    QUICK_TOOLTIP_AGENT_CARD_MIN_HEIGHT,
  );
  const voice = useVoiceSession({
    getStartArgs: () => {
      const info = readProviderInfo();
      return {
        provider: info.provider,
        model: info.model,
        visionProvider: info.provider,
        visionModel: info.model,
        autonomy: readAgentAutonomy(),
        scriptingEnabled: readAgentScriptingEnabled(),
        webLookupEnabled: readAgentWebLookupEnabled(),
      };
    },
  });
  // Refs for the mount-once listeners and DOM handlers that must observe
  // the live voice state without re-registering.
  const voiceActiveRef = useRef(false);
  voiceActiveRef.current = voice.active;
  const voiceBusyRef = useRef(false);
  voiceBusyRef.current = voice.busy;
  const modelOptions = useMemo(
    () => modelOptionsForProvider(providerInfo.provider, providerInfo.model),
    [providerInfo.provider, providerInfo.model],
  );
  const activeModelLabel = useMemo(
    () => selectedModelLabel(modelOptions, providerInfo.model),
    [modelOptions, providerInfo.model],
  );
  const hasStatus = Boolean(
    error || confirmation || question || agentRunning || savedClip,
  );
  const chatVisible = expanded && !hasStatus;
  const voiceMicState: VoiceMicState = voice.micDenied
    ? "denied"
    : voice.modelMissing
      ? "model-missing"
      : voice.status === "transcribing"
        ? "transcribing"
        : voice.status === "listening"
          ? "listening"
          : "idle";
  const canStartNewChat =
    messages.length > 0 || prompt.trim().length > 0 || streaming !== null || error !== null;
  const rootStyle = useMemo(
    () =>
      ({
        "--quick-tooltip-status-h": `${Math.round(statusHeight)}px`,
        "--quick-tooltip-agent-card-h": `${Math.round(agentCardHeight)}px`,
      }) as CSSProperties,
    [statusHeight, agentCardHeight],
  );

  useQuickTooltipFrostRegions(true);

  useEffect(() => {
    agentInputOpenRef.current = agentInputOpen;
  }, [agentInputOpen]);

  useEffect(() => {
    const onStorage = () => {
      setProviderInfo(readProviderInfo());
      setAutonomy(readAgentAutonomy());
    };
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  useEffect(() => {
    let cancelled = false;
    let unlisten: (() => void) | null = null;
    getCurrentWindow()
      .listen<AgentConfirmationRequest>("agent-confirmation-requested", (event) => {
        if (cancelled) return;
        setConfirmation(event.payload);
        setExpanded(false);
        agentInputOpenRef.current = false;
        setAgentInputOpen(false);
        setAgentModelMenuOpen(false);
        invoke("set_quick_tooltip_keyboard_mode", { enabled: false }).catch((e) => {
          console.error("set_quick_tooltip_keyboard_mode(false) failed:", e);
        });
      })
      .then((off) => {
        if (cancelled) {
          off();
        } else {
          unlisten = off;
        }
      })
      .catch((e) => {
        console.error("agent confirmation listener failed:", e);
      });
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    let unlisten: (() => void) | null = null;
    getCurrentWindow()
      .listen<AgentQuestionRequest>("agent-question-requested", (event) => {
        if (cancelled) return;
        setQuestion(event.payload);
        setQuestionAnswer("");
        setExpanded(false);
        agentInputOpenRef.current = false;
        setAgentInputOpen(false);
        setAgentModelMenuOpen(false);
        // Free-text answers need keystrokes routed to the tooltip.
        invoke("set_quick_tooltip_keyboard_mode", { enabled: true }).catch((e) => {
          console.error("set_quick_tooltip_keyboard_mode(true) failed:", e);
        });
        window.setTimeout(() => questionInputRef.current?.focus(), 60);
      })
      .then((off) => {
        if (cancelled) {
          off();
        } else {
          unlisten = off;
        }
      })
      .catch((e) => {
        console.error("agent question listener failed:", e);
      });
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    let unlisten: (() => void) | null = null;
    getCurrentWindow()
      .listen<AgentTaskFinished>("agent-task-finished", (event) => {
        if (cancelled) return;
        setAgentRunning(false);
        setAgentStatus(null);
        // Defensive: the run-end cleanup stops any session, so the pill must
        // never outlive the run even if the state event was missed.
        setRecordingActive(false);
        setQuestion((current) => {
          if (current) {
            invoke("set_quick_tooltip_keyboard_mode", { enabled: false }).catch(
              () => {},
            );
          }
          return null;
        });
        const status = event.payload?.status;
        const failure = event.payload?.failureReason?.trim();
        if ((status === "failed" || status === "maxStepsReached") && failure) {
          // Voice runs report failure on their chip; the error card would
          // overlap the open agent card.
          if (!voiceBusyRef.current && !voiceActiveRef.current) {
            setError(failure);
            setExpanded(false);
          } else {
            console.warn("voice agent task failed:", failure);
          }
        }
      })
      .then((off) => {
        if (cancelled) {
          off();
        } else {
          unlisten = off;
        }
      })
      .catch((e) => {
        console.error("agent task finished listener failed:", e);
      });
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    let unlisten: (() => void) | null = null;
    getCurrentWindow()
      .listen<AgentStepUpdate>("agent-step-update", (event) => {
        if (cancelled || !event.payload) return;
        setAgentStatus(event.payload);
      })
      .then((off) => {
        if (cancelled) {
          off();
        } else {
          unlisten = off;
        }
      })
      .catch((e) => {
        console.error("agent step update listener failed:", e);
      });
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    let unlisten: (() => void) | null = null;
    getCurrentWindow()
      .listen<AgentRecordingState>("agent-recording-state", (event) => {
        if (cancelled) return;
        setRecordingActive(Boolean(event.payload?.active));
      })
      .then((off) => {
        if (cancelled) {
          off();
        } else {
          unlisten = off;
        }
      })
      .catch((e) => {
        console.error("agent recording state listener failed:", e);
      });
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, []);

  useEffect(() => {
    let cancelled = false;
    let unlisten: (() => void) | null = null;
    getCurrentWindow()
      .listen<AgentSavedClip>("agent-clip-saved", (event) => {
        if (cancelled || !event.payload?.path) return;
        setSavedClip(event.payload);
      })
      .then((off) => {
        if (cancelled) {
          off();
        } else {
          unlisten = off;
        }
      })
      .catch((e) => {
        console.error("agent clip listener failed:", e);
      });
    return () => {
      cancelled = true;
      unlisten?.();
    };
  }, []);

  const setQuickTooltipKeyboardMode = useCallback(
    async (enabled: boolean, restorePreviousApp = false) => {
      try {
        await invoke("set_quick_tooltip_keyboard_mode", {
          enabled,
          restorePreviousApp,
        });
      } catch (e) {
        console.error(
          `set_quick_tooltip_keyboard_mode(${enabled}) failed:`,
          e,
        );
        throw e;
      }
    },
    [],
  );

  const restoreQuickTooltipKeyboardMode = useCallback(() => {
    setQuickTooltipKeyboardMode(false).catch(() => {});
  }, [setQuickTooltipKeyboardMode]);

  // Inline chip editing mirrors the main input's keyboard claim/restore:
  // claim on focus; on exit, hand keys back to the target app only during a
  // hands-free session (same condition as the input's onBlur).
  const handleChipEditFocus = useCallback(() => {
    setQuickTooltipKeyboardMode(true).catch(() => {});
  }, [setQuickTooltipKeyboardMode]);

  const handleChipEditBlur = useCallback(() => {
    if (voiceActiveRef.current) {
      restoreQuickTooltipKeyboardMode();
    }
    // Outside a hands-free session the open card keeps the keyboard —
    // deliberately no-op.
  }, [restoreQuickTooltipKeyboardMode]);

  const closeAgentInput = useCallback(() => {
    if (!agentInputOpenRef.current) return;
    agentInputOpenRef.current = false;
    setAgentInputOpen(false);
    setAgentModelMenuOpen(false);
    restoreQuickTooltipKeyboardMode();
  }, [restoreQuickTooltipKeyboardMode]);

  const openAgentInput = useCallback(() => {
    agentInputOpenRef.current = true;
    setError(null);
    setExpanded(false);
    setAgentModelMenuOpen(false);
    setAgentInputOpen(true);
  }, []);

  const measureStatusHeight = useCallback(() => {
    const card = statusCardRef.current;
    const content = card?.querySelector<HTMLElement>(
      QUICK_TOOLTIP_STATUS_CONTENT_SELECTOR,
    );
    if (!content) return;
    const next = clamp(
      Math.ceil(content.scrollHeight),
      QUICK_TOOLTIP_STATUS_MIN_HEIGHT,
      QUICK_TOOLTIP_STATUS_MAX_HEIGHT,
    );
    setStatusHeight((current) =>
      Math.abs(current - next) < 1 ? current : next,
    );
  }, []);

  useLayoutEffect(() => {
    if (!hasStatus) {
      setStatusHeight(QUICK_TOOLTIP_STATUS_MIN_HEIGHT);
      return;
    }

    measureStatusHeight();
    const card = statusCardRef.current;
    if (!card) return;

    const ro = new ResizeObserver(measureStatusHeight);
    ro.observe(card);
    const content = card.querySelector<HTMLElement>(
      QUICK_TOOLTIP_STATUS_CONTENT_SELECTOR,
    );
    if (content) ro.observe(content);

    const raf = window.requestAnimationFrame(measureStatusHeight);
    return () => {
      window.cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, [agentRunning, agentStatus, confirmation, question, error, hasStatus, measureStatusHeight]);

  useEffect(() => {
    if (!confirmation) return;
    const id = window.setTimeout(() => {
      setConfirmation((current) =>
        current?.requestId === confirmation.requestId ? null : current,
      );
    }, 30_000);
    return () => window.clearTimeout(id);
  }, [confirmation]);

  // Auto-grow the goal textarea with its content. Collapse to 0 before
  // reading scrollHeight so the measurement is pure content + padding —
  // "auto" resolves differently for the placeholder vs real text in WebKit,
  // which made the empty card a different height than the typed one. CSS
  // min/max-height clamp the result (42px single line .. ~5 lines), and the
  // form's ResizeObserver grows the card + native window to follow.
  useLayoutEffect(() => {
    if (!agentInputOpen) return;
    const el = agentInputRef.current;
    if (!el) return;
    el.style.height = "0px";
    el.style.height = `${el.scrollHeight}px`;
  }, [agentInputOpen, agentGoal]);

  // The agent card grows with the voice feed (statusHeight pattern): the
  // form is content-sized, the measurement drives both the CSS var and the
  // native window height.
  const measureAgentCard = useCallback(() => {
    const form = agentFormRef.current;
    if (!form) return;
    const next = clamp(
      Math.ceil(form.scrollHeight),
      QUICK_TOOLTIP_AGENT_CARD_MIN_HEIGHT,
      QUICK_TOOLTIP_AGENT_CARD_MAX_HEIGHT,
    );
    setAgentCardHeight((current) =>
      Math.abs(current - next) < 1 ? current : next,
    );
  }, []);

  useLayoutEffect(() => {
    if (!agentInputOpen) {
      setAgentCardHeight(QUICK_TOOLTIP_AGENT_CARD_MIN_HEIGHT);
      return;
    }
    measureAgentCard();
    const form = agentFormRef.current;
    if (!form) return;
    const ro = new ResizeObserver(measureAgentCard);
    ro.observe(form);
    const raf = window.requestAnimationFrame(measureAgentCard);
    return () => {
      window.cancelAnimationFrame(raf);
      ro.disconnect();
    };
  }, [agentInputOpen, measureAgentCard]);

  // Reopen the agent card (chips, mic state) once a confirmation/question/
  // error card resolves during a hands-free session — those listeners close
  // it to make room for the status card.
  const hadStatusRef = useRef(hasStatus);
  useEffect(() => {
    const had = hadStatusRef.current;
    hadStatusRef.current = hasStatus;
    if (had && !hasStatus && voiceActiveRef.current && !agentInputOpenRef.current) {
      openAgentInput();
    }
  }, [hasStatus, openAgentInput]);

  const toggleVoice = useCallback(async () => {
    if (voiceActiveRef.current) {
      await voice.stop();
      // Restore the card-open invariant (open card = keyboard mode on) that
      // the voice session suspended — unless voice tasks are still running,
      // in which case keys must stay with the app the agent is driving.
      if (agentInputOpenRef.current && !voiceBusyRef.current) {
        setQuickTooltipKeyboardMode(true).catch(() => {});
      }
      return;
    }
    const started = await voice.start();
    if (started) {
      // Hands-free: keystrokes stay with the previously-active app so the
      // agent types into it, not the tooltip. Clicking the goal input
      // re-enables keyboard mode via its onFocus handler.
      restoreQuickTooltipKeyboardMode();
    }
  }, [voice, restoreQuickTooltipKeyboardMode, setQuickTooltipKeyboardMode]);

  useEffect(() => {
    invoke("resize_quick_tooltip", {
      expanded: chatVisible,
      agentInputOpen,
      agentModelMenuOpen,
      statusOpen: hasStatus,
      statusHeight: hasStatus ? statusHeight : undefined,
      agentCardHeight: agentInputOpen ? agentCardHeight : undefined,
    }).catch((e) => {
        console.error("resize_quick_tooltip failed:", e);
    });
  }, [agentCardHeight, agentInputOpen, agentModelMenuOpen, chatVisible, hasStatus, statusHeight]);

  useEffect(() => {
    if (!chatVisible) return;
    const id = window.setTimeout(() => taRef.current?.focus(), 80);
    return () => window.clearTimeout(id);
  }, [chatVisible]);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, streaming, chatVisible]);

  useEffect(() => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(96, Math.max(22, ta.scrollHeight)) + "px";
  }, [prompt]);

  useEffect(() => {
    if (!agentInputOpen) return;
    // Hands-free reopen (voice active): leave keystrokes with the target
    // app and don't steal focus — the agent may be mid-task. Clicking the
    // input still enables keyboard mode via its onFocus handler.
    if (voiceActiveRef.current) return;
    let cancelled = false;
    let keyboardModeEnabled = false;
    let focusTimer: number | null = null;
    setError(null);
    setQuickTooltipKeyboardMode(true)
      .then(() => {
        keyboardModeEnabled = true;
      })
      .catch((e) => {
        if (!cancelled) setError(errorText(e));
      })
      .finally(() => {
        if (cancelled || !agentInputOpenRef.current) {
          if (keyboardModeEnabled) {
            setQuickTooltipKeyboardMode(false).catch(() => {});
          }
          return;
        }
        focusTimer = window.setTimeout(() => {
          if (agentInputOpenRef.current) {
            agentInputRef.current?.focus();
          }
        }, 40);
      });
    return () => {
      cancelled = true;
      if (focusTimer !== null) {
        window.clearTimeout(focusTimer);
      }
    };
  }, [agentInputOpen, setQuickTooltipKeyboardMode]);

  useEffect(() => {
    if (!agentInputOpen) return;
    const onPointerDown = (event: PointerEvent) => {
      const target = event.target;
      if (!(target instanceof HTMLElement)) return;
      if (
        target.closest(
          ".quick-tooltip-shell, .quick-tooltip-agent-card, .screenie-select-menu-portal",
        )
      ) {
        return;
      }
      // Hands-free sessions keep the card (and its chips) up while the user
      // works elsewhere; the Bot button still closes it explicitly.
      if (voiceActiveRef.current) return;
      closeAgentInput();
    };
    document.addEventListener("pointerdown", onPointerDown, true);
    return () => document.removeEventListener("pointerdown", onPointerDown, true);
  }, [agentInputOpen, closeAgentInput]);

  const cancelStream = () => {
    if (streaming === null) return;
    runSeqRef.current += 1;
    invoke("cancel_ai").catch((e) => {
      console.error("cancel_ai failed:", e);
    });
    setMessages((prev) =>
      streaming.trim()
        ? [...prev, { role: "assistant", content: streaming }]
        : prev,
    );
    setStreaming(null);
  };

  const newChat = () => {
    if (streaming !== null) {
      runSeqRef.current += 1;
      invoke("cancel_ai").catch(() => {});
    }
    setMessages([]);
    setStreaming(null);
    setError(null);
    setPrompt("");
  };

  const toggleAgentInput = () => {
    if (agentInputOpenRef.current) {
      closeAgentInput();
      return;
    }
    openAgentInput();
  };

  const updateModel = (model: string) => {
    localStorage.setItem(modelStorageKey(providerInfo.provider), model);
    setProviderInfo(readProviderInfo());
  };

  const updateAutonomy = (value: string) => {
    const next: AgentAutonomy =
      value === "ask" || value === "auto" ? value : "confirm";
    localStorage.setItem(AGENT_AUTONOMY_STORAGE_KEY, next);
    setAutonomy(next);
  };

  const runAi = async (history: ChatMessage[]) => {
    const info = readProviderInfo();
    setProviderInfo(info);
    const runId = ++runSeqRef.current;
    setError(null);
    setStreaming("");

    const channel = new Channel<AskEvent>();
    let acc = "";
    const usageBox: {
      value: { inputTokens: number; outputTokens: number } | null;
    } = { value: null };
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
        provider: info.provider,
        model: info.model,
        responseProfile: readPreferences().aiResponseStyle,
        messages: history,
        imageB64: "",
        onChunk: channel,
      });
      if (runSeqRef.current !== runId) return;
      setMessages((prev) => [...prev, { role: "assistant", content: acc }]);
      setStreaming(null);
      const usage = usageBox.value;
      if (usage) {
        recordUsage({
          provider: info.provider as ProviderId,
          model: info.model,
          inputTokens: usage.inputTokens,
          outputTokens: usage.outputTokens,
        });
      }
    } catch (e) {
      if (runSeqRef.current !== runId) return;
      setError(errorText(e));
      if (acc.trim()) {
        setMessages((prev) => [...prev, { role: "assistant", content: acc }]);
      }
      setStreaming(null);
    }
  };

  const sendUser = (text: string) => {
    if (streaming !== null) return;
    const content = text.trim();
    if (!content) return;
    const next = [...messages, { role: "user" as const, content }];
    setMessages(next);
    setPrompt("");
    void runAi(next);
  };

  const startPillDrag = (event: ReactMouseEvent<HTMLElement>) => {
    if (event.button !== 0) return;
    const target = event.target;
    if (!(target instanceof HTMLElement)) return;
    if (target.closest(QUICK_TOOLTIP_DRAG_BLOCKERS)) return;
    event.preventDefault();
    getCurrentWindow().startDragging().catch((e) => {
      console.error("startDragging failed:", e);
    });
  };

  const screenshotAndAsk = async () => {
    if (streaming !== null) cancelStream();
    closeAgentInput();
    setExpanded(false);
    try {
      await invoke("resize_quick_tooltip", { expanded: false });
      await invoke("start_capture");
    } catch (e) {
      console.error("start_capture failed:", e);
      setError(errorText(e));
    }
  };

  const openSettings = async () => {
    if (streaming !== null) cancelStream();
    closeAgentInput();
    setExpanded(false);
    try {
      await invoke("resize_quick_tooltip", { expanded: false });
      await invoke("show_settings_window");
    } catch (e) {
      console.error("show_settings_window failed:", e);
      setError(errorText(e));
    }
  };

  const respondToConfirmation = async (approved: boolean) => {
    const requestId = confirmation?.requestId;
    if (!requestId) return;
    setConfirmation(null);
    try {
      await invoke("respond_to_confirmation", { requestId, approved });
    } catch (e) {
      console.error("respond_to_confirmation failed:", e);
      setError(errorText(e));
    }
  };

  const respondToQuestion = async (answer: string) => {
    const requestId = question?.requestId;
    const trimmed = answer.trim();
    if (!requestId || !trimmed) return;
    setQuestion(null);
    setQuestionAnswer("");
    restoreQuickTooltipKeyboardMode();
    try {
      await invoke("respond_to_user_question", { requestId, answer: trimmed });
    } catch (e) {
      console.error("respond_to_user_question failed:", e);
      setError(errorText(e));
    }
  };

  const stopAgentTask = async () => {
    try {
      await invoke("stop_agent_task");
      setAgentRunning(false);
      setAgentStatus(null);
    } catch (e) {
      console.error("stop_agent_task failed:", e);
      setError(errorText(e));
    }
  };

  const submitAgentGoal = async () => {
    const goal = agentGoal.trim();
    if (!goal || agentRunning) return;
    const info = readProviderInfo();
    setProviderInfo(info);
    agentInputOpenRef.current = false;
    setAgentInputOpen(false);
    setAgentModelMenuOpen(false);
    restoreQuickTooltipKeyboardMode();
    setAgentGoal("");
    setAgentRunning(true);
    setAgentStatus(null);
    setError(null);
    // A new run replaces the previous run's saved-clip card.
    setSavedClip(null);
    try {
      await invoke("start_agent_task", {
        goal,
        provider: info.provider,
        model: info.model,
        visionProvider: info.provider,
        visionModel: info.model,
        autonomy: readAgentAutonomy(),
        scriptingEnabled: readAgentScriptingEnabled(),
        webLookupEnabled: readAgentWebLookupEnabled(),
      });
    } catch (e) {
      setAgentRunning(false);
      setError(errorText(e));
      setExpanded(false);
    }
  };

  return (
    <div
      className="quick-tooltip-root"
      data-expanded={chatVisible}
      data-agent-input-open={agentInputOpen}
      data-status-open={hasStatus}
      style={rootStyle}
    >
      <section
        className="quick-tooltip-shell screenie-toolbar"
        onMouseDown={startPillDrag}
        aria-label="Screenie AI quick tooltip"
      >
        <div className="quick-tooltip-bar">
          <div className="quick-tooltip-brand" aria-hidden>
            <img src={screenieLogoUrl} alt="" className="quick-tooltip-logo" />
          </div>
          <div className="quick-tooltip-actions">
            <button
              type="button"
              className="quick-tooltip-icon-btn"
              onClick={() => {
                void screenshotAndAsk();
              }}
              aria-label="Screenshot and ask"
              title="Screenshot and ask"
            >
              <Camera size={16} strokeWidth={1.9} aria-hidden />
            </button>
            <button
              type="button"
              className="quick-tooltip-icon-btn"
              data-active={expanded}
              onClick={() => {
                closeAgentInput();
                setExpanded((v) => !v);
              }}
              aria-label={expanded ? "Collapse ask" : "Ask"}
              title={expanded ? "Collapse ask" : "Ask"}
            >
              <MessageCircle size={16} strokeWidth={1.9} aria-hidden />
            </button>
            <button
              type="button"
              className="quick-tooltip-icon-btn"
              data-active={agentInputOpen}
              data-running={agentRunning || voice.active}
              onClick={toggleAgentInput}
              aria-label="Agent task"
              title="Agent task"
            >
              <Bot size={16} strokeWidth={1.9} aria-hidden />
            </button>
            <button
              type="button"
              className="quick-tooltip-icon-btn"
              onClick={() => {
                void openSettings();
              }}
              aria-label="Settings"
              title="Settings"
            >
              <Settings size={16} strokeWidth={1.9} aria-hidden />
            </button>
          </div>
        </div>
        <SvgInsetBorder radius={999} strokeAlpha={0.2} />
      </section>

      {agentInputOpen && (
        <section
          className="quick-tooltip-agent-card screenie-toolbar"
          aria-label="Agent task command"
        >
          <div
            ref={agentFormRef}
            className="quick-tooltip-agent-form"
            onMouseDown={(e) => e.stopPropagation()}
          >
            <div className="quick-tooltip-agent-composer">
              <div className="quick-tooltip-agent-row">
                <textarea
                  ref={agentInputRef}
                  value={agentGoal}
                  rows={1}
                  onChange={(e) => setAgentGoal(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Escape") {
                      e.preventDefault();
                      closeAgentInput();
                    } else if (e.key === "Enter" && !e.shiftKey) {
                      // Shift+Enter inserts a newline; plain Enter submits.
                      e.preventDefault();
                      void submitAgentGoal();
                    }
                  }}
                  onFocus={() => {
                    // Clicking into the field always reclaims keystrokes —
                    // covers both hands-free sessions (keys defaulted to the
                    // target app) and the just-stopped-voice state.
                    setQuickTooltipKeyboardMode(true).catch(() => {});
                  }}
                  onBlur={() => {
                    if (voiceActiveRef.current) {
                      restoreQuickTooltipKeyboardMode();
                    }
                  }}
                  placeholder="Tell Screenie what to do"
                  aria-label="Tell Screenie what to do"
                />
                <VoiceMicButton
                  state={voiceMicState}
                  onToggle={() => {
                    void toggleVoice();
                  }}
                />
                <button
                  type="button"
                  className="quick-tooltip-agent-submit screenie-send"
                  onClick={() => {
                    void submitAgentGoal();
                  }}
                  aria-label="Start agent task"
                  title="Start agent task"
                  disabled={!agentGoal.trim() || agentRunning || voice.busy}
                >
                  <ArrowUp size={15} strokeWidth={2} aria-hidden />
                </button>
              </div>
              {voice.active && (
                <VoiceLevelMeter fillRef={voice.meterFillRef} />
              )}
              <div className="quick-tooltip-agent-model-row">
                <div className="screenie-chat-model-select quick-tooltip-agent-model-select">
                  <CustomDropdown
                    value={providerInfo.model}
                    options={modelOptions}
                    onChange={updateModel}
                    ariaLabel={`${providerInfo.label} agent model`}
                    variant="ghost"
                    disabled={agentRunning}
                    placement="below"
                    onOpenChange={setAgentModelMenuOpen}
                    triggerLabel={
                      <span className="quick-tooltip-agent-model-label">
                        <span
                          className={`screenie-model-dot ${
                            providerInfo.cloud ? "cloud" : "local"
                          }`}
                        />
                        <span className="quick-tooltip-agent-model-provider">
                          {providerInfo.label}
                        </span>
                        <span className="quick-tooltip-agent-model-name">
                          {activeModelLabel}
                        </span>
                      </span>
                    }
                  />
                </div>
                <div className="screenie-chat-model-select quick-tooltip-agent-autonomy-select">
                  <CustomDropdown
                    value={autonomy}
                    options={AGENT_AUTONOMY_OPTIONS}
                    onChange={updateAutonomy}
                    ariaLabel="Agent autonomy"
                    variant="ghost"
                    disabled={agentRunning}
                    placement="below"
                    onOpenChange={setAgentModelMenuOpen}
                    triggerLabel={
                      <span className="quick-tooltip-agent-model-label">
                        <ShieldCheck size={11} strokeWidth={1.9} aria-hidden />
                        <span className="quick-tooltip-agent-model-name">
                          {
                            AGENT_AUTONOMY_OPTIONS.find(
                              (option) => option.value === autonomy,
                            )?.label
                          }
                        </span>
                      </span>
                    }
                  />
                </div>
              </div>
              <SvgInsetBorder radius={18} strokeAlpha={0.2} />
            </div>
            {voice.notice && (
              <div className="quick-tooltip-voice-notice">{voice.notice}</div>
            )}
            {voice.modelMissing ? (
              <VoiceModelDownloadCard
                pct={voice.downloadPct}
                model={voice.modelName}
                onDownload={() => {
                  void voice.downloadModel();
                }}
              />
            ) : (
              <VoiceTranscriptFeed
                chips={voice.chips}
                paused={voice.queuePaused}
                onSetPaused={(paused) => {
                  void voice.setQueuePaused(paused);
                }}
                onClear={() => {
                  void voice.clearQueue();
                }}
                onRemove={(id) => {
                  void voice.removeTask(id);
                }}
                onEdit={voice.editTask}
                onEditFocus={handleChipEditFocus}
                onEditBlur={handleChipEditBlur}
              />
            )}
          </div>
        </section>
      )}

      {hasStatus && (
        <section
          ref={statusCardRef}
          className="quick-tooltip-status-card screenie-toolbar"
          aria-label={
            confirmation
              ? "Agent confirmation"
              : question
                ? "Agent question"
                : error
                  ? "Screenie status"
                  : agentRunning
                    ? "Agent progress"
                    : "Saved recording"
          }
          role={error && !confirmation && !question ? "alert" : undefined}
          onMouseDown={(e) => e.stopPropagation()}
        >
          {confirmation ? (
            <div className="quick-tooltip-confirmation">
              <div className="quick-tooltip-status-heading">
                <ShieldCheck size={14} strokeWidth={1.9} aria-hidden />
                <span>Confirm action</span>
              </div>
              <div className="quick-tooltip-confirmation-action">
                {formatAgentAction(confirmation.action)}
              </div>
              <div className="quick-tooltip-confirmation-target">
                {formatAgentTarget(confirmation.target)}
              </div>
              {confirmation.action.script && (
                <pre className="quick-tooltip-confirmation-script">
                  {confirmation.action.script}
                </pre>
              )}
              <div className="quick-tooltip-confirmation-reason">
                {confirmation.reason}
              </div>
              <div className="quick-tooltip-confirmation-actions">
                <button
                  type="button"
                  className="quick-tooltip-confirm-btn"
                  onClick={() => {
                    void respondToConfirmation(false);
                  }}
                >
                  Deny
                </button>
                <button
                  type="button"
                  className="quick-tooltip-confirm-btn primary"
                  onClick={() => {
                    void respondToConfirmation(true);
                  }}
                >
                  Allow
                </button>
              </div>
            </div>
          ) : question ? (
            <div className="quick-tooltip-question">
              <div className="quick-tooltip-status-heading">
                <Bot size={14} strokeWidth={1.9} aria-hidden />
                <span>Agent question</span>
              </div>
              <div className="quick-tooltip-question-text">
                {question.question}
              </div>
              {question.options.length > 0 && (
                <div className="quick-tooltip-question-options">
                  {question.options.map((option) => (
                    <button
                      key={option}
                      type="button"
                      className="quick-tooltip-confirm-btn"
                      onClick={() => {
                        void respondToQuestion(option);
                      }}
                    >
                      {option}
                    </button>
                  ))}
                </div>
              )}
              <div className="quick-tooltip-question-answer">
                <input
                  ref={questionInputRef}
                  value={questionAnswer}
                  onChange={(e) => setQuestionAnswer(e.target.value)}
                  onKeyDown={(e) => {
                    if (e.key === "Enter") {
                      e.preventDefault();
                      void respondToQuestion(questionAnswer);
                    }
                  }}
                  placeholder="Type an answer"
                  aria-label="Answer the agent"
                />
                <button
                  type="button"
                  className="quick-tooltip-confirm-btn primary"
                  onClick={() => {
                    void respondToQuestion(questionAnswer);
                  }}
                  disabled={!questionAnswer.trim()}
                >
                  Send
                </button>
              </div>
            </div>
          ) : error ? (
            <div className="quick-tooltip-error">
              <div className="quick-tooltip-status-heading">
                <AlertTriangle size={14} strokeWidth={1.9} aria-hidden />
                <span>Error</span>
                <button
                  type="button"
                  className="quick-tooltip-status-close"
                  onClick={() => setError(null)}
                  aria-label="Dismiss error"
                  title="Dismiss error"
                >
                  <X size={13} strokeWidth={2} aria-hidden />
                </button>
              </div>
              <div className="quick-tooltip-error-text">{error}</div>
              {/screen recording/i.test(error) && (
                <div className="quick-tooltip-confirmation-actions">
                  <button
                    type="button"
                    className="quick-tooltip-confirm-btn"
                    onClick={() => {
                      invoke("open_screen_settings").catch((e) => {
                        console.error("open_screen_settings failed:", e);
                      });
                    }}
                  >
                    Open Settings
                  </button>
                  <button
                    type="button"
                    className="quick-tooltip-confirm-btn primary"
                    onClick={() => {
                      invoke("restart_app").catch((e) => {
                        console.error("restart_app failed:", e);
                      });
                    }}
                  >
                    Restart Screenie
                  </button>
                </div>
              )}
            </div>
          ) : agentRunning ? (
            <div className="quick-tooltip-agent-progress">
              <div className="quick-tooltip-status-heading">
                <Bot size={14} strokeWidth={1.9} aria-hidden />
                <span>Agent running</span>
                {recordingActive && (
                  <span className="quick-tooltip-rec-pill" title="Screen recording in progress">
                    REC
                  </span>
                )}
                <button
                  type="button"
                  className="quick-tooltip-status-close"
                  onClick={() => {
                    void stopAgentTask();
                  }}
                  aria-label="Stop agent task"
                  title="Stop agent task"
                >
                  <Square size={12} strokeWidth={2} aria-hidden />
                </button>
              </div>
              <div className="quick-tooltip-agent-progress-text">
                {agentStatus
                  ? `Step ${agentStatus.step} — ${
                      agentStatus.reason?.trim() ||
                      formatAgentAction(agentStatus.action)
                    }${
                      agentStatus.target ? ` · ${agentStatus.target}` : ""
                    }${agentStepDetail(agentStatus)}`
                  : "Starting…"}
              </div>
            </div>
          ) : savedClip ? (
            <div className="quick-tooltip-clip">
              <div className="quick-tooltip-status-heading">
                <Video size={14} strokeWidth={1.9} aria-hidden />
                <span>Recording saved</span>
                <button
                  type="button"
                  className="quick-tooltip-status-close"
                  onClick={() => setSavedClip(null)}
                  aria-label="Dismiss saved recording"
                  title="Dismiss"
                >
                  <X size={13} strokeWidth={2} aria-hidden />
                </button>
              </div>
              <div className="quick-tooltip-clip-row">
                <span className="quick-tooltip-clip-name" title={savedClip.path}>
                  {clipFileName(savedClip.path)}
                </span>
                <span className="quick-tooltip-clip-meta">
                  {Math.max(1, Math.round(savedClip.durationMs / 1000))}s
                </span>
                <button
                  type="button"
                  className="quick-tooltip-confirm-btn"
                  onClick={() => {
                    revealItemInDir(savedClip.path).catch((e) => {
                      console.error("revealItemInDir failed:", e);
                    });
                  }}
                >
                  Reveal in Finder
                </button>
              </div>
            </div>
          ) : null}
          <SvgInsetBorder radius={18} strokeAlpha={0.18} />
        </section>
      )}

      {chatVisible && (
        <section
          className="quick-tooltip-chat-panel screenie-chat-panel"
          aria-label="Quick ask chat"
        >
          <div className="quick-tooltip-chat">
            <div ref={scrollRef} className="quick-tooltip-scroll">
              {messages.length === 0 && streaming === null && (
                <div className="quick-tooltip-empty">Ask anything.</div>
              )}
              {messages.map((message, index) => (
                <TooltipMessage key={index} message={message} />
              ))}
              {streaming !== null && (
                <TooltipMessage
                  message={{ role: "assistant", content: streaming }}
                  streaming
                />
              )}
            </div>

            <div
              className="quick-tooltip-prompt screenie-chat-prompt"
              onMouseDown={(e) => e.stopPropagation()}
            >
              <textarea
                ref={taRef}
                rows={1}
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === "Enter" && !e.shiftKey) {
                    e.preventDefault();
                    sendUser(prompt);
                  }
                }}
                placeholder="Ask anything"
                aria-label="Ask anything"
              />
              <div className="quick-tooltip-prompt-actions">
                <div className="quick-tooltip-prompt-left">
                  <button
                    type="button"
                    className="quick-tooltip-mini-btn screenie-chat-flat-btn"
                    onClick={newChat}
                    aria-label="New chat"
                    title="New chat"
                    disabled={!canStartNewChat}
                  >
                    <Plus size={13} strokeWidth={1.85} aria-hidden />
                  </button>
                  <div className="screenie-chat-model-select quick-tooltip-provider-label">
                    <CustomDropdown
                      value={providerInfo.model}
                      options={modelOptions}
                      onChange={updateModel}
                      ariaLabel={`${providerInfo.label} model`}
                      variant="ghost"
                      disabled={streaming !== null}
                      triggerLabel={
                        <span className="screenie-model-label">
                          <span
                            className={`screenie-model-dot ${
                              providerInfo.cloud ? "cloud" : "local"
                            }`}
                          />
                          <span>{providerInfo.label}</span>
                        </span>
                      }
                    />
                  </div>
                </div>
                {streaming !== null ? (
                  <button
                    type="button"
                    className="quick-tooltip-send screenie-send"
                    onClick={cancelStream}
                    aria-label="Cancel response"
                    title="Cancel response"
                  >
                    <Square size={11} strokeWidth={2.1} aria-hidden />
                  </button>
                ) : (
                  <button
                    type="button"
                    className="quick-tooltip-send screenie-send"
                    onClick={() => sendUser(prompt)}
                    aria-label="Send"
                    title="Send"
                    disabled={!prompt.trim()}
                  >
                    <ArrowUp size={15} strokeWidth={2} aria-hidden />
                  </button>
                )}
              </div>
            </div>
          </div>
          <SvgInsetBorder radius={24} strokeAlpha={0.18} />
        </section>
      )}
    </div>
  );
}

function TooltipMessage({
  message,
  streaming,
}: {
  message: ChatMessage;
  streaming?: boolean;
}) {
  const formatted = formatAiMarkdown(message.content);
  const deferredFormatted = useDeferredValue(formatted);

  if (message.role === "user") {
    return <div className="quick-tooltip-user-message">{message.content}</div>;
  }

  return (
    <div className="quick-tooltip-assistant-message">
      <div className="screenie-md" data-density="compact">
        {message.content ? (
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[[rehypeKatex, SCREENIE_KATEX_OPTIONS], rehypeHighlight]}
          >
            {deferredFormatted}
          </ReactMarkdown>
        ) : (
          streaming && <span className="screenie-thinking-shimmer">Thinking...</span>
        )}
      </div>
    </div>
  );
}
