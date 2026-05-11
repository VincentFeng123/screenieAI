import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState } from "react";
import { flushSync } from "react-dom";
import { Channel, invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import {
  ArrowUp,
  Check,
  Clock,
  Copy,
  Download,
  ExternalLink,
  MessageSquare,
  ScanText,
  X,
} from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeHighlight from "rehype-highlight";
import rehypeKatex from "rehype-katex";
import "katex/dist/katex.min.css";
import "highlight.js/styles/github-dark.css";
import "./markdown.css";
import "./overlay.css";
import CustomDropdown, { type CustomDropdownOption } from "./components/CustomDropdown";
import EditAffordance from "./components/EditAffordance";
import EditCanvas from "./components/EditCanvas";
import {
  BlurredBackdrop,
  CHAT_PANEL_FROST,
  SvgInsetBorder,
} from "./components/Frosted";
import {
  ANTHROPIC_MODELS,
  GEMINI_MODELS,
  OPENAI_MODELS,
  type OllamaStatus,
  type Provider,
} from "./settings/constants";
import {
  applyStoredPreferences,
  readPreferences,
  subscribePreferences,
  type AiRenderDensityPreference,
  type ScreeniePreferences,
} from "./settings/preferences";
import {
  formatAiMarkdown,
  SCREENIE_KATEX_OPTIONS,
} from "./lib/formatAiMarkdown";
import {
  composeEditedCrop,
  strokesFingerprint,
} from "./lib/composeEditedCrop";
import { saveHistoryEntry } from "./lib/history";
import HistoryList from "./components/HistoryList";
import {
  OCR_CLIPBOARD_TEMPLATE,
  OCR_CLIPBOARD_TEMPLATE_ID,
  readTemplates,
  subscribeTemplates,
  type PromptTemplate,
} from "./lib/templates";
import {
  estimateCostCents,
  formatUsageSummary,
  recordUsage,
  type AskEvent,
  type ProviderId,
  usageTokensFromEvent,
} from "./lib/usage";
import {
  EDIT_AFFORDANCE_SIZE,
  EDIT_PILL_HORIZONTAL_LEN,
  EDIT_PILL_VERTICAL_LEN,
  type EditAnchor,
} from "./lib/editTypes";
import { useEditController, type EditController } from "./lib/useEditController";

type ScreenCapture = {
  png_base64: string;
  width: number;
  height: number;
  cursor_x?: number | null;
  cursor_y?: number | null;
  blank: boolean;
};
type CroppedCapture = { png_base64: string; width: number; height: number };

async function preloadScreenCapture(capture: ScreenCapture): Promise<void> {
  const img = new Image();
  img.src = `data:image/png;base64,${capture.png_base64}`;
  // `decode()` resolves once the bitmap is in the GPU-ready cache (not just
  // parsed for dimensions like `onload`), so the next <img> mount that
  // points at the same data URL paints on the first frame instead of
  // decoding mid-render. That's what eliminates the "panel chrome appears,
  // then the blurred backdrop snaps in 100 ms later" gap on first open.
  try {
    await img.decode();
  } catch {
    /* Invalid PNG or decoder rejected — proceed without the cache warm-up
       and let the <img> fall back to its normal async decode when it
       actually mounts. */
  }
}

type Rect = { x: number; y: number; w: number; h: number };
type Point = { x: number; y: number };
type OverlayInteractionRegion = { x: number; y: number; w: number; h: number };

function setOverlayMouseCapture(active: boolean) {
  invoke("set_overlay_mouse_capture", { active }).catch((e) => {
    console.error("set_overlay_mouse_capture failed:", e);
  });
}

function relayOverlayPointerClick(button: number) {
  invoke("relay_overlay_pointer_click", { button }).catch((e) => {
    console.error("relay_overlay_pointer_click failed:", e);
  });
}

// Track scroll-gesture phase so the macOS synthetic CGEvents the relay
// posts carry the same Began/Changed/Ended sequence a real trackpad
// gesture would. Without these, the receiving app treats each event as
// a discrete mouse-wheel tick (instant jump, no momentum), and the
// stream mixes phaseless synth ticks with phased real-passthrough
// events — the result reads as choppy scrolling. Phase values match
// kCGScrollPhase: 1=Began, 2=Changed, 4=Ended. The Rust side just
// forwards the int to native; Windows ignores it (Win32 SendInput has
// no phase concept).
const WHEEL_PHASE_BEGAN = 1;
const WHEEL_PHASE_CHANGED = 2;
const WHEEL_PHASE_ENDED = 4;
const WHEEL_IDLE_END_MS = 100;
let wheelLastEventMs = 0;
let wheelEndedTimer: number | null = null;

function relayOverlayWheel(deltaX: number, deltaY: number) {
  const now = performance.now();
  const sinceLast = now - wheelLastEventMs;
  wheelLastEventMs = now;
  if (wheelEndedTimer !== null) {
    window.clearTimeout(wheelEndedTimer);
    wheelEndedTimer = null;
  }
  // First event after an idle period starts a new gesture. JS WheelEvent
  // doesn't expose phase info directly, so we infer Began vs Changed
  // from the gap between events.
  const phase =
    sinceLast > WHEEL_IDLE_END_MS ? WHEEL_PHASE_BEGAN : WHEEL_PHASE_CHANGED;
  invoke("relay_overlay_wheel", { deltaX, deltaY, phase }).catch((e) => {
    console.error("relay_overlay_wheel failed:", e);
  });
  // Synth an Ended event after a brief idle so the receiving app
  // terminates its scroll gesture cleanly. The user's real Ended event
  // may or may not have passed through (depends on the relay/passthrough
  // alternation); the timer ensures the app always sees one.
  wheelEndedTimer = window.setTimeout(() => {
    wheelEndedTimer = null;
    invoke("relay_overlay_wheel", {
      deltaX: 0,
      deltaY: 0,
      phase: WHEEL_PHASE_ENDED,
    }).catch((e) => {
      console.error("relay_overlay_wheel (ended) failed:", e);
    });
  }, WHEEL_IDLE_END_MS);
}

type Handle =
  | "nw" | "n" | "ne"
  | "w"        | "e"
  | "sw" | "s" | "se";

type Mode =
  | { kind: "loading" }
  | { kind: "selecting" }
  | { kind: "adjusting"; rect: Rect }
  | { kind: "result"; rect: Rect; cropped: CroppedCapture; prompt: string };

const HANDLES: Handle[] = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];

const MIN_RECT = 8;
const CLICK_THRESHOLD = 5;

const OVERLAY_INTERACTION_REGION_SELECTOR = [
  "button",
  "textarea",
  "input",
  "select",
  '[contenteditable="true"]',
  '[role="button"]',
  '[role="listbox"]',
  '[role="option"]',
  '[data-screenie-hit-region="true"]',
  ".screenie-capture-region",
  ".screenie-handle-hit",
  ".screenie-move-hit",
  ".screenie-edit-canvas",
  ".screenie-edit-pill",
  ".screenie-edit-pill-wrap",
  ".screenie-edit-popover-wrap",
  ".screenie-toolbar",
  ".screenie-chat-panel",
  ".screenie-select",
  ".screenie-select-menu",
  ".screenie-select-menu-portal",
  ".screenie-action",
  ".screenie-action-group",
  ".screenie-edit-text-input",
].join(", ");

function collectOverlayInteractionRegions(): OverlayInteractionRegion[] {
  const regions: OverlayInteractionRegion[] = [];
  const viewportW = window.innerWidth;
  const viewportH = window.innerHeight;

  document
    .querySelectorAll<HTMLElement>(OVERLAY_INTERACTION_REGION_SELECTOR)
    .forEach((el) => {
      const style = window.getComputedStyle(el);
      if (style.display === "none" || style.visibility === "hidden") return;
      if (style.pointerEvents === "none") return;

      for (const clientRect of Array.from(el.getClientRects())) {
        const x1 = clamp(clientRect.left, 0, viewportW);
        const y1 = clamp(clientRect.top, 0, viewportH);
        const x2 = clamp(clientRect.right, 0, viewportW);
        const y2 = clamp(clientRect.bottom, 0, viewportH);
        const w = x2 - x1;
        const h = y2 - y1;
        if (w < 1 || h < 1) continue;
        regions.push({ x: x1, y: y1, w, h });
      }
    });

  return regions;
}

function regionsSignature(
  passthroughEnabled: boolean,
  regions: OverlayInteractionRegion[],
): string {
  return JSON.stringify({
    passthroughEnabled,
    regions: regions.map((r) => ({
      x: Math.round(r.x),
      y: Math.round(r.y),
      w: Math.round(r.w),
      h: Math.round(r.h),
    })),
  });
}

function useOverlayInteractionRegions(enabled: boolean) {
  const lastSignatureRef = useRef("");
  const enabledRef = useRef(enabled);
  const rafRef = useRef<number | null>(null);
  enabledRef.current = enabled;

  const cancelScheduledSync = useCallback(() => {
    if (rafRef.current === null) return;
    window.cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
  }, []);

  // Compute regions + send to native, signature-diff to skip redundant IPC.
  const syncNow = useCallback(() => {
    const regions = enabledRef.current ? collectOverlayInteractionRegions() : [];
    const passthroughEnabled = enabledRef.current;
    const signature = regionsSignature(passthroughEnabled, regions);
    if (signature === lastSignatureRef.current) return;
    lastSignatureRef.current = signature;
    invoke("set_overlay_interaction_regions", {
      regions,
      passthroughEnabled,
    }).catch((e) => {
      console.error("set_overlay_interaction_regions failed:", e);
    });
  }, []);

  const scheduleSync = useCallback(() => {
    cancelScheduledSync();
    rafRef.current = window.requestAnimationFrame(() => {
      rafRef.current = null;
      syncNow();
    });
  }, [cancelScheduledSync, syncNow]);

  // Sync SYNCHRONOUSLY after every render. useLayoutEffect runs after the DOM
  // commits but before paint, so getClientRects() sees the new layout and the
  // IPC fires immediately. That keeps newly visible controls from briefly
  // passing clicks through before native knows they are interactive regions.
  useLayoutEffect(() => {
    syncNow();
  });

  useEffect(() => {
    const observer = new MutationObserver(scheduleSync);
    observer.observe(document.body, {
      attributes: true,
      childList: true,
      subtree: true,
      attributeFilter: ["class", "style", "data-open", "data-visible"],
    });
    window.addEventListener("resize", scheduleSync);
    return () => {
      cancelScheduledSync();
      observer.disconnect();
      window.removeEventListener("resize", scheduleSync);
      invoke("set_overlay_interaction_regions", {
        regions: [],
        passthroughEnabled: false,
      }).catch(() => {});
      lastSignatureRef.current = "";
    };
  }, [cancelScheduledSync, scheduleSync]);
}

/* ------------------------------------------------------------------ */
/* Frost regions — drives the native NSVisualEffectView pool          */
/*                                                                     */
/* The overlay's panels (toolbar, chat, hint, edit pill, popover,      */
/* action pills, dropdown menus) are rendered with TRANSPARENT         */
/* backgrounds in CSS. Behind the WebView, native code mounts one      */
/* NSVisualEffectView per visible panel at the same viewport rect, so  */
/* the user sees a live, system-blurred frosted glass behind the       */
/* React-rendered chrome — updates in real time as windows move        */
/* behind the overlay, no SCK polling, no React renders required.     */
/*                                                                     */
/* `radius` is read from the panel's computed border-radius so the     */
/* native layer can clip the vibrancy to match the rounded panel.      */
/* ------------------------------------------------------------------ */

type OverlayFrostRegion = {
  x: number;
  y: number;
  w: number;
  h: number;
  radius: number;
};

const OVERLAY_FROST_REGION_SELECTOR = [
  ".screenie-toolbar",
  ".screenie-chat-panel",
  ".screenie-hint",
  ".screenie-edit-pill",
  ".screenie-edit-popover",
  ".screenie-action",
  ".screenie-select-menu-portal",
].join(", ");

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

function collectOverlayFrostRegions(): OverlayFrostRegion[] {
  const regions: OverlayFrostRegion[] = [];
  const viewportW = window.innerWidth;
  const viewportH = window.innerHeight;

  document
    .querySelectorAll<HTMLElement>(OVERLAY_FROST_REGION_SELECTOR)
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

function frostSignature(regions: OverlayFrostRegion[]): string {
  return JSON.stringify(
    regions.map((r) => ({
      x: Math.round(r.x),
      y: Math.round(r.y),
      w: Math.round(r.w),
      h: Math.round(r.h),
      radius: Math.round(r.radius),
    })),
  );
}

function useOverlayFrostRegions(enabled: boolean) {
  const lastSignatureRef = useRef("");
  const enabledRef = useRef(enabled);
  const rafRef = useRef<number | null>(null);
  enabledRef.current = enabled;

  const cancelScheduledSync = useCallback(() => {
    if (rafRef.current === null) return;
    window.cancelAnimationFrame(rafRef.current);
    rafRef.current = null;
  }, []);

  const syncNow = useCallback(() => {
    const regions = enabledRef.current ? collectOverlayFrostRegions() : [];
    const signature = frostSignature(regions);
    if (signature === lastSignatureRef.current) return;
    lastSignatureRef.current = signature;
    invoke("set_overlay_vibrancy_regions", { regions }).catch((e) => {
      console.error("set_overlay_vibrancy_regions failed:", e);
    });
  }, []);

  const scheduleSync = useCallback(() => {
    cancelScheduledSync();
    rafRef.current = window.requestAnimationFrame(() => {
      rafRef.current = null;
      syncNow();
    });
  }, [cancelScheduledSync, syncNow]);

  // Sync after every render so React-driven layout changes (toolbar
  // following the rect, chat panel placement, etc.) are pushed
  // immediately.
  useLayoutEffect(() => {
    syncNow();
  });

  useEffect(() => {
    // ResizeObserver catches CSS-transition-driven size changes the edit
    // pill (36 -> 280px width) does over 180ms, plus textarea auto-grow
    // and the chat panel's user-driven resize. Without this the vibrancy
    // would jump to the final size on the first React render and stay
    // stale through the animation.
    const ro = new ResizeObserver(scheduleSync);
    document
      .querySelectorAll(OVERLAY_FROST_REGION_SELECTOR)
      .forEach((el) => ro.observe(el));

    const observer = new MutationObserver((mutations) => {
      // Re-observe newly-added panels so freshly mounted toasts/popovers
      // get tracked through their CSS animations.
      for (const m of mutations) {
        m.addedNodes.forEach((n) => {
          if (!(n instanceof HTMLElement)) return;
          if (n.matches?.(OVERLAY_FROST_REGION_SELECTOR)) ro.observe(n);
          n.querySelectorAll?.(OVERLAY_FROST_REGION_SELECTOR).forEach((el) =>
            ro.observe(el),
          );
        });
      }
      scheduleSync();
    });
    observer.observe(document.body, {
      attributes: true,
      childList: true,
      subtree: true,
      attributeFilter: ["class", "style", "data-open", "data-visible"],
    });
    window.addEventListener("resize", scheduleSync);
    return () => {
      cancelScheduledSync();
      ro.disconnect();
      observer.disconnect();
      window.removeEventListener("resize", scheduleSync);
      invoke("set_overlay_vibrancy_regions", { regions: [] }).catch(() => {});
      lastSignatureRef.current = "";
    };
  }, [cancelScheduledSync, scheduleSync]);
}

const PERMISSION_BANNER_DISMISSED_KEY = "screenie.permissionBannerDismissed";

export default function Overlay() {
  const [screen, setScreen] = useState<ScreenCapture | null>(null);
  const [mode, setMode] = useState<Mode>({ kind: "loading" });
  const [preferences, setPreferences] = useState<ScreeniePreferences>(() =>
    readPreferences(),
  );
  // The capture-side `is_blank` heuristic can false-positive on dark
  // content / TCC edge cases. A user who has dismissed the banner is
  // saying "I know what I'm doing, stop nagging" — respect that for the
  // life of this install. Clearable from devtools or by removing the
  // localStorage key.
  const [permissionBannerDismissed, setPermissionBannerDismissed] = useState(
    () => {
      try {
        return localStorage.getItem(PERMISSION_BANNER_DISMISSED_KEY) === "1";
      } catch {
        return false;
      }
    },
  );
  // Edit controller is lifted to the overlay root so strokes survive the
  // adjusting → result transition (and the user's open/tool state survives an
  // explicit Esc-to-collapse).
  const editCtl = useEditController();
  const [refreshVisualHidden, setRefreshVisualHiddenState] = useState(false);
  const refreshVisualHiddenRef = useRef(false);
  const setRefreshHidden = useCallback((hidden: boolean) => {
    refreshVisualHiddenRef.current = hidden;
    setRefreshVisualHiddenState(hidden);
  }, []);

  useEffect(() => {
    applyStoredPreferences();
    return subscribePreferences((next) => {
      setPreferences(next);
      applyStoredPreferences();
    });
  }, []);

  const overlayInteractionEnabled =
    !!screen &&
    !screen.blank &&
    mode.kind !== "loading" &&
    mode.kind !== "selecting" &&
    !refreshVisualHidden;

  // Selective passthrough: only actual controls/handles disable native
  // passthrough. The captured pixels stay passthrough in adjust/result modes
  // so the user can click, hover, scroll, and drag the app underneath.
  // Selecting mode stays fully mouse-active because the user is drawing the
  // capture rect anywhere on the screen.
  useOverlayInteractionRegions(overlayInteractionEnabled);

  // Live frosted glass behind every visible panel. Native NSVisualEffectView
  // siblings of the WKWebView at each panel's viewport rect — see
  // macos_window.m's vibrancy section. Enabled whenever the overlay has
  // content to show (no need to gate on `overlayInteractionEnabled` —
  // selecting mode also wants frost behind the cursor callout / hint).
  useOverlayFrostRegions(!!screen && !screen.blank && mode.kind !== "loading" && !refreshVisualHidden);

  // Tell native code whenever a text input gains/loses focus inside the
  // overlay. While focused, the overlay's NSEvent monitor passes every
  // keystroke through (typing). When NOT focused, unhandled keystrokes
  // are forwarded to the user's previously-active app — Cmd+1 → Safari
  // switches tab, etc. — instead of dying inside our WKWebView.
  useEffect(() => {
    const isTypingTarget = (el: Element | null): boolean => {
      if (!el) return false;
      const tag = el.tagName?.toLowerCase();
      if (tag === "input") {
        const type = (el as HTMLInputElement).type?.toLowerCase() ?? "text";
        // Checkbox / radio / button-style inputs don't need keystroke
        // routing into the overlay; only the text-entry kinds do.
        return (
          type === "text" ||
          type === "search" ||
          type === "password" ||
          type === "email" ||
          type === "url" ||
          type === "number" ||
          type === "tel"
        );
      }
      if (tag === "textarea") return true;
      if ((el as HTMLElement).isContentEditable) return true;
      return false;
    };

    let lastSent: boolean | null = null;
    const sync = () => {
      const focused = isTypingTarget(document.activeElement);
      if (focused === lastSent) return;
      lastSent = focused;
      invoke("set_overlay_text_input_focused", { focused }).catch(() => {});
    };

    document.addEventListener("focusin", sync);
    document.addEventListener("focusout", sync);
    sync();
    return () => {
      document.removeEventListener("focusin", sync);
      document.removeEventListener("focusout", sync);
      invoke("set_overlay_text_input_focused", { focused: false }).catch(() => {});
    };
  }, []);

  const loadPendingCapture = useCallback(async (closeWhenEmpty = false) => {
    try {
      const cap = await invoke<ScreenCapture | null>("take_pending_capture");
      if (cap) {
        setRefreshHidden(false);
        setScreen(cap);
        // If this capture was triggered by the "repeat last" hotkey, the
        // Rust side has set repeat_pending; consume the flag and pull the
        // previously-used rect from localStorage. Drop straight into
        // adjusting mode at that rect so the user can hit Enter / tweak,
        // skipping the drag step entirely.
        let nextMode: Mode = { kind: "selecting" };
        try {
          const repeat = await invoke<boolean>("consume_repeat_pending");
          if (repeat) {
            const raw = window.localStorage.getItem("screenie.last_rect");
            if (raw) {
              const r = JSON.parse(raw) as Partial<Rect>;
              if (
                typeof r.x === "number" &&
                typeof r.y === "number" &&
                typeof r.w === "number" &&
                typeof r.h === "number"
              ) {
                const W = window.innerWidth;
                const H = window.innerHeight;
                const clamped: Rect = {
                  x: clamp(r.x, 0, Math.max(0, W - 1)),
                  y: clamp(r.y, 0, Math.max(0, H - 1)),
                  w: Math.max(MIN_RECT, Math.min(r.w, W - r.x)),
                  h: Math.max(MIN_RECT, Math.min(r.h, H - r.y)),
                };
                nextMode = { kind: "adjusting", rect: clamped };
              }
            }
          }
        } catch (e) {
          console.error("consume_repeat_pending failed:", e);
        }
        setMode(nextMode);
        // A new capture invalidates the previous strokes — drop them and
        // dismiss any pending toast.
        editCtl.clear();
        editCtl.setOpen(false);
        editCtl.setTool(null);
      } else if (closeWhenEmpty) {
        await invoke("close_overlay");
      }
    } catch (e) {
      console.error("take_pending_capture failed:", e);
      if (closeWhenEmpty) await invoke("close_overlay");
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // Pull pending capture once on mount. This is the normal first-window path.
  useEffect(() => {
    void loadPendingCapture(true);
  }, [loadPendingCapture]);

  // The frosted controls are backed by a cached screenshot. Refresh only when
  // the OS reports a real background transition (Spaces / horizontal
  // navigation), a pass-through outside click lands underneath, or the WebView
  // is actually hidden.
  const refreshInFlightRef = useRef(false);
  const refreshPendingRef = useRef(false);
  const suppressRefreshEventsUntilRef = useRef(0);
  useEffect(() => {
    let revealTimer: number | null = null;
    const clearRevealTimer = () => {
      if (revealTimer === null) return;
      window.clearTimeout(revealTimer);
      revealTimer = null;
    };
    const revealAfterRefresh = () => {
      clearRevealTimer();
      revealTimer = window.setTimeout(() => {
        revealTimer = null;
        setRefreshHidden(false);
      }, 32);
    };
    const hideForRefresh = () => {
      if (refreshVisualHiddenRef.current) return;
      flushSync(() => setRefreshHidden(true));
    };
    const shouldSuppress = () => Date.now() < suppressRefreshEventsUntilRef.current;
    const runRefresh = async (force = false) => {
      if (!force && document.visibilityState === "hidden") return;
      if (!force && shouldSuppress()) return;
      if (!refreshPendingRef.current || refreshInFlightRef.current) {
        if (!refreshPendingRef.current && !refreshInFlightRef.current) {
          setRefreshHidden(false);
        }
        return;
      }
      refreshPendingRef.current = false;
      refreshInFlightRef.current = true;
      suppressRefreshEventsUntilRef.current = Date.now() + 250;
      let refreshed = false;
      let usedLegacyRefresh = false;
      try {
        let next: ScreenCapture;
        try {
          next = await invoke<ScreenCapture>("refresh_overlay_backdrop_capture");
        } catch (sckError) {
          console.warn("refresh_overlay_backdrop_capture failed, falling back:", sckError);
          usedLegacyRefresh = true;
          hideForRefresh();
          next = await invoke<ScreenCapture>("refresh_overlay_capture");
        }
        await preloadScreenCapture(next);
        flushSync(() => setScreen(next));
        refreshed = true;
      } catch (e) {
        console.error("refresh_overlay_capture failed:", e);
      } finally {
        if (refreshed) {
          if (usedLegacyRefresh) {
            await invoke("show_overlay_after_refresh").catch((e) =>
              console.error("show_overlay_after_refresh failed:", e),
            );
          }
          revealAfterRefresh();
        } else {
          refreshPendingRef.current = true;
          // Refresh failed. If we were in the legacy fallback path, the
          // Rust side already hid the native window before capturing —
          // restore it now, otherwise the overlay would stay invisible
          // until the next user event triggers another refresh. Without
          // this, rapid Space switches that overload SCK could leave the
          // overlay stuck offscreen.
          if (usedLegacyRefresh) {
            await invoke("show_overlay_after_refresh").catch((e) =>
              console.error("show_overlay_after_refresh failed:", e),
            );
          }
          revealAfterRefresh();
        }
        window.setTimeout(() => {
          refreshInFlightRef.current = false;
          suppressRefreshEventsUntilRef.current = 0;
          // Switches that arrived DURING this refresh got their callbacks
          // dropped (inflight guard returned early). After a successful
          // refresh, run again so the bitmap catches up to the user's
          // current Space. Without this, rapid switches leave us stuck on
          // a stale bitmap until the user happens to trigger another
          // event manually. Only retry on success — re-running a failed
          // refresh would loop, since the next user event will retry
          // anyway.
          if (refreshed && refreshPendingRef.current) {
            void runRefresh(true);
          }
        }, 80);
      }
    };
    const markStale = (force = false) => {
      if (!force && shouldSuppress()) return;
      refreshPendingRef.current = true;
    };
    const requestRefresh = (force = false) => {
      markStale(force);
      void runRefresh(force);
    };
    const onVisibility = () => {
      if (document.visibilityState === "hidden") markStale();
      else void runRefresh();
    };
    const onPageShow = () => void runRefresh();
    const onGestureBackgroundChange = () => requestRefresh(true);
    const onWheel = (event: WheelEvent) => {
      if (
        Math.abs(event.deltaX) > 6 &&
        Math.abs(event.deltaX) > Math.abs(event.deltaY) * 1.35
      ) {
        requestRefresh(true);
      }
    };

    const nativeBackgroundChanged = listen("overlay-background-changed", () => {
      requestRefresh(true);
    });

    window.addEventListener("pageshow", onPageShow);
    window.addEventListener("gesturestart", onGestureBackgroundChange);
    window.addEventListener("wheel", onWheel, { capture: true, passive: true });
    document.addEventListener("visibilitychange", onVisibility);
    return () => {
      clearRevealTimer();
      window.removeEventListener("pageshow", onPageShow);
      window.removeEventListener("gesturestart", onGestureBackgroundChange);
      window.removeEventListener("wheel", onWheel, { capture: true });
      document.removeEventListener("visibilitychange", onVisibility);
      nativeBackgroundChanged.then((fn) => fn());
    };
  }, [setRefreshHidden]);

  // Re-pressing the hotkey while overlay is open emits "overlay-refresh".
  useEffect(() => {
    const ul = listen("overlay-refresh", async () => {
      await loadPendingCapture();
    });
    return () => {
      ul.then((fn) => fn());
    };
  }, [loadPendingCapture]);

  // Hidden Tauri webviews can occasionally miss a custom event while being
  // re-shown. Opportunistically check for a pending capture so the window
  // cannot remain transparent.
  useEffect(() => {
    const hydrate = () => {
      if (document.visibilityState === "hidden") return;
      void loadPendingCapture();
    };
    window.addEventListener("pageshow", hydrate);
    document.addEventListener("visibilitychange", hydrate);
    return () => {
      window.removeEventListener("pageshow", hydrate);
      document.removeEventListener("visibilitychange", hydrate);
    };
  }, [loadPendingCapture]);

  // Esc handling — context-aware. A single press collapses the edit pill
  // entirely (popover + pill + active tool, all at once); only when the
  // editor is fully idle does Esc fall through to the existing
  // `adjusting → selecting → close` ladder. Stable across renders via a
  // ref, since `editCtl` is a fresh object on every render.
  const editCtlRef = useRef(editCtl);
  editCtlRef.current = editCtl;
  useEffect(() => {
    // Shared dismiss policy: collapse the edit toolbar if it's open,
    // otherwise drop adjusting → selecting, otherwise close. Returns
    // true when something was actually handled so the keydown caller
    // can preventDefault, false when we deferred to a more local
    // handler (open dropdown, focused text-edit input).
    const dismiss = (): boolean => {
      if (document.querySelector('.screenie-select[data-open="true"]')) return false;
      const active = document.activeElement;
      if (active && active.classList.contains("screenie-edit-text-input")) {
        return false;
      }
      const ctl = editCtlRef.current;
      if (ctl.open || ctl.popoverOpen || ctl.tool !== null) {
        ctl.setPopoverOpen(false);
        ctl.setOpen(false);
        ctl.setTool(null);
        return true;
      }
      if (mode.kind === "adjusting") {
        setMode({ kind: "selecting" });
        return true;
      }
      void invoke("close_overlay");
      return true;
    };

    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      if (dismiss()) {
        e.preventDefault();
        e.stopPropagation();
        e.stopImmediatePropagation();
      }
    };

    // Native NSEvent monitor fallback. Fires whenever the overlay panel
    // is visible and the user presses Esc — even when our app is
    // inactive and the WKWebView isn't receiving keydowns (which is the
    // common case under the nonactivating-panel mask: the user has
    // Cmd+Tabbed to another app, the panel is no longer key, JS keydown
    // never reaches us). Rust forwards via emit_to("overlay", ...);
    // dismiss() then makes the same close-vs-redraw decision the JS
    // keydown handler would.
    const unlistenP = listen("overlay-escape-pressed", () => {
      dismiss();
    });

    window.addEventListener("keydown", onKey, true);
    document.addEventListener("keydown", onKey, true);
    return () => {
      unlistenP.then((fn) => fn());
      window.removeEventListener("keydown", onKey, true);
      document.removeEventListener("keydown", onKey, true);
    };
  }, [mode.kind]);

  if (!screen || mode.kind === "loading") return null;

  // The OS handed us an all-black frame — almost always Screen Recording
  // permission was likely denied. Surface a recovery banner instead of
  // the regular flow, unless the user has explicitly told us to stop
  // showing it (they may know better than the heuristic does).
  if (screen.blank && !permissionBannerDismissed) {
    return (
      <PermissionBanner
        onClose={() => invoke("close_overlay")}
        onRetry={async () => {
          // Discard the blank capture; ask the user to press the hotkey
          // again. We'd love to re-trigger from here, but capturing again
          // requires running screencapture from the AppHandle context
          // (which has the cursor/monitor info). Closing the overlay is
          // the user-visible "I'm done with this banner" — they then re-
          // press the hotkey and the new capture flows through normally.
          await invoke("close_overlay");
        }}
        onDismissPermanently={() => {
          try {
            localStorage.setItem(PERMISSION_BANNER_DISMISSED_KEY, "1");
          } catch {
            /* localStorage unavailable — dismissal won't persist */
          }
          setPermissionBannerDismissed(true);
        }}
      />
    );
  }

  const dpr = window.devicePixelRatio || 1;

  // Stable, stale-closure-proof setters via functional setMode.
  const setAdjustingRect = (r: Rect) =>
    setMode((prev) => (prev.kind === "adjusting" ? { ...prev, rect: r } : prev));
  const setResultRect = (r: Rect) =>
    setMode((prev) => (prev.kind === "result" ? { ...prev, rect: r } : prev));
  const setResultCropped = (c: CroppedCapture) =>
    setMode((prev) => (prev.kind === "result" ? { ...prev, cropped: c } : prev));

  return (
    <div
      style={{
        ...rootStyle,
        opacity: refreshVisualHidden ? 0 : 1,
        pointerEvents: refreshVisualHidden ? "none" : "auto",
        transition: refreshVisualHidden
          ? "opacity 70ms ease-out"
          : "opacity 180ms cubic-bezier(0.22, 0.61, 0.36, 1)",
        willChange: "opacity",
      }}
    >
      {mode.kind === "selecting" && (
        <SelectingLayer
          initialCursor={captureCursorPoint(screen)}
          screen={screen}
          dpr={dpr}
          onComplete={(rect) => {
            setMode({ kind: "adjusting", rect });
          }}
          onCancel={() => invoke("close_overlay")}
        />
      )}
      {mode.kind === "adjusting" && (
        <AdjustingLayer
          rect={mode.rect}
          setRect={setAdjustingRect}
          screen={screen}
          dpr={dpr}
          preferences={preferences}
          editCtl={editCtl}
          onSend={async (prompt) => {
            const r = mode.rect;
            // Remember the rect for the next "repeat last" hotkey press.
            try {
              window.localStorage.setItem("screenie.last_rect", JSON.stringify(r));
            } catch {
              /* localStorage may be unavailable */
            }
            // Fresh-capture at the moment of Send so the AI sees the screen
            // exactly as it looks right now. The cached `screen.png_base64`
            // only refreshes on Space switches / outside clicks, so without
            // this re-capture the user could send a stale crop. SCK's
            // exclude-self path keeps the overlay visible during capture so
            // there's no flicker. Edits are baked in downstream by
            // `composeEditedCrop` inside `ensureSendableImage`.
            let sourceB64 = screen.png_base64;
            try {
              const fresh = await invoke<ScreenCapture>(
                "refresh_overlay_backdrop_capture",
              );
              sourceB64 = fresh.png_base64;
              setScreen(fresh);
            } catch (e) {
              console.warn("send-time fresh capture failed, using cached:", e);
            }
            const cropped = await invoke<CroppedCapture>("crop_capture", {
              srcB64: sourceB64,
              x: Math.round(r.x * dpr),
              y: Math.round(r.y * dpr),
              w: Math.round(r.w * dpr),
              h: Math.round(r.h * dpr),
            });
            setMode({ kind: "result", rect: mode.rect, cropped, prompt });
          }}
        />
      )}
      {mode.kind === "result" && (
        <ResultLayer
          screen={screen}
          rect={mode.rect}
          setRect={setResultRect}
          cropped={mode.cropped}
          setCropped={setResultCropped}
          prompt={mode.prompt}
          dpr={dpr}
          preferences={preferences}
          editCtl={editCtl}
        />
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Shared rect-drag hook — stable handlers, no listener leak          */
/* ------------------------------------------------------------------ */

type DragSession = {
  kind: "move" | Handle;
  start: Rect;
  mouseX: number;
  mouseY: number;
  button: number;
  relayClickThrough: boolean;
  dragging: boolean;
};

type RectDragOptions = {
  relayClickThrough?: boolean;
};

function useRectDrag(
  rect: Rect,
  setRect: (r: Rect) => void,
  onEnd?: (next: Rect, start: Rect) => void,
) {
  // Refs that always reflect the latest props — let the registered-once
  // listener read fresh state without re-attaching every render.
  const rectRef = useRef(rect);
  rectRef.current = rect;
  const setRectRef = useRef(setRect);
  setRectRef.current = setRect;
  const onEndRef = useRef(onEnd);
  onEndRef.current = onEnd;

  const dragRef = useRef<DragSession | null>(null);

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      const d = dragRef.current;
      if (!d) return;
      const dx = e.clientX - d.mouseX;
      const dy = e.clientY - d.mouseY;
      if (
        !d.dragging &&
        Math.abs(dx) < CLICK_THRESHOLD &&
        Math.abs(dy) < CLICK_THRESHOLD
      ) {
        return;
      }
      d.dragging = true;
      // Keep the rect inside the screen — moves stop at edges, resizes
      // shrink the dimension that crosses one.
      const bounds = { W: window.innerWidth, H: window.innerHeight };
      setRectRef.current(applyDrag(d.start, d.kind, dx, dy, bounds));
    };
    const onUp = (e: MouseEvent) => {
      const d = dragRef.current;
      if (!d) return;
      const dx = e.clientX - d.mouseX;
      const dy = e.clientY - d.mouseY;
      const wasClick =
        !d.dragging &&
        Math.abs(dx) < CLICK_THRESHOLD &&
        Math.abs(dy) < CLICK_THRESHOLD;
      dragRef.current = null;
      setOverlayMouseCapture(false);
      if (d.relayClickThrough && wasClick) {
        relayOverlayPointerClick(d.button);
      }
      // Pass the rect at drag-start alongside the final rect so callers can
      // tell a no-op click apart from a real move/resize. Comparing `rect`
      // captured by the consumer's closure doesn't work — by the time onUp
      // fires the closure has already re-rendered with the latest rect, so
      // the two would always match.
      onEndRef.current?.(rectRef.current, d.start);
    };
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      if (dragRef.current) {
        dragRef.current = null;
        setOverlayMouseCapture(false);
      }
    };
  }, []);

  return useCallback(
    (kind: "move" | Handle, options?: RectDragOptions) => (e: React.MouseEvent) => {
      e.stopPropagation();
      e.preventDefault();
      dragRef.current = {
        kind,
        start: rectRef.current,
        mouseX: e.clientX,
        mouseY: e.clientY,
        button: e.button,
        relayClickThrough: kind === "move" && !!options?.relayClickThrough,
        dragging: false,
      };
      setOverlayMouseCapture(true);
    },
    [],
  );
}

const TOOLBAR_RADIUS = 24;
const CHAT_PANEL_RADIUS = 24;

/* ------------------------------------------------------------------ */
/* Selecting: user drags out the initial rectangle                    */
/* ------------------------------------------------------------------ */

function captureCursorPoint(screen: ScreenCapture): Point | null {
  if (typeof screen.cursor_x !== "number" || typeof screen.cursor_y !== "number") {
    return null;
  }
  return { x: screen.cursor_x, y: screen.cursor_y };
}

function SelectingLayer({
  initialCursor,
  screen,
  dpr,
  onComplete,
  onCancel,
}: {
  initialCursor: Point | null;
  screen: ScreenCapture;
  dpr: number;
  onComplete: (r: Rect) => void;
  onCancel: () => void;
}) {
  const [drag, setDrag] = useState<{ x0: number; y0: number; x1: number; y1: number } | null>(
    null,
  );
  const dragRef = useRef<typeof drag>(null);
  dragRef.current = drag;
  const onCompleteRef = useRef(onComplete);
  onCompleteRef.current = onComplete;
  const onCancelRef = useRef(onCancel);
  onCancelRef.current = onCancel;

  useEffect(() => {
    const clampX = (n: number) => clamp(n, 0, window.innerWidth);
    const clampY = (n: number) => clamp(n, 0, window.innerHeight);
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "Escape") return;
      e.preventDefault();
      e.stopPropagation();
      e.stopImmediatePropagation();
      onCancelRef.current();
    };
    const onDown = (e: MouseEvent) => {
      // Ignore mousedowns from non-primary buttons.
      if (e.button !== 0) return;
      const x = clampX(e.clientX);
      const y = clampY(e.clientY);
      setDrag({ x0: x, y0: y, x1: x, y1: y });
    };
    const onMove = (e: MouseEvent) => {
      if (!dragRef.current) return;
      setDrag({
        ...dragRef.current,
        x1: clampX(e.clientX),
        y1: clampY(e.clientY),
      });
    };
    const onUp = () => {
      const d = dragRef.current;
      if (!d) return;
      const r = normalizeRect(d);
      setDrag(null);
      // True clicks (no meaningful drag) just reset — let the user try again.
      if (r.w < CLICK_THRESHOLD && r.h < CLICK_THRESHOLD) return;
      // Floor tiny dimensions so handles are immediately reachable, but never
      // exceed the viewport.
      const W = window.innerWidth;
      const H = window.innerHeight;
      const final: Rect = {
        x: r.x,
        y: r.y,
        w: Math.min(Math.max(r.w, MIN_RECT), W - r.x),
        h: Math.min(Math.max(r.h, MIN_RECT), H - r.y),
      };
      onCompleteRef.current(final);
    };
    window.addEventListener("mousedown", onDown);
    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    window.addEventListener("keydown", onKey, true);
    document.addEventListener("keydown", onKey, true);
    return () => {
      window.removeEventListener("mousedown", onDown);
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      window.removeEventListener("keydown", onKey, true);
      document.removeEventListener("keydown", onKey, true);
    };
  }, []);

  const r = drag ? normalizeRect(drag) : null;

  return (
    <div style={{ ...fullLayer, cursor: "default" }}>
      {!r && <div style={dimStyle} />}
      {r && (
        <div
          style={{
            ...selectionRectStyle(r),
            boxShadow: `0 0 0 9999px rgba(0,0,0,0.58)`,
            pointerEvents: "none",
          }}
        />
      )}
      {drag ? (
        <Hint text="Release to lock the selection" screen={screen} dpr={dpr} />
      ) : (
        <CursorCallout
          text="Drag to select"
          initialPoint={initialCursor}
          screen={screen}
          dpr={dpr}
        />
      )}
    </div>
  );
}

/// Floating action group (Save image · OCR · Copy response) that renders on
/// the opposite end of the same edge as the pencil edit affordance. Each
/// button is its OWN 36 × 36 frosted circle — visually identical to the
/// pencil's collapsed state. The group can lay out horizontally above/below
/// the capture or vertically on the left/right, depending on which outside
/// side has clean room.
function ActionsBar({
  pencilAnchor,
  pencilOpen = false,
  avoidBoxes = [],
  rect,
  screenW,
  screenH,
  screenPngB64,
  screenCssW,
  screenCssH,
  buttons,
}: {
  pencilAnchor: EditAnchor;
  pencilOpen?: boolean;
  avoidBoxes?: ReadonlyArray<PlacementBox>;
  rect: Rect;
  screenW: number;
  screenH: number;
  screenPngB64: string;
  screenCssW: number;
  screenCssH: number;
  buttons: Array<{
    icon: React.ReactNode;
    label: string;
    onClick: () => void;
    disabled?: boolean;
  }>;
}) {
  if (pencilAnchor.hidden || buttons.length === 0) return null;
  const SIZE = EDIT_AFFORDANCE_SIZE; // 36 — same as the pencil pill collapsed
  const GAP = 4;
  const placement = placeActionsAffordance(
    pencilAnchor,
    rect,
    screenW,
    screenH,
    buttons.length,
    pencilOpen,
    avoidBoxes,
  );
  const horizontal = placement.axis === "horizontal";
  const groupW = horizontal ? SIZE * buttons.length + GAP * (buttons.length - 1) : SIZE;
  const groupH = horizontal ? SIZE : SIZE * buttons.length + GAP * (buttons.length - 1);

  const PILL_RADIUS = 9999;
  return (
    <div
      className="screenie-action-group"
      style={{
        position: "absolute",
        left: placement.x,
        top: placement.y,
        width: groupW,
        height: groupH,
        pointerEvents: "auto",
        zIndex: 12,
      }}
      onMouseDown={(e) => e.stopPropagation()}
    >
      {buttons.map((b, i) => {
        const offset = i * (SIZE + GAP);
        return (
          <button
            key={i}
            type="button"
            className="screenie-edit-pill screenie-action-btn"
            disabled={b.disabled}
            aria-label={b.label}
            onClick={(e) => {
              e.stopPropagation();
              if (!b.disabled) b.onClick();
            }}
            onMouseDown={(e) => e.stopPropagation()}
            style={{
              position: "absolute",
              left: horizontal ? offset : 0,
              top: horizontal ? 0 : offset,
              width: SIZE,
              height: SIZE,
              borderRadius: PILL_RADIUS,
              padding: 0,
              border: "none",
              color: "rgba(255, 255, 255, 0.95)",
              cursor: b.disabled ? "not-allowed" : "pointer",
              opacity: b.disabled ? 0.45 : 1,
              overflow: "hidden",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontFamily: "inherit",
            }}
          >
            <BlurredBackdrop
              src={screenPngB64}
              screenW={screenCssW}
              screenH={screenCssH}
              blurRadius={26}
              imageBrightness={0.64}
              tint="rgba(34, 36, 35, 0.43)"
              fill="rgba(18, 19, 18, 0.17)"
              persistImage
            />
            <span style={{ position: "relative", zIndex: 1, display: "flex" }}>
              {b.icon}
            </span>
            <SvgInsetBorder radius={PILL_RADIUS} />
          </button>
        );
      })}
    </div>
  );
}

type ActionBarPlacement = {
  x: number;
  y: number;
  axis: "horizontal" | "vertical";
};

/// Place the action group by checking the side opposite the prompt toolbar
/// first, then the perpendicular sides, then the toolbar side. Placements are
/// edge-aligned to the capture, not centered. Only when no outside side can
/// hold the buttons without hitting a blocker do we tuck them inside.
function placeActionsAffordance(
  pencil: EditAnchor,
  rect: Rect,
  screenW: number,
  screenH: number,
  buttonCount: number,
  pencilOpen: boolean,
  avoidBoxes: ReadonlyArray<PlacementBox>,
): ActionBarPlacement {
  const SCREEN_PAD = OVERLAY_PAD;
  const TOP = TOP_HINT_RESERVED_H;
  const SIZE = EDIT_AFFORDANCE_SIZE;
  const GAP = 4;
  const dimsFor = (axis: "horizontal" | "vertical") => ({
    w: axis === "horizontal" ? SIZE * buttonCount + GAP * (buttonCount - 1) : SIZE,
    h: axis === "horizontal" ? SIZE : SIZE * buttonCount + GAP * (buttonCount - 1),
  });
  const blockers = [...avoidBoxes, editAnchorBox(pencil, pencilOpen)];
  const hitsBlocker = (b: PlacementBox): boolean =>
    blockers.some((box) => intersectsWithGap(b, box, SCREEN_PAD));
  const onScreen = (b: PlacementBox): boolean =>
    b.x >= SCREEN_PAD &&
    b.x + b.w <= screenW - SCREEN_PAD &&
    b.y >= TOP &&
    b.y + b.h <= screenH - SCREEN_PAD;
  const onScreenLoose = (b: PlacementBox): boolean =>
    b.x >= 0 && b.x + b.w <= screenW && b.y >= 0 && b.y + b.h <= screenH;
  const edgeRoom = (edge: Edge): number => {
    if (edge === "top") return rect.y - TOP;
    if (edge === "bottom") return screenH - SCREEN_PAD - (rect.y + rect.h);
    if (edge === "left") return rect.x - SCREEN_PAD;
    return screenW - SCREEN_PAD - (rect.x + rect.w);
  };
  const toolbarBox = avoidBoxes[0] ?? null;
  const toolbarBelow = toolbarBox
    ? toolbarBox.y >= rect.y + rect.h - 1 ||
      toolbarBox.y + toolbarBox.h / 2 > rect.y + rect.h * 0.75
    : true;
  const oppositeToolbarEdge: Edge = toolbarBelow ? "top" : "bottom";
  const toolbarEdge: Edge = toolbarBelow ? "bottom" : "top";
  const perpendicularEdges = (["left", "right"] as Edge[]).sort(
    (a, b) => edgeRoom(b) - edgeRoom(a),
  );
  const edgeOrder: Edge[] = [
    oppositeToolbarEdge,
    ...perpendicularEdges,
    toolbarEdge,
  ];

  const anchorOrderFor = (edge: Edge): ("start" | "end")[] => {
    if (edge === "top" || edge === "bottom") {
      const pencilLeft = pencil.x + SIZE / 2 < rect.x + rect.w / 2;
      return pencilLeft ? ["end", "start"] : ["start", "end"];
    }
    const pencilHigh = pencil.y + SIZE / 2 < rect.y + rect.h / 2;
    return pencilHigh ? ["end", "start"] : ["start", "end"];
  };

  const outsideOnEdge = (
    edge: Edge,
    anchor: "start" | "end",
  ): ActionBarPlacement | null => {
    const axis: "horizontal" | "vertical" =
      edge === "top" || edge === "bottom" ? "horizontal" : "vertical";
    const { w, h } = dimsFor(axis);
    const x =
      edge === "left"
        ? rect.x - SCREEN_PAD - w
        : edge === "right"
          ? rect.x + rect.w + SCREEN_PAD
          : clamp(
              anchor === "start" ? rect.x : rect.x + rect.w - w,
              SCREEN_PAD,
              Math.max(SCREEN_PAD, screenW - w - SCREEN_PAD),
            );
    const y =
      edge === "top"
        ? rect.y - SCREEN_PAD - h
        : edge === "bottom"
          ? rect.y + rect.h + SCREEN_PAD
          : clamp(
              anchor === "start" ? rect.y : rect.y + rect.h - h,
              TOP,
              Math.max(TOP, screenH - h - SCREEN_PAD),
            );
    const box = { x, y, w, h };
    if (!onScreen(box) || hitsBlocker(box)) return null;
    return { x, y, axis };
  };

  for (const edge of edgeOrder) {
    for (const anchor of anchorOrderFor(edge)) {
      const placement = outsideOnEdge(edge, anchor);
      if (placement) return placement;
    }
  }

  const insideInset = 8;
  const insideCandidates: ActionBarPlacement[] = [];
  for (const edge of edgeOrder) {
    const axis: "horizontal" | "vertical" =
      edge === "left" || edge === "right" ? "vertical" : "horizontal";
    const { w, h } = dimsFor(axis);
    for (const anchor of anchorOrderFor(edge)) {
      const x =
        edge === "left"
          ? rect.x + insideInset
          : edge === "right"
            ? rect.x + rect.w - w - insideInset
            : clamp(
                anchor === "start"
                  ? rect.x + insideInset
                  : rect.x + rect.w - w - insideInset,
                rect.x + insideInset,
                rect.x + rect.w - w - insideInset,
              );
      const y =
        edge === "top"
          ? rect.y + insideInset
          : edge === "bottom"
            ? rect.y + rect.h - h - insideInset
            : clamp(
                anchor === "start"
                  ? rect.y + insideInset
                  : rect.y + rect.h - h - insideInset,
                rect.y + insideInset,
                rect.y + rect.h - h - insideInset,
              );
      insideCandidates.push({ x, y, axis });
    }
  }

  for (const placement of insideCandidates) {
    const { w, h } = dimsFor(placement.axis);
    const box = { x: placement.x, y: placement.y, w, h };
    if (onScreenLoose(box) && !hitsBlocker(box)) return placement;
  }

  const fallbackAxis: "horizontal" | "vertical" = "horizontal";
  const { w, h } = dimsFor(fallbackAxis);
  return {
    x: clamp(rect.x + rect.w - w - insideInset, 0, Math.max(0, screenW - w)),
    y: clamp(rect.y + insideInset, 0, Math.max(0, screenH - h)),
    axis: fallbackAxis,
  };
}

/// Compute the badge's clamped on-screen position given the cursor location.
function placeCursorCallout(
  clientX: number,
  clientY: number,
  cw: number,
  ch: number,
): { x: number; y: number } {
  const W = window.innerWidth;
  const H = window.innerHeight;
  const OFFSET_X = 18;
  const OFFSET_Y = 22;
  let x = clientX + OFFSET_X;
  let y = clientY + OFFSET_Y;
  if (x + cw > W - 8) x = clientX - cw - OFFSET_X;
  if (y + ch > H - 8) y = clientY - ch - OFFSET_Y;
  x = clamp(x, 8, Math.max(8, W - cw - 8));
  y = clamp(y, 8, Math.max(8, H - ch - 8));
  return { x, y };
}

/// A small floating badge that follows the cursor during the selection phase.
/// Uses React state for position so the BlurredBackdrop bitmap inside can
/// reposition itself on every move (state change → render → useLayoutEffect
/// reanchors the bitmap to the screen). Same frost recipe as the prompt
/// toolbar and the top-center Hint.
function CursorCallout({
  text,
  initialPoint,
  screen,
  dpr,
}: {
  text: string;
  initialPoint: Point | null;
  screen: ScreenCapture;
  dpr: number;
}) {
  const ref = useRef<HTMLDivElement>(null);
  const [pos, setPos] = useState<{ x: number; y: number } | null>(null);

  // First-frame snap: place the badge at the cursor position Rust handed us
  // alongside the capture, before any mousemove fires. Without this the
  // badge would briefly render center-screen on cold start.
  useLayoutEffect(() => {
    if (!initialPoint) return;
    const el = ref.current;
    if (!el) return;
    const cw = el.offsetWidth;
    const ch = el.offsetHeight;
    setPos(placeCursorCallout(initialPoint.x, initialPoint.y, cw, ch));
  }, [initialPoint]);

  useEffect(() => {
    const onMove = (e: MouseEvent) => {
      const el = ref.current;
      if (!el) return;
      const cw = el.offsetWidth;
      const ch = el.offsetHeight;
      setPos(placeCursorCallout(e.clientX, e.clientY, cw, ch));
    };
    window.addEventListener("mousemove", onMove);
    return () => window.removeEventListener("mousemove", onMove);
  }, []);

  const HINT_RADIUS = 999;
  const positioned = pos !== null;

  return (
    <div
      ref={ref}
      className="screenie-fadein"
      style={{
        position: "absolute",
        left: positioned ? pos.x : "50%",
        top: positioned ? pos.y : "50%",
        transform: positioned ? "none" : "translate(-50%, -50%)",
        pointerEvents: "none",
      }}
    >
      <div
        className="screenie-hint"
        style={{
          position: "relative",
          padding: "8px 14px",
          borderRadius: HINT_RADIUS,
          fontSize: 12.5,
          fontWeight: 500,
          letterSpacing: 0.2,
          whiteSpace: "nowrap",
          color: "rgba(255,255,255,0.95)",
          userSelect: "none",
          overflow: "hidden",
        }}
      >
        <BlurredBackdrop
          src={screen.png_base64}
          screenW={screen.width / dpr}
          screenH={screen.height / dpr}
          blurRadius={26}
          imageBrightness={0.64}
          tint="rgba(34, 36, 35, 0.43)"
          fill="rgba(18, 19, 18, 0.17)"
          persistImage
        />
        <span style={{ position: "relative", zIndex: 1 }}>{text}</span>
        <SvgInsetBorder radius={HINT_RADIUS} />
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Adjusting: handles + drag-to-move + toolbar                        */
/* ------------------------------------------------------------------ */

function AdjustingLayer({
  rect,
  setRect,
  screen,
  dpr,
  preferences,
  editCtl,
  onSend,
}: {
  rect: Rect;
  setRect: (r: Rect) => void;
  screen: ScreenCapture;
  dpr: number;
  preferences: ScreeniePreferences;
  editCtl: EditController;
  onSend: (prompt: string) => void;
}) {
  // The cropped image dimensions for editing in this phase = scaled rect,
  // since we haven't actually called crop_capture yet. Use device-pixel
  // dimensions so strokes survive the scale into the final cropped PNG.
  const editDims = {
    width: Math.max(1, Math.round(rect.w * dpr)),
    height: Math.max(1, Math.round(rect.h * dpr)),
  };

  // When the user drags rect handles after annotating, translate the strokes
  // through screenshot-space so they stay anchored to the underlying pixels.
  // ResultLayer does the same against its `cropped` image; here the "crop" is
  // simply the current rect.
  const onRectDragEnd = useCallback(
    (next: Rect, start: Rect) => {
      if (!editCtl.hasStrokes) return;
      if (
        next.x === start.x &&
        next.y === start.y &&
        next.w === start.w &&
        next.h === start.h
      ) {
        return;
      }
      const newDims = {
        width: Math.max(1, Math.round(next.w * dpr)),
        height: Math.max(1, Math.round(next.h * dpr)),
      };
      const oldDims = {
        width: Math.max(1, Math.round(start.w * dpr)),
        height: Math.max(1, Math.round(start.h * dpr)),
      };
      editCtl.remapForCrop(start, next, oldDims, newDims);
    },
    [editCtl, dpr],
  );

  const beginDrag = useRectDrag(rect, setRect, onRectDragEnd);

  // Toolbar bbox + edit anchor, recomputed from rect on every render. The
  // editor lives only in the foreground here (no chat panel during adjusting).
  const W = window.innerWidth;
  const H = window.innerHeight;
  const toolbarBox = computeToolbarBbox(
    rect,
    W,
    H,
    preferences.overlayShowPresets,
    TOP_HINT_RESERVED_H,
  );
  const editAnchor = placeEditAffordance(rect, W, H, toolbarBox, null, null);
  const pillBox =
    editCtl.open && !editAnchor.hidden ? editAnchorBox(editAnchor, true) : null;
  const hiddenHandles = useObscuredHandles(rect, pillBox);

  const [adjustingToast, setAdjustingToast] = useState<{
    text: string;
    key: number;
  } | null>(null);
  useEffect(() => {
    if (!adjustingToast) return;
    const t = setTimeout(() => setAdjustingToast(null), 1800);
    return () => clearTimeout(t);
  }, [adjustingToast]);

  /// Run OCR-to-clipboard for the current rect using the on-device Vision
  /// framework. Crops the screen first (so we only OCR the user-selected
  /// region, not the whole capture), bakes any annotations in, then pipes
  /// the bytes through the native `ocr_image_local` command. Fully local,
  /// no AI tokens consumed.
  const ocrToClipboardAdjusting = async () => {
    try {
      setAdjustingToast({ text: "Extracting text…", key: Date.now() });
      const cropped = await invoke<CroppedCapture>("crop_capture", {
        srcB64: screen.png_base64,
        x: Math.round(rect.x * dpr),
        y: Math.round(rect.y * dpr),
        w: Math.round(rect.w * dpr),
        h: Math.round(rect.h * dpr),
      });
      let pngB64 = cropped.png_base64;
      if (editCtl.hasStrokes) {
        pngB64 = await composeEditedCrop(
          cropped.png_base64,
          { width: cropped.width, height: cropped.height },
          editCtl.strokes,
        );
      }
      const txt = (await invoke<string>("ocr_image_local", { pngB64 })).trim();
      if (!txt) {
        setAdjustingToast({ text: "No text found", key: Date.now() });
        return;
      }
      await navigator.clipboard.writeText(txt).catch(() => {});
      const preview = txt.length > 48 ? txt.slice(0, 45) + "…" : txt;
      setAdjustingToast({ text: `Copied: ${preview}`, key: Date.now() });
    } catch (e) {
      console.error("OCR-to-clipboard (adjusting) failed:", e);
      const msg = typeof e === "string" ? e : (e as Error).message ?? String(e);
      setAdjustingToast({ text: `OCR failed: ${msg}`, key: Date.now() });
    }
  };

  /// Save the current rect + annotations to disk without going through the
  /// AI flow. Crops via the existing Rust command, composites strokes if
  /// any, then writes via save_annotated_image.
  const saveAdjustingImage = async () => {
    try {
      const cropped = await invoke<CroppedCapture>("crop_capture", {
        srcB64: screen.png_base64,
        x: Math.round(rect.x * dpr),
        y: Math.round(rect.y * dpr),
        w: Math.round(rect.w * dpr),
        h: Math.round(rect.h * dpr),
      });
      let pngB64 = cropped.png_base64;
      if (editCtl.hasStrokes) {
        pngB64 = await composeEditedCrop(
          cropped.png_base64,
          { width: cropped.width, height: cropped.height },
          editCtl.strokes,
        );
      }
      const path = await invoke<string>("save_annotated_image", { pngB64 });
      const tail = path.split("/").pop() ?? "Screenie.png";
      setAdjustingToast({ text: `Saved · ${tail}`, key: Date.now() });
    } catch (e) {
      console.error("save image (adjusting) failed:", e);
      setAdjustingToast({ text: "Save failed", key: Date.now() });
    }
  };

  return (
    <div style={fullLayer}>
      <div
        className="screenie-capture-region"
        style={{
          ...selectionRectStyle(rect),
          boxShadow: `0 0 0 9999px rgba(0,0,0,0.58)`,
          cursor: "default",
          pointerEvents: editCtl.tool ? "none" : "auto",
        }}
        onMouseDown={beginDrag("move", { relayClickThrough: true })}
        onWheel={(e) => {
          if (editCtl.tool) return;
          e.preventDefault();
          e.stopPropagation();
          relayOverlayWheel(e.deltaX, e.deltaY);
        }}
      />
      {moveHitAreas(rect).map((style, i) => (
        <div
          key={`move-${i}`}
          className="screenie-move-hit"
          onMouseDown={beginDrag("move")}
          style={{ ...style, pointerEvents: editCtl.tool ? "none" : "auto" }}
        />
      ))}
      {HANDLES.filter((h) => !hiddenHandles.has(h)).map((h) => (
        <div key={`v-${h}`} className="screenie-handle" style={handleStyle(rect, h)} />
      ))}
      {HANDLES.map((h) => (
        <div
          key={`hit-${h}`}
          className="screenie-handle-hit"
          onMouseDown={beginDrag(h)}
          style={{
            ...handleHitArea(rect, h),
            pointerEvents: editCtl.tool ? "none" : "auto",
          }}
        />
      ))}
      <EditCanvas
        ctl={editCtl}
        rect={rect}
        cropped={editDims}
        active={editCtl.tool !== null}
        colorPickerSource={{
          b64: screen.png_base64,
          offsetX: Math.round(rect.x * dpr),
          offsetY: Math.round(rect.y * dpr),
        }}
      />
      <EditAffordance
        ctl={editCtl}
        anchor={editAnchor}
        screenPngB64={screen.png_base64}
        screenW={screen.width / dpr}
        screenH={screen.height / dpr}
        avoidBoxes={[toolbarBox]}
      />
      <ActionsBar
        pencilAnchor={editAnchor}
        pencilOpen={editCtl.open}
        avoidBoxes={[toolbarBox]}
        rect={rect}
        screenW={W}
        screenH={H}
        screenPngB64={screen.png_base64}
        screenCssW={screen.width / dpr}
        screenCssH={screen.height / dpr}
        buttons={[
          {
            icon: <Download size={15} strokeWidth={1.85} aria-hidden />,
            label: "Save image",
            onClick: () => {
              void saveAdjustingImage();
            },
          },
          {
            icon: <ScanText size={15} strokeWidth={1.85} aria-hidden />,
            label: "OCR → Clipboard",
            onClick: () => {
              void ocrToClipboardAdjusting();
            },
          },
        ]}
      />
      <Toolbar
        rect={rect}
        screen={screen}
        dpr={dpr}
        onSend={onSend}
        disabled={false}
        allowEmpty={preferences.overlayAllowEmptySend}
        showPresets={preferences.overlayShowPresets}
        topReserve={TOP_HINT_RESERVED_H}
      />
      {editCtl.trimmedNotice && (
        <TrimmedNoticeToast
          count={editCtl.trimmedNotice.count}
          onDismiss={editCtl.dismissTrimmedNotice}
          screen={screen}
          dpr={dpr}
        />
      )}
      {editCtl.pickedColor && (
        <PickedColorToast
          hex={editCtl.pickedColor.hex}
          onDismiss={() => editCtl.setPickedColor(null)}
          screen={screen}
          dpr={dpr}
        />
      )}
      {adjustingToast && (
        <StatusToast
          text={adjustingToast.text}
          onDismiss={() => setAdjustingToast(null)}
          screen={screen}
          dpr={dpr}
        />
      )}
      <Hint
        text="Drag the box · Resize via handles · Enter to send · Esc to redraw · Esc Esc to close"
        screen={screen}
        dpr={dpr}
      />
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Toolbar: prompt + presets — disabled while streaming               */
/* ------------------------------------------------------------------ */

/// Subscribe to user-defined templates. The OCR-to-clipboard preset has
/// been moved out of the chip row into the floating actions bar (it's an
/// icon there, next to Save image), so the chips show only user-configured
/// templates now.
function useTemplates(): PromptTemplate[] {
  const [list, setList] = useState<PromptTemplate[]>(() => readTemplates());
  useEffect(() => subscribeTemplates(setList), []);
  return list;
}

function PresetChipRow({
  disabled,
  onPick,
}: {
  disabled: boolean;
  onPick: (t: PromptTemplate) => void;
}) {
  const templates = useTemplates();
  if (templates.length === 0) return null;
  return (
    <div style={{ display: "flex", gap: 4, flexWrap: "wrap" }}>
      {templates.map((t) => (
        <button
          key={t.id}
          className="screenie-chip"
          disabled={disabled}
          onClick={() => onPick(t)}
        >
          {t.label}
        </button>
      ))}
    </div>
  );
}

const OVERLAY_PAD = 14;
const TOP_HINT_RESERVED_H = 76;
const TOOLBAR_W = 480;
const TOOLBAR_MAX_INPUT_H = 120;

// Toolbar geometry — derived from the actual layout so the "does it fit
// above / below the rect?" decision matches the rendered height. Hard-coding
// this constant used to undershoot when the presets row was visible, which
// let the toolbar pop above the rect into the top hint zone.
const TOOLBAR_PADDING = 8;
const TOOLBAR_INNER_GAP = 6;
const TOOLBAR_PRESETS_ROW_H = 26;
// Sum of the input shell's top + bottom padding (see the shell JSX). Asymmetric
// (less on top) so the textarea text reads as vertically balanced against the
// taller send button when at single-line minimum.
const TOOLBAR_INPUT_SHELL_PAD = 10;

function toolbarReservedHeight(showPresets: boolean): number {
  const presets = showPresets ? TOOLBAR_PRESETS_ROW_H + TOOLBAR_INNER_GAP : 0;
  return (
    TOOLBAR_PADDING * 2 +
    presets +
    TOOLBAR_INPUT_SHELL_PAD +
    TOOLBAR_MAX_INPUT_H
  );
}

/// Where the toolbar will render given the rect + screen + preset-row state.
/// Both the Toolbar (for actual layout) and ResultLayer (for chat-panel
/// placement) call this so they agree on the toolbar's natural footprint
/// without one having to "see" the other's render.
function computeToolbarBbox(
  rect: Rect,
  screenW: number,
  screenH: number,
  showPresets: boolean,
  topReserve: number,
): Rect {
  const PAD = OVERLAY_PAD;
  const maxH = toolbarReservedHeight(showPresets);
  const topLimit = Math.max(PAD, topReserve);

  const fitsBelow = rect.y + rect.h + PAD + maxH <= screenH - PAD;
  const fitsAbove = rect.y - PAD - maxH >= topLimit;

  let left = rect.x + rect.w / 2 - TOOLBAR_W / 2;
  left = clamp(left, PAD, Math.max(PAD, screenW - TOOLBAR_W - PAD));

  const top = fitsBelow
    ? rect.y + rect.h + PAD
    : fitsAbove
      ? rect.y - PAD - maxH
      : screenH - PAD - maxH;

  return { x: left, y: top, w: TOOLBAR_W, h: maxH };
}

function Toolbar({
  rect,
  screen,
  dpr,
  onSend,
  onTemplate,
  disabled,
  streaming = false,
  onStop,
  avoidPanel,
  allowEmpty = false,
  showPresets = true,
  topReserve = OVERLAY_PAD,
}: {
  rect: Rect;
  screen: ScreenCapture;
  dpr: number;
  onSend: (prompt: string) => void;
  /// Optional template-aware send. When set, preset chips call this with
  /// the selected template instead of `onSend`. The parent routes special
  /// templates (e.g. OCR → clipboard) without going through the chat path.
  onTemplate?: (template: PromptTemplate) => void;
  disabled: boolean;
  streaming?: boolean;
  onStop?: () => void;
  avoidPanel?: Rect | null;
  allowEmpty?: boolean;
  showPresets?: boolean;
  topReserve?: number;
}) {
  const [prompt, setPrompt] = useState("");
  const taRef = useRef<HTMLTextAreaElement>(null);

  const MIN_INPUT_H = 22;
  const MAX_INPUT_H = TOOLBAR_MAX_INPUT_H;
  const adjustHeight = () => {
    const ta = taRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = Math.min(Math.max(ta.scrollHeight, MIN_INPUT_H), MAX_INPUT_H) + "px";
  };
  useEffect(adjustHeight, [prompt]);

  const PAD = OVERLAY_PAD;
  // Worst-case height — chips row + maxed textarea + paddings. Used only for
  // the "fits below / above?" decision; actual placement uses CSS top OR
  // bottom anchors so the toolbar grows the right direction as the textarea
  // expands. Computed (instead of hard-coded) so toggling the presets row
  // doesn't leave a stale reserve that places the toolbar over the top hint.
  const TOOLBAR_MAX_H = toolbarReservedHeight(showPresets);
  const screenW = window.innerWidth;
  const screenH = window.innerHeight;
  const topLimit = Math.max(PAD, topReserve);

  // Decide vertical placement.
  const fitsBelow = rect.y + rect.h + PAD + TOOLBAR_MAX_H <= screenH - PAD;
  const fitsAbove = rect.y - PAD - TOOLBAR_MAX_H >= topLimit;

  // Horizontal placement — center on the rect, clamped to the viewport.
  let baseLeft = rect.x + rect.w / 2 - TOOLBAR_W / 2;
  baseLeft = clamp(baseLeft, PAD, Math.max(PAD, screenW - TOOLBAR_W - PAD));

  const positionStyle: React.CSSProperties = fitsBelow
    ? { top: rect.y + rect.h + PAD }
    : fitsAbove
    ? // Anchoring via `bottom` keeps the toolbar's bottom edge PAD px from
      // the rect's top edge — so it sits right next to the rect instead of
      // floating far above as the textarea grows.
      { bottom: screenH - rect.y + PAD }
    : // Neither above nor below fits (full-screen rect). Stick to the bottom
      // of the screen, inside the captured region.
      { bottom: PAD };

  // Compute a shift-out displacement when the natural position overlaps the
  // chat panel. This is applied via a CSS transform on an inner wrapper —
  // so rect-following stays snappy (outer top/left have no transition) and
  // only the avoidance shift animates.
  let shiftX = 0;
  if (avoidPanel) {
    // Conservative bbox: assume max toolbar height for vertical extent.
    const tlx = baseLeft;
    const trx = baseLeft + TOOLBAR_W;
    const tty = fitsBelow
      ? rect.y + rect.h + PAD
      : fitsAbove
      ? rect.y - PAD - TOOLBAR_MAX_H
      : screenH - PAD - TOOLBAR_MAX_H;
    const tby = tty + TOOLBAR_MAX_H;

    const plx = avoidPanel.x;
    const prx = avoidPanel.x + avoidPanel.w;
    const ply = avoidPanel.y;
    const pry = avoidPanel.y + avoidPanel.h;

    const overlap = !(
      trx + PAD <= plx ||
      tlx >= prx + PAD ||
      tby + PAD <= ply ||
      tty >= pry + PAD
    );
    if (overlap) {
      const targetLeftSide = plx - PAD - TOOLBAR_W;
      const targetRightSide = prx + PAD;
      if (targetLeftSide >= PAD) {
        shiftX = targetLeftSide - baseLeft;
      } else if (targetRightSide + TOOLBAR_W <= screenW - PAD) {
        shiftX = targetRightSide - baseLeft;
      }
    }
  }

  const submit = (text: string) => {
    if (disabled) return;
    if (!allowEmpty && !text.trim()) return;
    onSend(text);
    setPrompt("");
  };

  // Animate the toolbar's avoidance shift only when the chat panel is
  // closed/reopened. We can't use a CSS transition because every other shiftX
  // change (rect resize) needs to be instant — keeping a transition on the
  // transform would make those frames lag. Instead, when `isAvoiding` flips
  // null↔panel, kick off a one-shot Web Animations API tween from the prior
  // shift to the new one. WAAPI overrides the inline transform for the
  // duration of the animation, then hands back to whatever the inline value
  // is when it ends — so subsequent rect-resize updates remain snappy.
  const isAvoiding = !!avoidPanel;
  const innerToolbarRef = useRef<HTMLDivElement>(null);
  const prevAvoidRef = useRef(isAvoiding);
  const prevShiftRef = useRef(shiftX);
  useLayoutEffect(() => {
    if (prevAvoidRef.current !== isAvoiding) {
      prevAvoidRef.current = isAvoiding;
      const el = innerToolbarRef.current;
      if (el && prevShiftRef.current !== shiftX) {
        el.animate(
          [
            { transform: `translateX(${prevShiftRef.current}px)` },
            { transform: `translateX(${shiftX}px)` },
          ],
          {
            duration: 260,
            easing: "cubic-bezier(0.22, 0.61, 0.36, 1)",
            fill: "none",
          },
        );
      }
    }
    prevShiftRef.current = shiftX;
  }, [isAvoiding, shiftX]);

  return (
    <div
      className="screenie-toolbar-spawn"
      style={{
        position: "absolute",
        ...positionStyle,
        left: baseLeft,
        width: TOOLBAR_W,
        // Outer is purely a positioning anchor; the toolbar inside this
        // wrapper handles all events. Outer doesn't transition — so when the
        // rect moves, the toolbar's natural position follows instantly.
        pointerEvents: "none",
      }}
    >
    {/* Avoidance-shift wrapper. Holds the WAAPI ref + the inline transform
        for the chat-panel-avoidance shift. Kept *separate* from the frosted
        surface because WebKit drops the backdrop blur whenever its host
        element is also a transform target — having a non-frosted wrapper
        here keeps the frost stable across re-shifts. */}
    <div
      ref={innerToolbarRef}
      style={{
        width: TOOLBAR_W,
        // Snap the avoidance shift instantly. A transition here smooths the
        // ON/OFF moment but lags every intermediate frame during a rect
        // resize, making the gap to the chat panel visibly "rubber-band".
        transform: `translateX(${shiftX}px)`,
      }}
    >
    <div
      onMouseDown={(e) => e.stopPropagation()}
      className="screenie-toolbar"
      style={{
        width: TOOLBAR_W,
        boxSizing: "border-box",
        borderRadius: TOOLBAR_RADIUS,
        display: "flex",
        flexDirection: "column",
        padding: TOOLBAR_PADDING,
        gap: TOOLBAR_INNER_GAP,
        pointerEvents: "auto",
        overflow: "hidden",
      }}
    >
      <BlurredBackdrop
        src={screen.png_base64}
        screenW={screen.width / dpr}
        screenH={screen.height / dpr}
        blurRadius={26}
        imageBrightness={0.64}
        tint="rgba(34, 36, 35, 0.43)"
        fill="rgba(18, 19, 18, 0.17)"
        persistImage
      />
      {showPresets && <PresetChipRow disabled={disabled} onPick={(t) => {
        if (onTemplate) onTemplate(t);
        else submit(t.prompt);
      }} />}
      <div
        className="screenie-input-shell"
        style={{
          display: "flex",
          alignItems: "flex-end",
          gap: 6,
          // Asymmetric padding-top vs padding-bottom: the textarea is shorter
          // than the send button at single-line, and `align-items: flex-end`
          // bottom-aligns both, leaving extra empty space above the textarea.
          // 4px top + 6px bottom compensates so the text caret reads centered.
          padding: "4px 6px 6px 12px",
          minHeight: 32,
        }}
      >
        <textarea
          ref={taRef}
          rows={1}
          value={prompt}
          className="screenie-textarea"
          onChange={(e) => setPrompt(e.target.value)}
          onKeyDown={(e) => {
            if (e.key === "Enter" && !e.shiftKey) {
              e.preventDefault();
              submit(prompt);
            }
          }}
          placeholder={
            disabled
              ? "Streaming response…"
              : allowEmpty
                ? "Ask about this region, or send blank…"
                : "Ask about this region…"
          }
          style={{
            flex: 1,
            position: "relative",
            zIndex: 1,
            background: "transparent",
            border: "none",
            outline: "none",
            color: "#ffffff",
            fontSize: 14,
            fontFamily: "inherit",
            letterSpacing: 0.1,
            resize: "none",
            padding: "5px 0",
            margin: 0,
            lineHeight: 1.45,
            minHeight: MIN_INPUT_H,
            maxHeight: MAX_INPUT_H,
            overflow: "auto",
          }}
        />
        <button
          className="screenie-send"
          // While streaming the button is the stop control — never disabled.
          // Otherwise it follows the usual "needs prompt unless allowEmpty"
          // gating.
          disabled={!streaming && (disabled || (!allowEmpty && !prompt.trim()))}
          onClick={() => {
            if (streaming) {
              onStop?.();
              return;
            }
            submit(prompt);
          }}
          style={{ marginBottom: 2, position: "relative", zIndex: 1 }}
          aria-label={streaming ? "Stop generating" : "Send"}
        >
          {streaming ? (
            <span
              aria-hidden
              style={{
                display: "inline-block",
                width: 10,
                height: 10,
                background: "currentColor",
                borderRadius: 1.5,
              }}
            />
          ) : (
            <ArrowUp size={15} strokeWidth={2} aria-hidden />
          )}
        </button>
      </div>
      <SvgInsetBorder radius={TOOLBAR_RADIUS} />
    </div>
    </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Result: editable rect + chat panel                                  */
/* ------------------------------------------------------------------ */

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
  usage?: {
    inputTokens: number;
    outputTokens: number;
    provider: string;
    model: string;
    costCents: number;
  };
};

type ProviderInfo = {
  provider: Provider;
  cloud: boolean;
  label: string;
  model: string;
};

const PROVIDER_LABELS: Record<Provider, { label: string; cloud: boolean; defaultModel: string }> = {
  anthropic: { label: "Claude", cloud: true, defaultModel: ANTHROPIC_MODELS[0].id },
  openai: { label: "OpenAI", cloud: true, defaultModel: OPENAI_MODELS[0].id },
  gemini: { label: "Gemini", cloud: true, defaultModel: GEMINI_MODELS[0].id },
  ollama: { label: "Ollama", cloud: false, defaultModel: "llama3.2-vision" },
};

const MODEL_STORAGE_KEYS: Record<Provider, string> = {
  anthropic: "anthropic_model",
  openai: "openai_model",
  gemini: "gemini_model",
  ollama: "ollama_model",
};

function modelStorageKey(provider: Provider): string {
  return MODEL_STORAGE_KEYS[provider];
}

function modelOptionsForProvider(
  provider: Provider,
  currentModel: string,
  ollamaModels: string[],
): CustomDropdownOption[] {
  const withCurrent = (options: CustomDropdownOption[]) => {
    if (!currentModel || options.some((option) => option.value === currentModel)) {
      return options;
    }
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

  const options = ollamaModels.map((model) => ({ value: model, label: model }));
  return withCurrent(
    options.length > 0
      ? options
      : [{ value: currentModel || "llama3.2-vision", label: currentModel || "llama3.2-vision" }],
  );
}

function compactModelLabel(label: string): string {
  return label.split(" — ")[0];
}

type ChatPanelSlot = "right" | "left" | "overlap";

type ChatPanelPlacement = Rect & {
  slot: ChatPanelSlot;
  overlapsCapture: boolean;
};

type FloatingPanelSession =
  | { kind: "move"; start: Rect; mouseX: number; mouseY: number }
  | { kind: "resize"; edge: Handle; start: Rect; mouseX: number; mouseY: number };

const FLOATING_PANEL_DRAG_BLOCK_SELECTOR = [
  "button",
  "textarea",
  "input",
  "select",
  "a",
  '[contenteditable="true"]',
  '[role="button"]',
  '[role="listbox"]',
  '[role="option"]',
  ".screenie-select",
  ".screenie-select-menu",
  ".screenie-history-row",
].join(", ");

const CHAT_PANEL_W = 400;
const CHAT_PANEL_MIN_W = 320;
const CHAT_PANEL_H = 540;
const CHAT_PANEL_MIN_H = 240;
const OVERLAP_PANEL_MIN_W = 260;
const OVERLAP_PANEL_MIN_H = 220;
const OVERLAP_PANEL_RESIZE_ZONE = 10;

/// Compute a non-overlapping chat-panel rect to the side of the capture.
/// Side-only by design: above/below placement was removed because it forced
/// the capture into a thin horizontal slot whenever the chat opened. The
/// width compresses with available room down to `CHAT_PANEL_MIN_W`; anything
/// tighter than that on both sides returns `null`, and the caller collapses
/// the chat to the "Show chat" button.
///
/// `toolbarBox` (when supplied) is the toolbar's natural footprint. We
/// prefer the panel side that doesn't overlap it — this lets the toolbar
/// stay centered on the rect rather than getting pushed to the opposite
/// side of the captured region (which read as "the textfield is on the
/// wrong side of my screenshot"). When neither side avoids the toolbar, we
/// fall back to the original right-then-left preference and accept the
/// visual overlap; the toolbar no longer shifts to avoid it.
function computeSafeChatPanel(
  rect: Rect,
  screenW: number,
  screenH: number,
  toolbarBox: Rect | null,
): ChatPanelPlacement | null {
  const gap = OVERLAY_PAD;
  const viewportW = screenW - 2 * gap;
  const viewportH = screenH - 2 * gap;
  if (viewportW < CHAT_PANEL_MIN_W || viewportH < CHAT_PANEL_MIN_H) {
    return null;
  }

  const sideH = Math.min(CHAT_PANEL_H, viewportH);
  const sideY = clamp(rect.y, gap, Math.max(gap, screenH - sideH - gap));
  const rightRoom = screenW - (rect.x + rect.w) - 2 * gap;
  const leftRoom = rect.x - 2 * gap;

  const rightSlot: ChatPanelPlacement | null =
    rightRoom >= CHAT_PANEL_MIN_W
      ? {
          slot: "right",
          overlapsCapture: false,
          x: rect.x + rect.w + gap,
          y: sideY,
          w: Math.min(CHAT_PANEL_W, rightRoom),
          h: sideH,
        }
      : null;

  const leftW = leftRoom >= CHAT_PANEL_MIN_W ? Math.min(CHAT_PANEL_W, leftRoom) : 0;
  const leftSlot: ChatPanelPlacement | null =
    leftRoom >= CHAT_PANEL_MIN_W
      ? {
          slot: "left",
          overlapsCapture: false,
          x: rect.x - leftW - gap,
          y: sideY,
          w: leftW,
          h: sideH,
        }
      : null;

  if (!toolbarBox) {
    return rightSlot ?? leftSlot;
  }

  const overlapsToolbar = (panel: ChatPanelPlacement): boolean =>
    !(
      panel.x + panel.w + gap <= toolbarBox.x ||
      panel.x >= toolbarBox.x + toolbarBox.w + gap ||
      panel.y + panel.h + gap <= toolbarBox.y ||
      panel.y >= toolbarBox.y + toolbarBox.h + gap
    );

  // When a side-by-side slot would overlap the toolbar, keep the panel's
  // y anchored to the rect (don't float it up or down — the user has
  // explicitly said vertical shifts feel wrong) and just shrink the
  // panel's height so its bottom edge sits above the toolbar's top.
  // SHRUNK_MIN_H is intentionally smaller than CHAT_PANEL_MIN_H — when
  // shrinking is the only way to avoid colliding with the toolbar, a
  // shorter-than-usual chat is preferable to either overlapping or
  // detaching from the rect.
  const SHRUNK_MIN_H = 180;

  const shrinkToAvoid = (
    panel: ChatPanelPlacement,
  ): ChatPanelPlacement | null => {
    if (!overlapsToolbar(panel)) return panel;
    if (panel.y >= toolbarBox.y) return null; // toolbar above us; can't shrink without shifting
    const newH = toolbarBox.y - gap - panel.y;
    if (newH < SHRUNK_MIN_H) return null;
    return { ...panel, h: newH };
  };

  if (rightSlot && !overlapsToolbar(rightSlot)) return rightSlot;
  if (leftSlot && !overlapsToolbar(leftSlot)) return leftSlot;

  const rightShrunk = rightSlot ? shrinkToAvoid(rightSlot) : null;
  if (rightShrunk) return rightShrunk;
  const leftShrunk = leftSlot ? shrinkToAvoid(leftSlot) : null;
  if (leftShrunk) return leftShrunk;

  return rightSlot ?? leftSlot;
}

function computeOverlapChatPanel(rect: Rect, screenW: number, screenH: number): ChatPanelPlacement {
  const gap = OVERLAY_PAD;
  const w = Math.min(CHAT_PANEL_W, screenW - 2 * gap);
  const h = Math.min(420, screenH - 2 * gap);
  const rightRaw = screenW - (rect.x + rect.w);
  const leftRaw = rect.x;
  return {
    slot: "overlap",
    overlapsCapture: true,
    x: rightRaw >= leftRaw ? Math.max(gap, screenW - w - gap) : gap,
    y: clamp(rect.y, gap, Math.max(gap, screenH - h - gap)),
    w,
    h,
  };
}

function clampFloatingPanel(rect: Rect, screenW: number, screenH: number): Rect {
  const gap = OVERLAY_PAD;
  const maxW = Math.max(OVERLAP_PANEL_MIN_W, screenW - 2 * gap);
  const maxH = Math.max(OVERLAP_PANEL_MIN_H, screenH - 2 * gap);
  const w = clamp(rect.w, OVERLAP_PANEL_MIN_W, maxW);
  const h = clamp(rect.h, OVERLAP_PANEL_MIN_H, maxH);
  return {
    x: clamp(rect.x, gap, Math.max(gap, screenW - w - gap)),
    y: clamp(rect.y, gap, Math.max(gap, screenH - h - gap)),
    w,
    h,
  };
}

function panelResizeEdge(
  clientX: number,
  clientY: number,
  element: HTMLElement,
): Handle | null {
  const box = element.getBoundingClientRect();
  const nearLeft = clientX - box.left <= OVERLAP_PANEL_RESIZE_ZONE;
  const nearRight = box.right - clientX <= OVERLAP_PANEL_RESIZE_ZONE;
  const nearTop = clientY - box.top <= OVERLAP_PANEL_RESIZE_ZONE;
  const nearBottom = box.bottom - clientY <= OVERLAP_PANEL_RESIZE_ZONE;

  if (nearTop && nearLeft) return "nw";
  if (nearTop && nearRight) return "ne";
  if (nearBottom && nearLeft) return "sw";
  if (nearBottom && nearRight) return "se";
  if (nearTop) return "n";
  if (nearBottom) return "s";
  if (nearLeft) return "w";
  if (nearRight) return "e";
  return null;
}

function panelCursor(edge: Handle | null): string {
  if (!edge) return "default";
  if (edge === "n" || edge === "s") return "ns-resize";
  if (edge === "e" || edge === "w") return "ew-resize";
  if (edge === "nw" || edge === "se") return "nwse-resize";
  return "nesw-resize";
}

function resizeFloatingPanel(
  start: Rect,
  edge: Handle,
  dx: number,
  dy: number,
  screenW: number,
  screenH: number,
): Rect {
  let { x, y, w, h } = start;
  if (edge.includes("e")) {
    w += dx;
  }
  if (edge.includes("s")) {
    h += dy;
  }
  if (edge.includes("w")) {
    x += dx;
    w -= dx;
  }
  if (edge.includes("n")) {
    y += dy;
    h -= dy;
  }

  const gap = OVERLAY_PAD;
  const maxW = Math.max(OVERLAP_PANEL_MIN_W, screenW - 2 * gap);
  const maxH = Math.max(OVERLAP_PANEL_MIN_H, screenH - 2 * gap);
  const nextW = clamp(w, OVERLAP_PANEL_MIN_W, maxW);
  const nextH = clamp(h, OVERLAP_PANEL_MIN_H, maxH);

  if (edge.includes("w")) x = start.x + start.w - nextW;
  if (edge.includes("n")) y = start.y + start.h - nextH;

  return clampFloatingPanel({ x, y, w: nextW, h: nextH }, screenW, screenH);
}

function placeShowChatAction(
  rect: Rect,
  screenW: number,
  screenH: number,
  actionW: number,
  actionH: number,
): { x: number; y: number } {
  const gap = 8;
  const rightInset = 2;
  const topRightX = clamp(
    rect.x + rect.w - actionW - rightInset,
    gap,
    Math.max(gap, screenW - actionW - gap),
  );

  const aboveY = rect.y - actionH - gap;
  if (aboveY >= gap) {
    return { x: topRightX, y: aboveY };
  }

  if (rect.w >= actionW + rightInset + gap && rect.h >= actionH + gap * 2) {
    return {
      x: rect.x + rect.w - actionW - rightInset,
      y: rect.y + gap,
    };
  }

  return {
    x: topRightX,
    y: clamp(rect.y + gap, gap, Math.max(gap, screenH - actionH - gap)),
  };
}

type PlacementBox = { x: number; y: number; w: number; h: number };

type CornerKey = "corner-tl" | "corner-tr" | "corner-bl" | "corner-br";
type Edge = "top" | "bottom" | "left" | "right";

function intersectsWithGap(a: PlacementBox, b: PlacementBox, gap: number): boolean {
  return !(
    a.x + a.w + gap <= b.x ||
    a.x >= b.x + b.w + gap ||
    a.y + a.h + gap <= b.y ||
    a.y >= b.y + b.h + gap
  );
}

function editAnchorBox(anchor: EditAnchor, open: boolean): PlacementBox {
  const size = EDIT_AFFORDANCE_SIZE;
  if (!open) return { x: anchor.x, y: anchor.y, w: size, h: size };
  if (anchor.axis === "horizontal") {
    const len = EDIT_PILL_HORIZONTAL_LEN;
    return {
      x: anchor.expandToward === "start" ? anchor.x + size - len : anchor.x,
      y: anchor.y,
      w: len,
      h: size,
    };
  }
  const len = EDIT_PILL_VERTICAL_LEN;
  return {
    x: anchor.x,
    y: anchor.expandToward === "start" ? anchor.y + size - len : anchor.y,
    w: size,
    h: len,
  };
}

/// Decide where the edit affordance lands. Outside placements inspect all
/// four edge gutters and pick the cleanest side with room for the expanded
/// pill; only if no outside side works do we fall back inside the capture.
/// So an icon can be:
///   - in the top gutter, anchored to the rect's top-left or top-right corner
///     (pill is horizontal, grows along the top edge)
///   - in the bottom gutter (horizontal, along the bottom edge)
///   - in the left gutter, anchored to top-left or bottom-left (pill is
///     vertical, grows along the left edge)
///   - in the right gutter (vertical, along the right edge)
/// If the best anchor would clip off-screen, the pill slides along that edge
/// rather than rejecting a valid side.
function placeEditAffordance(
  rect: Rect,
  screenW: number,
  screenH: number,
  toolbarBox: Rect,
  panel: PlacementBox | null,
  showChatPill: PlacementBox | null,
  size: number = EDIT_AFFORDANCE_SIZE,
  expandedH: number = EDIT_PILL_HORIZONTAL_LEN,
  expandedV: number = EDIT_PILL_VERTICAL_LEN,
): EditAnchor {
  const PAD = OVERLAY_PAD;
  const TOP = TOP_HINT_RESERVED_H;
  if (rect.w < 48 || rect.h < 48) {
    return {
      side: "corner-tr",
      x: rect.x + Math.max(0, rect.w - size) - 4,
      y: rect.y + 4,
      axis: "horizontal",
      expandToward: "start",
      hidden: true,
    };
  }

  const toolbarBelow =
    toolbarBox.y >= rect.y + rect.h - 1 ||
    toolbarBox.y + toolbarBox.h / 2 > rect.y + rect.h * 0.75;

  const blockers: PlacementBox[] = [toolbarBox];
  if (panel) blockers.push(panel);
  if (showChatPill) blockers.push(showChatPill);

  const intersects = (b: PlacementBox): boolean =>
    blockers.some(
      (box) =>
        !(
          b.x + b.w + PAD <= box.x ||
          b.x >= box.x + box.w + PAD ||
          b.y + b.h + PAD <= box.y ||
          b.y >= box.y + box.h + PAD
        ),
    );

  const onScreen = (b: PlacementBox): boolean =>
    b.x >= PAD && b.x + b.w <= screenW - PAD && b.y >= TOP && b.y + b.h <= screenH - PAD;

  // Distinct from the cropped-image rect's own border: stay clear of the
  // 8 white resize handles on the corner so the icon doesn't sit on top of one.
  const insideInset = 8;

  /// Try a specific edge-anchor placement and return null if it doesn't fit
  /// or collides with a blocker.
  const tryOutsideEdge = (
    edge: Edge,
    anchor: "start" | "end",
  ): EditAnchor | null => {
    let x: number;
    let y: number;
    let axis: "horizontal" | "vertical";
    let expandToward: "start" | "end";
    let corner: CornerKey;

    if (edge === "top") {
      // Pill in top gutter. anchor "start" = left end, "end" = right end.
      axis = "horizontal";
      y = rect.y - PAD - size;
      x = anchor === "start" ? rect.x : rect.x + rect.w - size;
      expandToward = "start";
      corner = anchor === "start" ? "corner-tl" : "corner-tr";
    } else if (edge === "bottom") {
      axis = "horizontal";
      y = rect.y + rect.h + PAD;
      x = anchor === "start" ? rect.x : rect.x + rect.w - size;
      expandToward = "start";
      corner = anchor === "start" ? "corner-bl" : "corner-br";
    } else if (edge === "left") {
      axis = "vertical";
      x = rect.x - PAD - size;
      y = anchor === "start" ? rect.y : rect.y + rect.h - size;
      expandToward = "start";
      corner = anchor === "start" ? "corner-tl" : "corner-bl";
    } else {
      axis = "vertical";
      x = rect.x + rect.w + PAD;
      y = anchor === "start" ? rect.y : rect.y + rect.h - size;
      expandToward = "start";
      corner = anchor === "start" ? "corner-tr" : "corner-br";
    }

    // Verify the expanded pill fits on-screen and doesn't collide. If the
    // edge has room but the exact rect-corner anchor would clip off-screen,
    // slide the pill along that edge instead of rejecting the side outright.
    const len = axis === "horizontal" ? expandedH : expandedV;
    let pillX: number;
    let pillY: number;
    let pillW: number;
    let pillH: number;
    if (axis === "horizontal") {
      pillW = len;
      pillH = size;
      pillY = y;
      pillX = x + size - len;
      const clampedPillX = clamp(
        pillX,
        PAD,
        Math.max(PAD, screenW - len - PAD),
      );
      if (clampedPillX !== pillX) {
        pillX = clampedPillX;
        x = pillX + len - size;
      }
    } else {
      pillW = size;
      pillH = len;
      pillX = x;
      pillY = y + size - len;
      const clampedPillY = clamp(
        pillY,
        TOP,
        Math.max(TOP, screenH - len - PAD),
      );
      if (clampedPillY !== pillY) {
        pillY = clampedPillY;
        y = pillY + len - size;
      }
    }
    const iconBox: PlacementBox = { x, y, w: size, h: size };
    const pillBox: PlacementBox = { x: pillX, y: pillY, w: pillW, h: pillH };
    if (!onScreen(iconBox)) return null;
    if (intersects(iconBox)) return null;
    if (!onScreen(pillBox)) return null;
    if (intersects(pillBox)) return null;

    return { side: corner, x, y, axis, expandToward };
  };

  /// Looser on-screen check for inside placements: the rect's own edges
  /// (not the global PAD gutter) are what bound the icon, so we only
  /// require the icon to be on the visible viewport at all. Without this
  /// the inside fallback would refuse to place anything when the rect
  /// hugs a screen edge — the case the user hit with a full-screen
  /// selection.
  const onScreenLoose = (b: PlacementBox): boolean =>
    b.x >= 0 &&
    b.x + b.w <= screenW &&
    b.y >= 0 &&
    b.y + b.h <= screenH;

  /// Inside corner fallback when no outside edge has room. Clamps to keep
  /// the icon inside the rect's bounds even if the rect itself extends
  /// past the viewport (shouldn't happen, but cheap belt-and-braces).
  const tryInsideCorner = (corner: CornerKey): EditAnchor | null => {
    const top = corner === "corner-tl" || corner === "corner-tr";
    const left = corner === "corner-tl" || corner === "corner-bl";
    const x = left ? rect.x + insideInset : rect.x + rect.w - size - insideInset;
    const y = top ? rect.y + insideInset : rect.y + rect.h - size - insideInset;
    const iconBox: PlacementBox = { x, y, w: size, h: size };
    if (!onScreenLoose(iconBox)) return null;
    if (intersects(iconBox)) return null;
    const expandToward: "start" | "end" = "start";
    const len = expandedH;
    const pillX = x + size - len;
    const pillBox: PlacementBox = { x: pillX, y, w: len, h: size };
    // Pill must also stay on-screen; allow it to extend up to the screen
    // edges (no PAD gutter required for inside placements).
    if (!onScreenLoose(pillBox)) {
      // Try to clamp the pill horizontally so it stays on-screen — keeps
      // the icon at its corner while the body slides toward the rect's
      // interior.
      const clampedX = Math.max(0, Math.min(pillBox.x, screenW - len));
      pillBox.x = clampedX;
      if (!onScreenLoose(pillBox)) return null;
    }
    if (intersects(pillBox)) return null;
    return { side: corner, x, y, axis: "horizontal", expandToward };
  };

  /// Place the pencil immediately to the LEFT of the show-chat pill on the
  /// same y, so the two affordances cluster as a row at one end of the
  /// rect's outside edge instead of splitting to opposite corners. Without
  /// this, the standard "end" anchor would collide with the chat pill (the
  /// regular `intersects` check enforces a PAD=14 gap), and the pencil
  /// would fall back to the "start" anchor at the opposite edge of the
  /// rect — which reads as the pencil "going to the middle / wrong end".
  ///
  /// Excludes `showChatPill` from the blocker check (we're snugging up to
  /// it on purpose) but keeps the toolbar + chat panel as blockers. The
  /// expanded pill grows leftward (away from the chat pill), so when
  /// expanded it never overlaps. Returns null when there's no horizontal
  /// room — the regular candidate loop then picks something else.
  const tryAdjacentToShowChat = (): EditAnchor | null => {
    if (!showChatPill) return null;
    const ADJ_GAP = 4;

    const x = showChatPill.x - ADJ_GAP - size;
    const y = showChatPill.y;
    const iconBox: PlacementBox = { x, y, w: size, h: size };

    // Loose on-screen check covers both above-rect and inside-top-right
    // placements of the show-chat pill (matching `tryInsideCorner`).
    if (!onScreenLoose(iconBox)) return null;

    const otherBlockers: PlacementBox[] = [toolbarBox];
    if (panel) otherBlockers.push(panel);
    const hitsOther = (b: PlacementBox): boolean =>
      otherBlockers.some(
        (box) =>
          !(
            b.x + b.w + PAD <= box.x ||
            b.x >= box.x + box.w + PAD ||
            b.y + b.h + PAD <= box.y ||
            b.y >= box.y + box.h + PAD
          ),
      );
    if (hitsOther(iconBox)) return null;

    // Expanded pill grows leftward from the icon. If the natural left edge
    // would clip off-screen, reject — clamping right would slide the pill
    // back over the chat pill.
    const expBox: PlacementBox = {
      x: x + size - expandedH,
      y,
      w: expandedH,
      h: size,
    };
    if (expBox.x < 0) return null;
    if (hitsOther(expBox)) return null;

    const pillCenterY = showChatPill.y + showChatPill.h / 2;
    const corner: CornerKey =
      pillCenterY < rect.y + rect.h / 2 ? "corner-tr" : "corner-br";
    return { side: corner, x, y, axis: "horizontal", expandToward: "start" };
  };

  // Same side priority as the companion action buttons: check the side
  // opposite the text field first, then the perpendicular gutters, then the
  // text-field side. Expansion direction is fixed by axis: top/bottom always
  // grow left, left/right always grow up.
  type Cand = { kind: "outside"; edge: Edge; anchor: "start" | "end" }
    | { kind: "inside"; corner: CornerKey };
  const candidates: Cand[] = [];

  // Expansion is always left/up, so prefer right/bottom anchors first; the
  // secondary anchor only exists for cramped cases where the clamp can slide
  // the pill along the chosen edge.
  const horizontalAnchors: ("start" | "end")[] = ["end", "start"];
  const verticalAnchors: ("start" | "end")[] = ["end", "start"];

  const edgeRoom = (edge: Edge): number => {
    if (edge === "top") return rect.y - TOP;
    if (edge === "bottom") return screenH - PAD - (rect.y + rect.h);
    if (edge === "left") return rect.x - PAD;
    return screenW - PAD - (rect.x + rect.w);
  };
  const oppositeToolbarEdge: Edge = toolbarBelow ? "top" : "bottom";
  const toolbarEdge: Edge = toolbarBelow ? "bottom" : "top";
  const perpendicularEdges = (["left", "right"] as Edge[]).sort(
    (a, b) => edgeRoom(b) - edgeRoom(a),
  );
  const outsideEdges: Edge[] = [
    oppositeToolbarEdge,
    ...perpendicularEdges,
    toolbarEdge,
  ];
  for (const edge of outsideEdges) {
    const anchors = edge === "top" || edge === "bottom"
      ? horizontalAnchors
      : verticalAnchors;
    for (const anchor of anchors) {
      candidates.push({ kind: "outside", edge, anchor });
    }
  }

  const insideCornersTop: CornerKey[] = ["corner-tr", "corner-tl"];
  const insideCornersBot: CornerKey[] = ["corner-br", "corner-bl"];
  const insideOrder = toolbarBelow
    ? [...insideCornersTop, ...insideCornersBot]
    : [...insideCornersBot, ...insideCornersTop];
  for (const c of insideOrder) candidates.push({ kind: "inside", corner: c });

  // Try adjacent-to-show-chat first. When the chat panel is hidden the
  // show-chat pill is rendered at the rect's top-right corner; standard
  // edge anchors would otherwise collide with it (PAD=14 gap) and the
  // pencil would jump to the opposite corner. Snugging the pencil up to
  // the chat pill keeps both affordances on the same row, reading as a
  // single cluster of controls.
  if (showChatPill) {
    const adjacent = tryAdjacentToShowChat();
    if (adjacent) return adjacent;
  }

  for (const c of candidates) {
    const a = c.kind === "outside"
      ? tryOutsideEdge(c.edge, c.anchor)
      : tryInsideCorner(c.corner);
    if (a) return a;
  }

  // Last resort: hide.
  return {
    side: "corner-tr",
    x: rect.x + rect.w - size - 4,
    y: rect.y + 4,
    axis: "horizontal",
    expandToward: "start",
    hidden: true,
  };
}

/// Small toast that floats at the top-center, like `Hint`, but auto-dismisses
/// itself after 1.8 s via the editCtl. Used to surface "N strokes were
/// trimmed when the rect changed".
function TrimmedNoticeToast({
  count,
  onDismiss,
  screen,
  dpr,
}: {
  count: number;
  onDismiss: () => void;
  screen: ScreenCapture;
  dpr: number;
}) {
  return (
    <ToastPill onDismiss={onDismiss} screen={screen} dpr={dpr}>
      {count === 1
        ? "1 mark was trimmed to fit the new crop"
        : `${count} marks were trimmed to fit the new crop`}
    </ToastPill>
  );
}

function PickedColorToast({
  hex,
  onDismiss,
  screen,
  dpr,
}: {
  hex: string;
  onDismiss: () => void;
  screen: ScreenCapture;
  dpr: number;
}) {
  return (
    <ToastPill onDismiss={onDismiss} screen={screen} dpr={dpr}>
      <span
        aria-hidden
        style={{
          display: "inline-block",
          width: 12,
          height: 12,
          borderRadius: 999,
          background: hex,
          border: "1px solid rgba(255,255,255,0.6)",
          marginRight: 8,
          verticalAlign: "middle",
        }}
      />
      Copied <code style={{ fontFamily: "ui-monospace, monospace" }}>{hex}</code>
    </ToastPill>
  );
}

function StatusToast({
  text,
  onDismiss,
  screen,
  dpr,
}: {
  text: string;
  onDismiss: () => void;
  screen: ScreenCapture;
  dpr: number;
}) {
  return (
    <ToastPill onDismiss={onDismiss} screen={screen} dpr={dpr}>
      {text}
    </ToastPill>
  );
}

/// Same frost recipe as the top-center Hint — BlurredBackdrop bitmap +
/// SvgInsetBorder — so toasts read as part of the same UI family. Sits
/// just below the Hint (which floats around top: 28 and ends ~58) with
/// an 8 px breathing margin so the two never visually stack.
function ToastPill({
  children,
  onDismiss,
  screen,
  dpr,
  topOffset = 66,
}: {
  children: React.ReactNode;
  onDismiss: () => void;
  screen: ScreenCapture;
  dpr: number;
  topOffset?: number;
}) {
  const PILL_RADIUS = 999;
  return (
    <div
      data-screenie-hit-region="true"
      onMouseDown={(e) => e.stopPropagation()}
      onClick={onDismiss}
      style={{
        position: "absolute",
        left: "50%",
        top: topOffset,
        transform: "translateX(-50%)",
        cursor: "pointer",
        userSelect: "none",
        zIndex: 20,
      }}
    >
      <div
        className="screenie-hint"
        style={{
          position: "relative",
          padding: "8px 14px",
          borderRadius: PILL_RADIUS,
          fontSize: 12.5,
          fontWeight: 500,
          letterSpacing: 0.1,
          color: "rgba(255,255,255,0.95)",
          whiteSpace: "nowrap",
          overflow: "hidden",
        }}
      >
        <BlurredBackdrop
          src={screen.png_base64}
          screenW={screen.width / dpr}
          screenH={screen.height / dpr}
          blurRadius={26}
          imageBrightness={0.64}
          tint="rgba(34, 36, 35, 0.43)"
          fill="rgba(18, 19, 18, 0.17)"
          persistImage
        />
        <span style={{ position: "relative", zIndex: 1 }}>{children}</span>
        <SvgInsetBorder radius={PILL_RADIUS} />
      </div>
    </div>
  );
}

/// Take Rust's `AiError` (serialized via Display, sometimes with a giant
/// JSON or HTML body) and produce a short, user-actionable message. The raw
/// detail is preserved as a tooltip via the chat panel's title attribute.
function formatAiError(e: unknown, info: ProviderInfo): string {
  const raw = typeof e === "string" ? e : (e as Error)?.message ?? String(e);
  // Common shapes from Rust:
  //   "api: 401 — { ... json ... }"
  //   "api: 429 — <!DOCTYPE html> ..."
  //   "http: error decoding response body: ..."
  //   "api key missing"
  //   "<provider> returned no text chunks"
  const apiMatch = raw.match(/^api:\s*(\d{3})\s*—\s*([\s\S]*)$/);
  if (apiMatch) {
    const status = parseInt(apiMatch[1], 10);
    const body = apiMatch[2];
    let detail = "";
    // Try to extract `error.message` from JSON body; fall back to first line.
    try {
      const parsed = JSON.parse(body);
      detail =
        parsed?.error?.message ??
        parsed?.message ??
        parsed?.error?.toString?.() ??
        "";
    } catch {
      const firstLine = body.split(/[\r\n]+/).find((l) => l.trim().length) ?? "";
      detail = firstLine.replace(/<[^>]+>/g, "").trim();
    }
    if (detail.length > 240) detail = detail.slice(0, 237) + "…";
    if (status === 401 || status === 403) {
      return `${info.label} rejected the API key (HTTP ${status}). Check it in Settings → Providers.`;
    }
    if (status === 429) {
      return `${info.label} is rate-limited (HTTP 429). Wait a moment, or switch provider.`;
    }
    if (status >= 500) {
      return `${info.label} returned HTTP ${status}. Try again shortly. ${detail}`.trim();
    }
    return `${info.label} HTTP ${status}: ${detail || "request failed"}`;
  }
  if (/api key missing/i.test(raw)) {
    return `${info.label} API key is missing. Add one in Settings → Providers.`;
  }
  if (/^keyring:/i.test(raw)) {
    return `Couldn't read the ${info.label} key from your system keychain. Open Settings → Providers, then retry.`;
  }
  if (/request too large/i.test(raw)) {
    return `That capture or chat history is too large to send. Crop a smaller region or start a new chat.`;
  }
  if (/invalid provider/i.test(raw)) {
    return `The selected provider is invalid. Open Settings → Providers and choose a provider again.`;
  }
  if (/no text chunks/i.test(raw)) {
    return `${info.label} returned an empty response. Try again, or pick a different model.`;
  }
  if (/ollama not reachable/i.test(raw)) {
    return `Ollama isn't reachable on localhost:11434. Open Ollama.app, then retry.`;
  }
  if (raw.length > 240) return raw.slice(0, 237) + "…";
  return raw;
}

function readProviderInfo(): ProviderInfo {
  const savedProvider = localStorage.getItem("provider");
  const p: Provider =
    savedProvider && savedProvider in PROVIDER_LABELS
      ? (savedProvider as Provider)
      : "anthropic";
  const meta = PROVIDER_LABELS[p];
  let model = meta.defaultModel;
  if (p === "ollama") model = localStorage.getItem("ollama_model") || meta.defaultModel;
  if (p === "openai") model = storedCloudModel("openai_model", OPENAI_MODELS, meta.defaultModel);
  if (p === "gemini") model = storedCloudModel("gemini_model", GEMINI_MODELS, meta.defaultModel);
  if (p === "anthropic") {
    model = storedCloudModel("anthropic_model", ANTHROPIC_MODELS, meta.defaultModel);
  }
  return { provider: p, cloud: meta.cloud, label: meta.label, model };
}

function storedCloudModel(
  key: string,
  options: Array<{ id: string }>,
  fallback: string,
): string {
  const saved = localStorage.getItem(key);
  return saved && options.some((option) => option.id === saved) ? saved : fallback;
}

function ResultLayer({
  screen,
  rect,
  setRect,
  cropped,
  setCropped,
  prompt,
  dpr,
  preferences,
  editCtl,
}: {
  screen: ScreenCapture;
  rect: Rect;
  setRect: (r: Rect) => void;
  cropped: CroppedCapture;
  setCropped: (c: CroppedCapture) => void;
  prompt: string;
  dpr: number;
  preferences: ScreeniePreferences;
  editCtl: EditController;
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([
    { role: "user", content: prompt || "Describe what's shown in this image." },
  ]);
  const [streaming, setStreaming] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [providerInfo, setProviderInfo] = useState<ProviderInfo>(() => readProviderInfo());
  const [ollamaModels, setOllamaModels] = useState<string[]>([]);
  const [overlapPanelRect, setOverlapPanelRect] = useState<Rect | null>(null);
  const [overlapPanelCursor, setOverlapPanelCursor] = useState("default");
  const [historyPreviewB64, setHistoryPreviewB64] = useState<string | null>(null);
  const overlapPanelSessionRef = useRef<FloatingPanelSession | null>(null);
  const scrollRef = useRef<HTMLDivElement>(null);

  // Keep providerInfo fresh if the user changes the provider in the main
  // window while the overlay is open.
  useEffect(() => {
    const onStorage = () => setProviderInfo(readProviderInfo());
    window.addEventListener("storage", onStorage);
    return () => window.removeEventListener("storage", onStorage);
  }, []);

  useEffect(() => {
    if (providerInfo.provider !== "ollama") {
      setOllamaModels([]);
      return;
    }

    let cancelled = false;
    invoke<OllamaStatus>("check_ollama")
      .then((status) => {
        if (!cancelled) setOllamaModels(status.models);
      })
      .catch(() => {
        if (!cancelled) setOllamaModels([]);
      });

    return () => {
      cancelled = true;
    };
  }, [providerInfo.provider]);

  useEffect(() => {
    const el = scrollRef.current;
    if (el) el.scrollTop = el.scrollHeight;
  }, [messages, streaming]);

  useEffect(() => {
    const onMove = (event: MouseEvent) => {
      const session = overlapPanelSessionRef.current;
      if (!session) return;
      event.preventDefault();
      const dx = event.clientX - session.mouseX;
      const dy = event.clientY - session.mouseY;
      const screenW = window.innerWidth;
      const screenH = window.innerHeight;

      if (session.kind === "move") {
        setOverlapPanelRect(
          clampFloatingPanel(
            {
              ...session.start,
              x: session.start.x + dx,
              y: session.start.y + dy,
            },
            screenW,
            screenH,
          ),
        );
      } else {
        setOverlapPanelRect(resizeFloatingPanel(session.start, session.edge, dx, dy, screenW, screenH));
      }
    };
    const onUp = () => {
      if (!overlapPanelSessionRef.current) return;
      overlapPanelSessionRef.current = null;
      setOverlayMouseCapture(false);
      setOverlapPanelCursor("default");
    };

    window.addEventListener("mousemove", onMove);
    window.addEventListener("mouseup", onUp);
    return () => {
      window.removeEventListener("mousemove", onMove);
      window.removeEventListener("mouseup", onUp);
      if (overlapPanelSessionRef.current) {
        overlapPanelSessionRef.current = null;
        setOverlayMouseCapture(false);
      }
    };
  }, []);

  // Memoized composite cache: when the user has drawn strokes we bake them
  // into the cropped PNG before sending to ask_ai. Keyed on the cropped b64
  // plus a stroke-content fingerprint so multi-turn follow-ups don't recompose.
  const composeCacheRef = useRef<{ key: string; result: string } | null>(null);
  const ensureSendableImage = useCallback(
    async (override?: CroppedCapture): Promise<string> => {
      const target = override ?? cropped;
      if (!editCtl.hasStrokes) return target.png_base64;
      // Cache key: use the b64 string length plus its tail (the body bytes vary
      // a lot more than the PNG header), combined with a stroke fingerprint.
      // A slice prefix would collide between captures with the same header.
      const tail = target.png_base64.slice(-32);
      const key = `${target.png_base64.length}:${tail}:${strokesFingerprint(editCtl.strokes)}`;
      const hit = composeCacheRef.current;
      if (hit && hit.key === key) return hit.result;
      const composed = await composeEditedCrop(
        target.png_base64,
        { width: target.width, height: target.height },
        editCtl.strokes,
      );
      composeCacheRef.current = { key, result: composed };
      return composed;
    },
    [editCtl, cropped],
  );

  // At every send-to-AI moment, re-capture the screen behind the rect so
  // the bytes the model sees match what's on screen RIGHT NOW — not the
  // snapshot from when the user opened the overlay. SCK's exclude-self
  // path keeps the overlay visible (no flicker, no permission prompt).
  // Falls back to the cached crop on any failure so the message still
  // goes through. Updates `cropped` so the chat thumbnail also reflects
  // what was actually sent.
  const captureFreshSendCrop = useCallback(async (): Promise<CroppedCapture> => {
    try {
      const fresh = await invoke<ScreenCapture>("refresh_overlay_backdrop_capture");
      const c = await invoke<CroppedCapture>("crop_capture", {
        srcB64: fresh.png_base64,
        x: Math.round(rect.x * dpr),
        y: Math.round(rect.y * dpr),
        w: Math.round(rect.w * dpr),
        h: Math.round(rect.h * dpr),
      });
      setCropped(c);
      return c;
    } catch (e) {
      console.warn("send-time fresh capture failed, using cached crop:", e);
      return cropped;
    }
  }, [rect, dpr, cropped, setCropped]);

  // Save the captured image + prompt + response to history exactly once per
  // ResultLayer lifetime (= once per capture session). Multi-turn follow-ups
  // don't spam new entries; the saved snapshot is the first-turn payload.
  const historySavedRef = useRef(false);

  // Chat panel mode: "chat" (default) shows the conversation; "history"
  // swaps in a list of past captures the user can click to re-open.
  const [chatView, setChatView] = useState<"chat" | "history">("chat");

  // Toggle between the user's last cloud provider and Ollama. Cmd+L
  // triggers the same flip from the keyboard.
  const flipProvider = useCallback(() => {
    const current = readProviderInfo();
    let next: Provider;
    if (current.provider === "ollama") {
      const last = (localStorage.getItem("provider_last_cloud") as Provider | null) ?? "anthropic";
      next = last;
    } else {
      localStorage.setItem("provider_last_cloud", current.provider);
      next = "ollama";
    }
    localStorage.setItem("provider", next);
    setProviderInfo(readProviderInfo());
  }, []);
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      if (e.key !== "l" && e.key !== "L") return;
      if (!(e.metaKey || e.ctrlKey)) return;
      // Don't steal Cmd+L from focused inputs (e.g. URL-style address bars
      // are uncommon in the overlay, but be safe).
      const tag = (document.activeElement?.tagName ?? "").toLowerCase();
      if (tag === "input" || tag === "textarea") return;
      e.preventDefault();
      flipProvider();
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [flipProvider]);

  // Hand off the current chat to a detached, normal-chrome window so the
  // user can keep working with other apps while the answer stays visible.
  const pinToDetachedWindow = useCallback(async () => {
    try {
      const sendCrop = await captureFreshSendCrop();
      const b64 = await ensureSendableImage(sendCrop);
      await invoke("open_chat_window", {
        pngB64: b64,
        width: sendCrop.width,
        height: sendCrop.height,
        provider: providerInfo.provider,
        model: providerInfo.model,
        messagesJson: JSON.stringify(messages),
      });
      // Close the overlay once the detached window is open — the user can
      // re-trigger another capture; the detached chat keeps the previous
      // session intact.
      await invoke("close_overlay");
    } catch (e) {
      console.error("open_chat_window failed:", e);
      setStatusToast({ text: "Could not pin chat", key: Date.now() });
    }
  }, [captureFreshSendCrop, ensureSendableImage, providerInfo, messages]);

  const runAi = async (history: ChatMessage[]) => {
    const info = readProviderInfo();
    setProviderInfo(info);

    setError(null);
    setStreaming("");

    // Refresh capture at send-time so the AI sees the current screen, not
    // a stale snapshot from when the overlay first opened. Multi-turn
    // follow-ups otherwise re-send the same cached pixels each turn.
    const sendCrop = await captureFreshSendCrop();

    let imageB64: string;
    try {
      imageB64 = await ensureSendableImage(sendCrop);
    } catch (e) {
      console.error("composeEditedCrop failed:", e);
      imageB64 = sendCrop.png_base64; // fall back to un-annotated PNG
    }

    const channel = new Channel<AskEvent>();
    let acc = "";
    // Boxed in a ref so the closure assignment isn't narrowed away by TS's
    // control-flow analysis after the await.
    const usageBox: { value: { inputTokens: number; outputTokens: number } | null } = {
      value: null,
    };
    channel.onmessage = (event) => {
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
        responseProfile: preferences.aiResponseStyle,
        messages: history,
        imageB64,
        onChunk: channel,
      });
      const usage = usageBox.value;
      const assistant: ChatMessage = {
        role: "assistant",
        content: acc,
        usage: usage
          ? {
              ...usage,
              provider: info.provider,
              model: info.model,
              costCents: estimateCostCents(
                info.provider as ProviderId,
                info.model,
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
          provider: info.provider as ProviderId,
          model: info.model,
          inputTokens: usage.inputTokens,
          outputTokens: usage.outputTokens,
        });
      }
      if (!historySavedRef.current && acc.trim()) {
        historySavedRef.current = true;
        const lastUser = [...history].reverse().find((m) => m.role === "user");
        saveHistoryEntry({
          pngB64: imageB64,
          width: sendCrop.width,
          height: sendCrop.height,
          provider: info.provider,
          model: info.model,
          prompt: lastUser?.content ?? "",
          response: acc,
        });
      }
    } catch (e) {
      setError(formatAiError(e, info));
      // Don't drop a partial response on error — commit it.
      if (acc) {
        setMessages((prev) => [...prev, { role: "assistant", content: acc }]);
      }
      setStreaming(null);
    }
  };

  const askedFirstRef = useRef(false);
  useEffect(() => {
    if (askedFirstRef.current) return;
    askedFirstRef.current = true;
    runAi(messages);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const sendUser = (text: string) => {
    if (streaming !== null) return;
    const content = text.trim()
      ? text
      : preferences.overlayAllowEmptySend
        ? "Describe what's shown in this image."
        : "";
    if (!content.trim()) return;
    const next = [...messages, { role: "user" as const, content }];
    setMessages(next);
    runAi(next);
  };

  // Toast bus for one-shot status messages ("OCR copied", "Saved to …").
  const [statusToast, setStatusToast] = useState<{
    text: string;
    key: number;
  } | null>(null);
  useEffect(() => {
    if (!statusToast) return;
    const t = setTimeout(() => setStatusToast(null), 2200);
    return () => clearTimeout(t);
  }, [statusToast]);

  /// Special template handler. Most templates flow through the regular
  /// chat path; the OCR-clipboard template skips the chat entirely.
  const onTemplatePick = async (t: PromptTemplate) => {
    if (t.id === OCR_CLIPBOARD_TEMPLATE_ID) {
      try {
        setStatusToast({ text: "Extracting text…", key: Date.now() });
        // Use the on-device Vision OCR — no network, no AI tokens, no
        // provider dependency. Pipes the cropped (or annotation-baked)
        // PNG straight into the native command.
        const pngB64 = await ensureSendableImage();
        const txt = (await invoke<string>("ocr_image_local", { pngB64 })).trim();
        if (!txt) {
          setStatusToast({ text: "No text found", key: Date.now() });
          return;
        }
        await navigator.clipboard.writeText(txt).catch(() => {});
        const preview = txt.length > 48 ? txt.slice(0, 45) + "…" : txt;
        setStatusToast({ text: `Copied: ${preview}`, key: Date.now() });
      } catch (e) {
        console.error("OCR-to-clipboard failed:", e);
        const msg = typeof e === "string" ? e : (e as Error).message ?? String(e);
        setStatusToast({ text: `OCR failed: ${msg}`, key: Date.now() });
      }
      return;
    }
    sendUser(t.prompt);
  };

  /// Save the annotated cropped image to disk via the Rust command. Strokes
  /// are baked in via composeEditedCrop; result-mode `cropped` is used as
  /// the source so re-cropping behaviour is preserved.
  const saveAnnotatedToDisk = async () => {
    try {
      const b64 = await ensureSendableImage();
      const path = await invoke<string>("save_annotated_image", { pngB64: b64 });
      const tail = path.split("/").pop() ?? "Screenie.png";
      setStatusToast({ text: `Saved · ${tail}`, key: Date.now() });
    } catch (e) {
      console.error("save_annotated_image failed:", e);
      setStatusToast({ text: "Save failed", key: Date.now() });
    }
  };

  // After a rect drag completes, re-crop so the next user-sent message
  // automatically attaches the new region. The seq guard ensures rapid
  // back-to-back drags don't see an out-of-order crop result clobber the
  // latest one. Strokes are translated into the new image-space; any that
  // fall fully outside the new crop are trimmed.
  const cropSeqRef = useRef(0);
  const recropAfterDrag = async (newRect: Rect, startRect: Rect) => {
    // No real move — a click on a handle without dragging. Skip the crop work.
    if (
      newRect.x === startRect.x &&
      newRect.y === startRect.y &&
      newRect.w === startRect.w &&
      newRect.h === startRect.h
    ) {
      return;
    }
    const seq = ++cropSeqRef.current;
    const oldDims = { width: cropped.width, height: cropped.height };
    try {
      const c = await invoke<CroppedCapture>("crop_capture", {
        srcB64: screen.png_base64,
        x: Math.round(newRect.x * dpr),
        y: Math.round(newRect.y * dpr),
        w: Math.round(newRect.w * dpr),
        h: Math.round(newRect.h * dpr),
      });
      if (seq !== cropSeqRef.current) return;
      setCropped(c);
      setHistoryPreviewB64(null);
      editCtl.remapForCrop(
        startRect,
        newRect,
        oldDims,
        { width: c.width, height: c.height },
      );
    } catch (e) {
      console.error("recrop failed:", e);
    }
  };
  const beginRectDrag = useRectDrag(rect, setRect, recropAfterDrag);

  // The panel is derived from the capture rect. It uses non-overlapping slots
  // first: right/left, then above/below while reserving space for the prompt
  // toolbar. If the capture consumes the whole useful viewport, the panel
  // stays docked behind a small tab until the user explicitly opens it.
  const [chatVisible, setChatVisible] = useState(
    () => preferences.overlayChatDefault !== "hidden",
  );
  const [allowOverlapChat, setAllowOverlapChat] = useState(
    () => preferences.overlayChatDefault === "open",
  );
  const W = window.innerWidth;
  const H = window.innerHeight;
  // The chat panel needs to know the toolbar's natural footprint so it can
  // pick the side that avoids the toolbar. The toolbar no longer shifts to
  // dodge the panel in safe-slot mode — it stays centered on the rect, and
  // the panel is the one that flexes left/right.
  const toolbarBox = computeToolbarBbox(
    rect,
    W,
    H,
    preferences.overlayShowPresets,
    OVERLAY_PAD,
  );
  const safePanel = computeSafeChatPanel(rect, W, H, toolbarBox);
  const hasSafePanel = safePanel !== null;
  const floatingPanel = overlapPanelRect
    ? {
        ...clampFloatingPanel(overlapPanelRect, W, H),
        slot: "overlap" as const,
        overlapsCapture: true,
      }
    : null;
  const overlapPanel = !floatingPanel && allowOverlapChat
    ? {
        ...clampFloatingPanel(overlapPanelRect ?? computeOverlapChatPanel(rect, W, H), W, H),
        slot: "overlap" as const,
        overlapsCapture: true,
      }
    : null;
  const panel = floatingPanel ?? safePanel ?? overlapPanel;
  const chatPanelVisible = chatVisible && panel !== null;

  useEffect(() => {
    if (
      preferences.overlayChatDefault !== "open" &&
      !hasSafePanel &&
      !allowOverlapChat
    ) {
      setChatVisible(false);
      return;
    }
    if (hasSafePanel && allowOverlapChat) {
      setAllowOverlapChat(false);
    }
    if (
      preferences.overlayChatDefault !== "hidden" &&
      hasSafePanel &&
      (streaming !== null || messages.length > 1)
    ) {
      setChatVisible(true);
    }
  }, [
    allowOverlapChat,
    hasSafePanel,
    preferences.overlayChatDefault,
    streaming,
    messages.length,
  ]);

  // Keep the collapsed chat affordance attached to the capture's top-right:
  // above the region when there is room, otherwise tucked just inside it.
  // Sized to match the Save / OCR action pills (36×36 circular icon button)
  // so the row of affordances reads as one family.
  const SHOW_CHAT_W = EDIT_AFFORDANCE_SIZE;
  const SHOW_CHAT_H = EDIT_AFFORDANCE_SIZE;
  const showChatAction = placeShowChatAction(rect, W, H, SHOW_CHAT_W, SHOW_CHAT_H);
  // Edit affordance + companion action buttons avoid the toolbar, chat panel,
  // and the show-chat pill when present.
  const showChatBox = !chatPanelVisible
    ? { x: showChatAction.x, y: showChatAction.y, w: SHOW_CHAT_W, h: SHOW_CHAT_H }
    : null;
  const editAnchor = placeEditAffordance(
    rect,
    W,
    H,
    toolbarBox,
    chatPanelVisible && panel
      ? { x: panel.x, y: panel.y, w: panel.w, h: panel.h }
      : null,
    showChatBox,
  );
  const pillBox =
    editCtl.open && !editAnchor.hidden ? editAnchorBox(editAnchor, true) : null;
  const hiddenHandles = useObscuredHandles(rect, pillBox);
  const affordanceAvoidBoxes: PlacementBox[] = [toolbarBox];
  if (chatPanelVisible && panel) {
    affordanceAvoidBoxes.push({ x: panel.x, y: panel.y, w: panel.w, h: panel.h });
  }
  if (showChatBox) {
    affordanceAvoidBoxes.push(showChatBox);
  }
  const modelOptions = modelOptionsForProvider(
    providerInfo.provider,
    providerInfo.model,
    ollamaModels,
  );
  const selectedModel =
    modelOptions.find((option) => option.value === providerInfo.model) ?? modelOptions[0];
  const modelLabel = selectedModel ? compactModelLabel(selectedModel.label) : providerInfo.model;
  const saveOverlayModel = (model: string) => {
    localStorage.setItem(modelStorageKey(providerInfo.provider), model);
    setProviderInfo(readProviderInfo());
  };
  const canStartFloatingPanelDrag = (event: React.MouseEvent<HTMLElement>) => {
    if (event.button !== 0) return false;
    const target = event.target;
    return target instanceof Element
      ? !target.closest(FLOATING_PANEL_DRAG_BLOCK_SELECTOR)
      : true;
  };
  const beginOverlapPanelMove = (event: React.MouseEvent<HTMLElement>) => {
    if (!panel || !canStartFloatingPanelDrag(event)) return;
    event.preventDefault();
    event.stopPropagation();
    overlapPanelSessionRef.current = {
      kind: "move",
      start: panel,
      mouseX: event.clientX,
      mouseY: event.clientY,
    };
    setOverlayMouseCapture(true);
    setOverlapPanelRect(panel);
    setOverlapPanelCursor("move");
  };
  const beginPanelSurfaceDrag = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!panel || overlapPanelSessionRef.current) return;
    if (!canStartFloatingPanelDrag(event)) return;

    const edge = panel.overlapsCapture
      ? panelResizeEdge(event.clientX, event.clientY, event.currentTarget)
      : null;
    if (!edge) {
      beginOverlapPanelMove(event);
      return;
    }

    event.preventDefault();
    event.stopPropagation();
    overlapPanelSessionRef.current = {
      kind: "resize",
      edge,
      start: panel,
      mouseX: event.clientX,
      mouseY: event.clientY,
    };
    setOverlayMouseCapture(true);
    setOverlapPanelRect(panel);
    setOverlapPanelCursor(panelCursor(edge));
  };
  const updateOverlapPanelCursor = (event: React.MouseEvent<HTMLDivElement>) => {
    if (!panel?.overlapsCapture || overlapPanelSessionRef.current) return;
    setOverlapPanelCursor(panelCursor(panelResizeEdge(event.clientX, event.clientY, event.currentTarget)));
  };

  return (
    <div
      style={fullLayer}
      onMouseDown={(e) => {
        // Only consider clicks that landed directly on the dim backdrop, not
        // on the rect, handles, chat panel, toolbar, or any other interactive
        // child. This prevents accidental dismissal during a drag.
        if (e.target !== e.currentTarget) return;
        // Editor takes priority: a stray click outside the pill/popover
        // collapses the editor instead of closing the overlay.
        if (editCtl.popoverOpen) {
          editCtl.setPopoverOpen(false);
          return;
        }
        if (editCtl.open) {
          editCtl.setOpen(false);
          editCtl.setTool(null);
          return;
        }
        // Empty overlay space is intentionally not a close affordance. In
        // result mode native passthrough normally sends these clicks/hover
        // events to the app underneath; this handler is just a fallback for
        // the moments before native region state catches up.
      }}
    >
      {/* Captured pixels are passthrough; move/resize lives on the outer frame. */}
      {historyPreviewB64 && (
        <img
          src={`data:image/png;base64,${historyPreviewB64}`}
          alt=""
          aria-hidden
          draggable={false}
          style={{
            position: "absolute",
            left: rect.x,
            top: rect.y,
            width: rect.w,
            height: rect.h,
            objectFit: "fill",
            borderRadius: 2,
            pointerEvents: "none",
            userSelect: "none",
          }}
        />
      )}
      <div
        className="screenie-capture-region"
        onMouseDown={beginRectDrag("move", { relayClickThrough: true })}
        onWheel={(e) => {
          if (editCtl.tool) return;
          e.preventDefault();
          e.stopPropagation();
          relayOverlayWheel(e.deltaX, e.deltaY);
        }}
        style={{
          position: "absolute",
          left: rect.x,
          top: rect.y,
          width: rect.w,
          height: rect.h,
          boxShadow: `0 0 0 9999px rgba(0,0,0,0.58)`,
          border: "1px solid rgba(255,255,255,0.5)",
          boxSizing: "border-box",
          borderRadius: 2,
          cursor: "default",
          pointerEvents: editCtl.tool ? "none" : "auto",
        }}
      />
      {moveHitAreas(rect).map((style, i) => (
        <div
          key={`move-${i}`}
          className="screenie-move-hit"
          onMouseDown={beginRectDrag("move")}
          style={{ ...style, pointerEvents: editCtl.tool ? "none" : "auto" }}
        />
      ))}
      {HANDLES.filter((h) => !hiddenHandles.has(h)).map((h) => (
        <div key={`v-${h}`} className="screenie-handle" style={handleStyle(rect, h)} />
      ))}
      {HANDLES.map((h) => (
        <div
          key={`hit-${h}`}
          className="screenie-handle-hit"
          onMouseDown={beginRectDrag(h)}
          style={{
            ...handleHitArea(rect, h),
            pointerEvents: editCtl.tool ? "none" : "auto",
          }}
        />
      ))}
      <EditCanvas
        ctl={editCtl}
        rect={rect}
        cropped={{ width: cropped.width, height: cropped.height }}
        active={editCtl.tool !== null}
        colorPickerSource={{
          b64: cropped.png_base64,
          offsetX: 0,
          offsetY: 0,
        }}
      />
      <EditAffordance
        ctl={editCtl}
        anchor={editAnchor}
        screenPngB64={screen.png_base64}
        screenW={screen.width / dpr}
        screenH={screen.height / dpr}
        avoidBoxes={affordanceAvoidBoxes}
      />
      <ActionsBar
        pencilAnchor={editAnchor}
        pencilOpen={editCtl.open}
        avoidBoxes={affordanceAvoidBoxes}
        rect={rect}
        screenW={W}
        screenH={H}
        screenPngB64={screen.png_base64}
        screenCssW={screen.width / dpr}
        screenCssH={screen.height / dpr}
        buttons={[
          {
            icon: <Download size={15} strokeWidth={1.85} aria-hidden />,
            label: "Save image",
            onClick: () => {
              void saveAnnotatedToDisk();
            },
          },
          {
            icon: <ScanText size={15} strokeWidth={1.85} aria-hidden />,
            label: "OCR → Clipboard",
            // Local Vision OCR doesn't depend on the AI stream, so this
            // is enabled even while a chat response is mid-flight.
            onClick: () => onTemplatePick(OCR_CLIPBOARD_TEMPLATE),
          },
        ]}
      />
      {editCtl.trimmedNotice && (
        <TrimmedNoticeToast
          count={editCtl.trimmedNotice.count}
          onDismiss={editCtl.dismissTrimmedNotice}
          screen={screen}
          dpr={dpr}
        />
      )}
      {editCtl.pickedColor && (
        <PickedColorToast
          hex={editCtl.pickedColor.hex}
          onDismiss={() => editCtl.setPickedColor(null)}
          screen={screen}
          dpr={dpr}
        />
      )}
      {statusToast && (
        <StatusToast
          text={statusToast.text}
          onDismiss={() => setStatusToast(null)}
          screen={screen}
          dpr={dpr}
        />
      )}

      {/* Chat panel — outer wrapper owns the spawn animation (transform-based
          keyframe), inner static surface owns the backdrop-filter. Keeping
          frost off the transformed element prevents WebKit from dropping the
          blur during/after the animation. */}
      {chatPanelVisible && panel && (
        <div
          className="screenie-chat-spawn"
          style={{
            position: "absolute",
            left: panel.x,
            top: panel.y,
            width: panel.w,
            height: panel.h,
          }}
        >
          <div
            className="screenie-chat-panel"
            data-overlap={panel.overlapsCapture}
            onMouseDown={beginPanelSurfaceDrag}
            onMouseMove={updateOverlapPanelCursor}
            onMouseLeave={() => {
              if (!overlapPanelSessionRef.current) setOverlapPanelCursor("default");
            }}
            style={{
              width: "100%",
              height: "100%",
              borderRadius: CHAT_PANEL_RADIUS,
              overflow: "hidden",
              display: "flex",
              flexDirection: "column",
              cursor: panel.overlapsCapture ? overlapPanelCursor : undefined,
            }}
          >
            <BlurredBackdrop
              src={screen.png_base64}
              screenW={screen.width / dpr}
              screenH={screen.height / dpr}
              blurRadius={CHAT_PANEL_FROST.blurRadius}
              imageBrightness={CHAT_PANEL_FROST.imageBrightness}
              tint={CHAT_PANEL_FROST.tint}
              fill={CHAT_PANEL_FROST.fill}
              persistImage
            />
            <div
              className="screenie-chat-titlebar"
              data-draggable="true"
              onMouseDown={beginOverlapPanelMove}
              style={{
                height: 38,
                display: "flex",
                alignItems: "center",
                justifyContent: "space-between",
                padding: "0 8px 0 12px",
                userSelect: "none",
                gap: 8,
              }}
            >
              <div style={{ display: "flex", alignItems: "center", gap: 6 }}>
                <CustomDropdown
                  value={providerInfo.model}
                  options={modelOptions}
                  onChange={saveOverlayModel}
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
                      <span className="screenie-model-separator">·</span>
                      <span className="screenie-model-name">{modelLabel}</span>
                    </span>
                  }
                />
              </div>
              <div style={{ display: "flex", alignItems: "center", gap: 4 }}>
                <button
                  type="button"
                  className="screenie-titlebar-icon"
                  onClick={() => setChatView((v) => (v === "history" ? "chat" : "history"))}
                  onMouseDown={(e) => e.stopPropagation()}
                  aria-label={chatView === "history" ? "Back to chat" : "Open history"}
                  data-active={chatView === "history"}
                >
                  <Clock size={13} strokeWidth={1.85} aria-hidden />
                </button>
                <button
                  type="button"
                  className="screenie-titlebar-icon"
                  onClick={pinToDetachedWindow}
                  onMouseDown={(e) => e.stopPropagation()}
                  aria-label="Pin chat to detached window"
                  style={{ marginRight: 8 }}
                >
                  <ExternalLink size={13} strokeWidth={1.85} aria-hidden />
                </button>
                <button
                  className="screenie-close-btn"
                  onClick={() => {
                    setChatVisible(false);
                    setAllowOverlapChat(false);
                  }}
                  onMouseDown={(e) => e.stopPropagation()}
                  aria-label="Close chat panel"
                >
                  <X size={8} strokeWidth={2.5} aria-hidden />
                </button>
              </div>
            </div>

            {chatView === "history" ? (
              <HistoryList
                onOpen={(entry, b64) => {
                  const cap: CroppedCapture = {
                    png_base64: b64,
                    width: entry.width,
                    height: entry.height,
                  };
                  setCropped(cap);
                  setHistoryPreviewB64(b64);
                  setMessages([
                    { role: "user", content: entry.prompt },
                    { role: "assistant", content: entry.response },
                  ]);
                  historySavedRef.current = true; // already in history
                  setChatView("chat");
                }}
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
                  marginRight: 6,
                  marginBottom: 8,
                  padding: "12px 8px 6px 14px",
                  display: "flex",
                  flexDirection: "column",
                  gap: 12,
                }}
              >
                {messages.map((m, i) => (
                  <MessageBubble
                    key={i}
                    message={m}
                    renderDensity={preferences.aiRenderDensity}
                  />
                ))}
                {streaming !== null && (
                  <MessageBubble
                    key={messages.length}
                    message={{ role: "assistant", content: streaming }}
                    renderDensity={preferences.aiRenderDensity}
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
            <SvgInsetBorder radius={CHAT_PANEL_RADIUS} />
          </div>
        </div>
      )}

      {/* Reused for follow-up replies. The toolbar stays centered on the
          rect; in safe-slot mode the chat panel already picks a side that
          avoids the toolbar (see computeSafeChatPanel). The avoidance shift
          only kicks in for overlap mode, where the panel is a draggable
          floater the user explicitly opened over the capture. While
          streaming, the send button becomes a stop button that trips the
          Rust-side cancel flag — partial response stays in the bubble. */}
      <Toolbar
        rect={rect}
        screen={screen}
        dpr={dpr}
        onSend={sendUser}
        onTemplate={onTemplatePick}
        disabled={streaming !== null}
        streaming={streaming !== null}
        onStop={() => {
          invoke("cancel_ai").catch((e) =>
            console.error("cancel_ai failed:", e),
          );
        }}
        avoidPanel={
          chatPanelVisible && panel?.overlapsCapture ? panel : null
        }
        allowEmpty={preferences.overlayAllowEmptySend}
        showPresets={preferences.overlayShowPresets}
      />

      {/* Bring chat back when hidden — placed near the captured rect */}
      {!chatPanelVisible && (
        <div
          onMouseDown={(e) => e.stopPropagation()}
          style={{
            position: "absolute",
            left: showChatAction.x,
            top: showChatAction.y,
            zIndex: 10,
          }}
        >
          <button
            type="button"
            // Same class duo + recipe the Save / OCR pills use in
            // ActionsBar — gives this affordance the matching frost,
            // hover scale, and inset border without a duplicate
            // `.screenie-action` style branch.
            className={`screenie-edit-pill screenie-action-btn ${streaming !== null ? "screenie-pulse" : ""}`}
            onClick={() => {
              if (!hasSafePanel) {
                setOverlapPanelRect((current) => current ?? computeOverlapChatPanel(rect, W, H));
                setAllowOverlapChat(true);
              }
              setChatVisible(true);
            }}
            aria-label="Show chat"
            style={{
              width: SHOW_CHAT_W,
              height: SHOW_CHAT_H,
              borderRadius: 9999,
              padding: 0,
              border: "none",
              color: "rgba(255, 255, 255, 0.95)",
              cursor: "pointer",
              overflow: "hidden",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontFamily: "inherit",
            }}
          >
            <BlurredBackdrop
              src={screen.png_base64}
              screenW={screen.width / dpr}
              screenH={screen.height / dpr}
              blurRadius={26}
              imageBrightness={0.64}
              tint="rgba(34, 36, 35, 0.43)"
              fill="rgba(18, 19, 18, 0.17)"
              persistImage
            />
            <span style={{ position: "relative", zIndex: 1, display: "flex" }}>
              <MessageSquare size={15} strokeWidth={1.85} aria-hidden />
            </span>
            <SvgInsetBorder radius={9999} />
          </button>
        </div>
      )}
      <Hint
        text="Drag the box to recrop · Type to follow up · Esc to close"
        screen={screen}
        dpr={dpr}
      />
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Message bubble (with copy)                                          */
/* ------------------------------------------------------------------ */

function MessageBubble({
  message,
  renderDensity,
  streaming,
}: {
  message: ChatMessage;
  renderDensity: AiRenderDensityPreference;
  streaming?: boolean;
}) {
  const [copied, setCopied] = useState(false);

  // Smooth typewriter reveal for assistant bubbles. Bursty network chunks
  // (especially from cloud providers) would otherwise dump big paragraphs
  // in one frame; we drain the buffer at a controlled rate that scales
  // with backlog so prose unfolds at a readable pace, then accelerate
  // once the upstream stream has finished so we never strand the user
  // staring at a half-rendered response.
  //
  // Initial state matches the bubble's lifecycle:
  //  - mounted as the in-flight bubble (`streaming` true): start empty
  //  - mounted as a finalized bubble re-rendered from history: full content
  const isAssistant = message.role === "assistant";
  const [revealed, setRevealed] = useState(() =>
    streaming && isAssistant ? "" : message.content,
  );
  const targetRef = useRef(message.content);
  targetRef.current = message.content;
  const streamingRef = useRef(streaming);
  streamingRef.current = streaming;

  useEffect(() => {
    if (!isAssistant) return;
    let rafId = 0;
    let lastTime = 0;
    let cancelled = false;

    const tick = (now: number) => {
      if (cancelled) return;
      const dt = lastTime ? Math.min(now - lastTime, 100) : 16;
      lastTime = now;

      setRevealed((current) => {
        const target = targetRef.current;
        if (current.length >= target.length) return current;
        const pending = target.length - current.length;
        // While streaming: smooth ~70 chars/sec baseline that accelerates
        // when the buffer grows so we never lag too far behind. After the
        // upstream stream ends, switch to aggressive catch-up.
        const drainRate = streamingRef.current
          ? Math.max(70, Math.min(500, pending * 4))
          : 1500;
        const charsToAdd = Math.max(1, Math.ceil((drainRate * dt) / 1000));
        return target.slice(0, current.length + charsToAdd);
      });

      rafId = requestAnimationFrame(tick);
    };

    rafId = requestAnimationFrame(tick);
    return () => {
      cancelled = true;
      cancelAnimationFrame(rafId);
    };
  }, [isAssistant]);

  const displayContent = isAssistant ? revealed : message.content;
  const fullyRevealed = revealed.length >= message.content.length;
  const showThinking = !!streaming && isAssistant && displayContent.length === 0;
  // Show the trailing cursor while we're actively streaming OR while we're
  // still draining the buffer post-stream — both signal "more to come".
  const showCursor =
    isAssistant &&
    displayContent.length > 0 &&
    (!!streaming || !fullyRevealed);

  const formattedContent = useMemo(() => {
    if (!isAssistant) return displayContent;
    try {
      return formatAiMarkdown(displayContent);
    } catch (e) {
      // If pre-formatting blows up on a partial chunk during streaming,
      // fall back to the unformatted markdown so the bubble still renders.
      console.error("formatAiMarkdown failed:", e);
      return displayContent;
    }
  }, [displayContent, isAssistant]);

  const copy = () => {
    if (!message.content) return;
    let toCopy = message.content;
    try {
      toCopy = formatAiMarkdown(message.content);
    } catch {
      /* fall back to raw content */
    }
    navigator.clipboard
      .writeText(toCopy)
      .then(() => {
        setCopied(true);
        setTimeout(() => setCopied(false), 1400);
      })
      .catch(() => {
        /* clipboard unavailable */
      });
  };

  if (message.role === "user") {
    return (
      <div
        style={{
          alignSelf: "flex-end",
          maxWidth: "85%",
          background: "rgba(253, 252, 248, 0.97)",
          padding: "9px 13px",
          borderRadius: 14,
          fontSize: 13.5,
          lineHeight: 1.45,
          color: "rgba(18, 18, 16, 0.96)",
          whiteSpace: "pre-wrap",
        }}
      >
        {message.content}
      </div>
    );
  }
  return (
    <div style={{ alignSelf: "flex-start", width: "100%" }}>
      <div
        className="screenie-md"
        data-density={renderDensity}
        style={{
          fontSize: 13.5,
          lineHeight: 1.55,
          color: "rgba(255,255,255,0.92)",
        }}
      >
        {showThinking ? (
          <span className="screenie-thinking-shimmer">Thinking…</span>
        ) : displayContent ? (
          <ReactMarkdown
            remarkPlugins={[remarkGfm, remarkMath]}
            rehypePlugins={[[rehypeKatex, SCREENIE_KATEX_OPTIONS], rehypeHighlight]}
          >
            {formattedContent}
          </ReactMarkdown>
        ) : null}
        {showCursor && (
          <span className="screenie-pulse" style={{ opacity: 0.5 }}>
            ▍
          </span>
        )}
      </div>
      {!streaming && fullyRevealed && message.content && (
        <div
          style={{
            marginTop: 6,
            display: "flex",
            alignItems: "center",
            gap: 8,
            flexWrap: "wrap",
          }}
        >
          <button
            className={`screenie-copy-btn ${copied ? "copied" : ""}`}
            onClick={copy}
            aria-label="Copy response"
          >
            {copied ? (
              <Check size={15} strokeWidth={2} aria-hidden />
            ) : (
              <Copy size={15} strokeWidth={1.9} aria-hidden />
            )}
          </button>
          {message.usage && (
            <span className="screenie-usage-chip">
              {formatUsageSummary(message.usage)}
            </span>
          )}
        </div>
      )}
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Permission banner — shown when capture comes back all-black         */
/* ------------------------------------------------------------------ */

function PermissionBanner({
  onClose,
  onRetry,
  onDismissPermanently,
}: {
  onClose: () => void;
  onRetry?: () => void;
  onDismissPermanently?: () => void;
}) {
  // Same banner UI on both platforms — only the explanatory copy and the
  // settings deep-link button label change. On Mac a blank capture nearly
  // always means Screen Recording permission was denied; on Windows it
  // typically means a DRM-protected window (Netflix, banking app with
  // capture protection) covered the region, since BitBlt itself has no
  // permission gate.
  const isMac =
    typeof navigator !== "undefined" && navigator.userAgent.includes("Mac");
  const explanation = isMac
    ? "If you've already enabled Screen Recording for Screenie AI in System Settings → Privacy & Security, you still need to quit and reopen the app — macOS caches the permission state per running process and only re-checks at launch. Use Quit & relaunch below. If you haven't granted it yet, open System Settings first."
    : "This usually means the region you captured contained DRM-protected content (Netflix, some banking apps, fullscreen games) or a window that opts out of capture. If the screen is intentionally black, close this banner and retry on a different region.";
  const settingsLabel = isMac
    ? "Open System Settings ↗"
    : "Open Privacy Settings ↗";
  return (
    <div
      style={{
        ...rootStyle,
        background: "rgba(8, 8, 12, 0.78)",
        display: "flex",
        alignItems: "center",
        justifyContent: "center",
      }}
    >
      <div
        className="screenie-fadein"
        onMouseDown={(e) => e.stopPropagation()}
        style={{
          width: 460,
          maxWidth: "calc(100vw - 32px)",
          padding: "26px 26px 22px",
          borderRadius: 18,
          background: "rgba(48, 50, 49, 0.74)",
          backdropFilter: "blur(46px) saturate(132%) brightness(0.86)",
          WebkitBackdropFilter: "blur(46px) saturate(132%) brightness(0.86)",
          border: "1px solid rgba(255, 255, 255, 0.14)",
          color: "rgba(255,255,255,0.95)",
        }}
      >
        <div
          style={{
            display: "flex",
            alignItems: "center",
            gap: 10,
            fontSize: 11,
            letterSpacing: 0.4,
            textTransform: "uppercase",
            opacity: 0.7,
            marginBottom: 8,
          }}
        >
          <span
            style={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              background: "#f59e0b",
            }}
          />
          Blank capture
        </div>
        <h2 style={{ margin: 0, fontSize: 18, fontWeight: 600 }}>
          Screenie captured a blank screen
        </h2>
        <p
          style={{
            margin: "10px 0 16px",
            fontSize: 13,
            lineHeight: 1.55,
            opacity: 0.82,
          }}
        >
          {explanation}
        </p>
        <div
          style={{
            display: "flex",
            gap: 8,
            justifyContent: "flex-end",
            flexWrap: "wrap",
          }}
        >
          <button className="screenie-action" onClick={onRetry ?? onClose}>
            Close
          </button>
          <button
            className="screenie-action"
            onClick={() => {
              invoke("open_screen_settings");
            }}
          >
            {settingsLabel}
          </button>
          {isMac && (
            <button
              className="screenie-action"
              style={{
                background: "rgba(255,255,255,0.95)",
                color: "#0a0a0a",
                borderColor: "rgba(255,255,255,1)",
              }}
              onClick={() => {
                invoke("restart_app").catch((e) =>
                  console.error("restart_app failed:", e),
                );
              }}
            >
              Quit &amp; relaunch
            </button>
          )}
        </div>
        {onDismissPermanently && (
          <div style={{ marginTop: 14, textAlign: "center" }}>
            <button
              type="button"
              onClick={onDismissPermanently}
              style={{
                background: "transparent",
                border: "none",
                padding: 0,
                fontSize: 11.5,
                color: "rgba(255,255,255,0.55)",
                textDecoration: "underline",
                cursor: "pointer",
                fontFamily: "inherit",
              }}
            >
              Don&apos;t show this again
            </button>
          </div>
        )}
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Helpers                                                             */
/* ------------------------------------------------------------------ */

function selectionRectStyle(rect: Rect): React.CSSProperties {
  return {
    position: "absolute",
    left: rect.x,
    top: rect.y,
    width: rect.w,
    height: rect.h,
    boxSizing: "border-box",
    pointerEvents: "auto",
  };
}

function clamp(n: number, lo: number, hi: number): number {
  return Math.min(Math.max(n, lo), hi);
}

function scaledHandleLength(
  size: number,
  ratio: number,
  minLength: number,
  maxLength: number,
): number {
  const upper = Math.max(1, Math.min(size, maxLength));
  return clamp(size * ratio, Math.min(minLength, upper), upper);
}

function normalizeRect(d: { x0: number; y0: number; x1: number; y1: number }): Rect {
  const x = Math.min(d.x0, d.x1);
  const y = Math.min(d.y0, d.y1);
  const w = Math.abs(d.x1 - d.x0);
  const h = Math.abs(d.y1 - d.y0);
  return { x, y, w, h };
}

function applyDrag(
  start: Rect,
  kind: "move" | Handle,
  dx: number,
  dy: number,
  bounds?: { W: number; H: number },
): Rect {
  let { x, y, w, h } = start;
  if (kind === "move") {
    x += dx;
    y += dy;
    if (bounds) {
      // Move preserves size — just stop the rect at the screen edge.
      w = Math.min(w, bounds.W);
      h = Math.min(h, bounds.H);
      x = clamp(x, 0, bounds.W - w);
      y = clamp(y, 0, bounds.H - h);
    }
    return { x, y, w, h };
  }
  if (kind.includes("n")) {
    y += dy;
    h -= dy;
  }
  if (kind.includes("s")) {
    h += dy;
  }
  if (kind.includes("w")) {
    x += dx;
    w -= dx;
  }
  if (kind.includes("e")) {
    w += dx;
  }
  if (w < MIN_RECT) {
    if (kind.includes("w")) x = start.x + start.w - MIN_RECT;
    w = MIN_RECT;
  }
  if (h < MIN_RECT) {
    if (kind.includes("n")) y = start.y + start.h - MIN_RECT;
    h = MIN_RECT;
  }
  if (bounds) {
    // Resize: when an edge would go off-screen, shrink that dimension instead
    // of pushing the opposite anchor — the rect's "fixed" corner stays put.
    if (x < 0) {
      w = Math.max(MIN_RECT, w + x);
      x = 0;
    }
    if (y < 0) {
      h = Math.max(MIN_RECT, h + y);
      y = 0;
    }
    if (x + w > bounds.W) {
      w = Math.max(MIN_RECT, bounds.W - x);
    }
    if (y + h > bounds.H) {
      h = Math.max(MIN_RECT, bounds.H - y);
    }
  }
  return { x, y, w, h };
}

function handleStyle(rect: Rect, h: Handle): React.CSSProperties {
  const shortEdge = Math.max(1, Math.min(rect.w, rect.h));
  const T = clamp(shortEdge * 0.07, 2, 4);
  const HALF = T / 2;
  const CORNER = scaledHandleLength(shortEdge, 0.32, 8, 22);
  const EDGE_H = scaledHandleLength(rect.w, 0.34, 10, 44);
  const EDGE_V = scaledHandleLength(rect.h, 0.34, 10, 44);
  const color = "rgba(255,255,255,0.98)";

  const base: React.CSSProperties = {
    position: "absolute",
    background: "transparent",
    boxSizing: "border-box",
    borderRadius: 1,
    pointerEvents: "none",
  };

  switch (h) {
    case "nw":
      return {
        ...base,
        left: rect.x - HALF,
        top: rect.y - HALF,
        width: CORNER,
        height: CORNER,
        borderTop: `${T}px solid ${color}`,
        borderLeft: `${T}px solid ${color}`,
      };
    case "ne":
      return {
        ...base,
        left: rect.x + rect.w - CORNER + HALF,
        top: rect.y - HALF,
        width: CORNER,
        height: CORNER,
        borderTop: `${T}px solid ${color}`,
        borderRight: `${T}px solid ${color}`,
      };
    case "se":
      return {
        ...base,
        left: rect.x + rect.w - CORNER + HALF,
        top: rect.y + rect.h - CORNER + HALF,
        width: CORNER,
        height: CORNER,
        borderBottom: `${T}px solid ${color}`,
        borderRight: `${T}px solid ${color}`,
      };
    case "sw":
      return {
        ...base,
        left: rect.x - HALF,
        top: rect.y + rect.h - CORNER + HALF,
        width: CORNER,
        height: CORNER,
        borderBottom: `${T}px solid ${color}`,
        borderLeft: `${T}px solid ${color}`,
      };
    case "n":
      return {
        ...base,
        left: rect.x + rect.w / 2 - EDGE_H / 2,
        top: rect.y - HALF,
        width: EDGE_H,
        height: T,
        background: color,
      };
    case "s":
      return {
        ...base,
        left: rect.x + rect.w / 2 - EDGE_H / 2,
        top: rect.y + rect.h - HALF,
        width: EDGE_H,
        height: T,
        background: color,
      };
    case "e":
      return {
        ...base,
        left: rect.x + rect.w - HALF,
        top: rect.y + rect.h / 2 - EDGE_V / 2,
        width: T,
        height: EDGE_V,
        background: color,
      };
    case "w":
      return {
        ...base,
        left: rect.x - HALF,
        top: rect.y + rect.h / 2 - EDGE_V / 2,
        width: T,
        height: EDGE_V,
        background: color,
      };
  }
}

/// Bbox of the visible resize handle — mirrors the geometry in
/// `handleStyle`. Used by `useObscuredHandles` to decide whether a given
/// handle is covered by the expanded edit pill or popover.
function handleBbox(rect: Rect, h: Handle): PlacementBox {
  const shortEdge = Math.max(1, Math.min(rect.w, rect.h));
  const T = clamp(shortEdge * 0.07, 2, 4);
  const HALF = T / 2;
  const CORNER = scaledHandleLength(shortEdge, 0.32, 8, 22);
  const EDGE_H = scaledHandleLength(rect.w, 0.34, 10, 44);
  const EDGE_V = scaledHandleLength(rect.h, 0.34, 10, 44);
  switch (h) {
    case "nw":
      return { x: rect.x - HALF, y: rect.y - HALF, w: CORNER, h: CORNER };
    case "ne":
      return { x: rect.x + rect.w - CORNER + HALF, y: rect.y - HALF, w: CORNER, h: CORNER };
    case "se":
      return {
        x: rect.x + rect.w - CORNER + HALF,
        y: rect.y + rect.h - CORNER + HALF,
        w: CORNER,
        h: CORNER,
      };
    case "sw":
      return { x: rect.x - HALF, y: rect.y + rect.h - CORNER + HALF, w: CORNER, h: CORNER };
    case "n":
      return { x: rect.x + rect.w / 2 - EDGE_H / 2, y: rect.y - HALF, w: EDGE_H, h: T };
    case "s":
      return { x: rect.x + rect.w / 2 - EDGE_H / 2, y: rect.y + rect.h - HALF, w: EDGE_H, h: T };
    case "e":
      return { x: rect.x + rect.w - HALF, y: rect.y + rect.h / 2 - EDGE_V / 2, w: T, h: EDGE_V };
    case "w":
      return { x: rect.x - HALF, y: rect.y + rect.h / 2 - EDGE_V / 2, w: T, h: EDGE_V };
  }
}

/// Returns the set of resize handles obscured by the expanded edit pill
/// or popover so the calling layer can skip rendering just those.
///
/// The pill's bbox is supplied by the caller (we have it without a DOM
/// query via `editAnchorBox`); the popover's bbox is read from the DOM
/// because EditPopover picks its own placement based on internal sizing
/// logic that the parent doesn't see.
function useObscuredHandles(
  rect: Rect,
  pillBox: PlacementBox | null,
): Set<Handle> {
  const [popoverBox, setPopoverBox] = useState<PlacementBox | null>(null);
  // Re-measure every render: the popover repositions when the pill moves
  // (rect drag), opens/closes on tool switches, and resizes on content
  // changes. State update bails when nothing moved, so this stays cheap.
  useLayoutEffect(() => {
    const el = document.querySelector<HTMLElement>(".screenie-edit-popover");
    if (!el || el.offsetWidth < 1 || el.offsetHeight < 1) {
      setPopoverBox((prev) => (prev === null ? prev : null));
      return;
    }
    const r = el.getBoundingClientRect();
    const next: PlacementBox = { x: r.left, y: r.top, w: r.width, h: r.height };
    setPopoverBox((prev) =>
      prev &&
      Math.abs(prev.x - next.x) < 0.5 &&
      Math.abs(prev.y - next.y) < 0.5 &&
      Math.abs(prev.w - next.w) < 0.5 &&
      Math.abs(prev.h - next.h) < 0.5
        ? prev
        : next,
    );
  });
  const hidden = new Set<Handle>();
  if (!pillBox && !popoverBox) return hidden;
  for (const h of HANDLES) {
    const hb = handleBbox(rect, h);
    if (pillBox && intersectsWithGap(hb, pillBox, 0)) {
      hidden.add(h);
      continue;
    }
    if (popoverBox && intersectsWithGap(hb, popoverBox, 0)) {
      hidden.add(h);
    }
  }
  return hidden;
}

function moveHitAreas(rect: Rect): React.CSSProperties[] {
  const OUTER = 12;
  const reserve = clamp(Math.min(rect.w, rect.h) * 0.22, 18, 48);
  const base: React.CSSProperties = {
    position: "absolute",
    background: "transparent",
    cursor: "default",
    zIndex: 7,
  };
  const areas: React.CSSProperties[] = [];
  const horizontalW = rect.w - reserve * 2;
  if (horizontalW > 8) {
    areas.push({
      ...base,
      left: rect.x + reserve,
      top: rect.y - OUTER,
      width: horizontalW,
      height: OUTER,
    });
    areas.push({
      ...base,
      left: rect.x + reserve,
      top: rect.y + rect.h,
      width: horizontalW,
      height: OUTER,
    });
  }
  const verticalH = rect.h - reserve * 2;
  if (verticalH > 8) {
    areas.push({
      ...base,
      left: rect.x - OUTER,
      top: rect.y + reserve,
      width: OUTER,
      height: verticalH,
    });
    areas.push({
      ...base,
      left: rect.x + rect.w,
      top: rect.y + reserve,
      width: OUTER,
      height: verticalH,
    });
  }
  return areas;
}

/// Invisible click target for each handle. It straddles the visible handle so
/// clicking the drawn corner/edge actually starts resize.
function handleHitArea(rect: Rect, h: Handle): React.CSSProperties {
  const shortEdge = Math.max(1, Math.min(rect.w, rect.h));
  const HIT_CORNER = scaledHandleLength(shortEdge, 0.8, 22, 44);
  const HIT_EDGE_LONG_H = scaledHandleLength(rect.w, 0.55, 28, 72);
  const HIT_EDGE_LONG_V = scaledHandleLength(rect.h, 0.55, 28, 72);
  const HIT_EDGE_PERP = clamp(shortEdge * 0.18, 12, 20);

  const base: React.CSSProperties = {
    position: "absolute",
    background: "transparent",
    pointerEvents: "auto",
    zIndex: 8,
  };
  const cornerHalf = HIT_CORNER / 2;
  const edgeHalf = HIT_EDGE_PERP / 2;

  switch (h) {
    case "nw":
      return {
        ...base,
        left: rect.x - cornerHalf,
        top: rect.y - cornerHalf,
        width: HIT_CORNER,
        height: HIT_CORNER,
        cursor: "nwse-resize",
      };
    case "ne":
      return {
        ...base,
        left: rect.x + rect.w - cornerHalf,
        top: rect.y - cornerHalf,
        width: HIT_CORNER,
        height: HIT_CORNER,
        cursor: "nesw-resize",
      };
    case "se":
      return {
        ...base,
        left: rect.x + rect.w - cornerHalf,
        top: rect.y + rect.h - cornerHalf,
        width: HIT_CORNER,
        height: HIT_CORNER,
        cursor: "nwse-resize",
      };
    case "sw":
      return {
        ...base,
        left: rect.x - cornerHalf,
        top: rect.y + rect.h - cornerHalf,
        width: HIT_CORNER,
        height: HIT_CORNER,
        cursor: "nesw-resize",
      };
    case "n":
      return {
        ...base,
        left: rect.x + rect.w / 2 - HIT_EDGE_LONG_H / 2,
        top: rect.y - edgeHalf,
        width: HIT_EDGE_LONG_H,
        height: HIT_EDGE_PERP,
        cursor: "ns-resize",
      };
    case "s":
      return {
        ...base,
        left: rect.x + rect.w / 2 - HIT_EDGE_LONG_H / 2,
        top: rect.y + rect.h - edgeHalf,
        width: HIT_EDGE_LONG_H,
        height: HIT_EDGE_PERP,
        cursor: "ns-resize",
      };
    case "e":
      return {
        ...base,
        left: rect.x + rect.w - edgeHalf,
        top: rect.y + rect.h / 2 - HIT_EDGE_LONG_V / 2,
        width: HIT_EDGE_PERP,
        height: HIT_EDGE_LONG_V,
        cursor: "ew-resize",
      };
    case "w":
      return {
        ...base,
        left: rect.x - edgeHalf,
        top: rect.y + rect.h / 2 - HIT_EDGE_LONG_V / 2,
        width: HIT_EDGE_PERP,
        height: HIT_EDGE_LONG_V,
        cursor: "ew-resize",
      };
  }
}

/// Top-center floating tooltip. Uses the exact same frost recipe as the
/// prompt toolbar — a BlurredBackdrop bitmap (so the blur survives focus
/// changes and the user never sees the "blur dropped" fallback) plus a
/// SvgInsetBorder for the crisp inner stroke. `persistImage` keeps the
/// bitmap visible across window blur events.
function Hint({
  text,
  screen,
  dpr,
}: {
  text: string;
  screen: ScreenCapture;
  dpr: number;
}) {
  const HINT_RADIUS = 999;
  return (
    <div
      style={{
        position: "absolute",
        left: "50%",
        top: 28,
        transform: "translateX(-50%)",
        pointerEvents: "none",
      }}
    >
      <div
        className="screenie-hint"
        style={{
          position: "relative",
          padding: "7px 14px",
          borderRadius: HINT_RADIUS,
          fontSize: 12.5,
          fontWeight: 500,
          letterSpacing: 0.1,
          whiteSpace: "nowrap",
          color: "rgba(255,255,255,0.95)",
          userSelect: "none",
          overflow: "hidden",
        }}
      >
        <BlurredBackdrop
          src={screen.png_base64}
          screenW={screen.width / dpr}
          screenH={screen.height / dpr}
          blurRadius={26}
          imageBrightness={0.64}
          tint="rgba(34, 36, 35, 0.43)"
          fill="rgba(18, 19, 18, 0.17)"
          persistImage
        />
        <span style={{ position: "relative", zIndex: 1 }}>{text}</span>
        <SvgInsetBorder radius={HINT_RADIUS} />
      </div>
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Styles                                                              */
/* ------------------------------------------------------------------ */

const rootStyle: React.CSSProperties = {
  position: "fixed",
  inset: 0,
  width: "100vw",
  height: "100vh",
  overflow: "hidden",
  background: "transparent",
  cursor: "default",
  fontFamily:
    '-apple-system, BlinkMacSystemFont, "Segoe UI", system-ui, sans-serif',
};

const fullLayer: React.CSSProperties = {
  position: "absolute",
  inset: 0,
  width: "100vw",
  height: "100vh",
};

const dimStyle: React.CSSProperties = {
  position: "absolute",
  inset: 0,
  background: "rgba(0,0,0,0.5)",
  pointerEvents: "none",
};
