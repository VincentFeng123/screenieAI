// Hub for all voice:* event wiring and session state. Listeners register
// once on mount (QuickTooltip renders WITHOUT StrictMode, so mount effects
// run exactly once) and stay mounted for the window's lifetime — late
// voice:task updates must still land after listening stops.

import { useCallback, useEffect, useRef, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { getCurrentWindow } from "@tauri-apps/api/window";
import type {
  VoiceBackendStatus,
  VoiceChip,
  VoiceErrorPayload,
  VoiceLevelPayload,
  VoiceModelDownloadPayload,
  VoiceStatusPayload,
  VoiceTaskPayload,
  VoiceUtterancePayload,
} from "./types";

/** Chips kept in memory; the feed renders fewer (see VoiceTranscriptFeed). */
const MAX_CHIPS = 20;
/** Transient notices clear themselves; the denied notice stays. */
const NOTICE_CLEAR_MS = 8_000;

export type VoiceSessionOptions = {
  /** Same agent settings the typed input passes to start_agent_task. */
  getStartArgs: () => Record<string, unknown>;
};

export type VoiceSession = {
  status: VoiceBackendStatus;
  /** True while a session is live (listening or transcribing). */
  active: boolean;
  /** True while any voice task is queued / running / awaiting confirmation. */
  busy: boolean;
  micDenied: boolean;
  modelMissing: boolean;
  /** Download progress 0-100, or null when no download is in flight. */
  downloadPct: number | null;
  /** Inline message for the voice area of the agent card (errors, auto-stop). */
  notice: string | null;
  chips: VoiceChip[];
  /**
   * Attach to the level-meter FILL element. Levels write straight to the
   * DOM (transform: scaleX) — never mirror rms into React state, that would
   * re-render the whole tooltip at 10 Hz.
   */
  meterFillRef: React.RefObject<HTMLDivElement | null>;
  /** Starts listening; resolves true when the session actually started. */
  start: () => Promise<boolean>;
  stop: () => Promise<void>;
  downloadModel: () => Promise<void>;
};

export function useVoiceSession(options: VoiceSessionOptions): VoiceSession {
  const [status, setStatus] = useState<VoiceBackendStatus>("idle");
  const [micDenied, setMicDenied] = useState(false);
  const [modelMissing, setModelMissing] = useState(false);
  const [downloadPct, setDownloadPct] = useState<number | null>(null);
  const [notice, setNotice] = useState<string | null>(null);
  const [chips, setChips] = useState<VoiceChip[]>([]);
  const meterFillRef = useRef<HTMLDivElement | null>(null);
  const noticeTimerRef = useRef<number | null>(null);

  // Keeps getStartArgs out of the mount-once listener effect's deps.
  const optionsRef = useRef(options);
  optionsRef.current = options;

  const showNotice = useCallback((message: string, sticky = false) => {
    if (noticeTimerRef.current !== null) {
      window.clearTimeout(noticeTimerRef.current);
      noticeTimerRef.current = null;
    }
    setNotice(message);
    if (!sticky) {
      noticeTimerRef.current = window.setTimeout(() => {
        noticeTimerRef.current = null;
        setNotice(null);
      }, NOTICE_CLEAR_MS);
    }
  }, []);

  const clearNotice = useCallback(() => {
    if (noticeTimerRef.current !== null) {
      window.clearTimeout(noticeTimerRef.current);
      noticeTimerRef.current = null;
    }
    setNotice(null);
  }, []);

  const writeLevel = useCallback((rms: number) => {
    const el = meterFillRef.current;
    if (!el) return;
    // Typical speech rms on a normalized f32 stream sits around 0.02-0.2;
    // the multiplier maps that onto a readable bar.
    const scale = Math.min(1, Math.max(0, rms * 8));
    el.style.transform = `scaleX(${scale})`;
  }, []);

  useEffect(() => {
    let cancelled = false;
    const unlisteners: Array<() => void> = [];
    const win = getCurrentWindow();

    const listen = <T,>(event: string, handler: (payload: T) => void) => {
      win
        .listen<T>(event, (e) => {
          if (!cancelled) handler(e.payload);
        })
        .then((off) => {
          if (cancelled) {
            off();
          } else {
            unlisteners.push(off);
          }
        })
        .catch((e) => {
          console.error(`${event} listener failed:`, e);
        });
    };

    listen<VoiceBackendStatus>("voice:status", (next) => {
      setStatus(next);
      if (next === "idle") writeLevel(0);
    });

    listen<VoiceLevelPayload>("voice:level", (payload) => {
      writeLevel(payload.rms);
    });

    listen<VoiceUtterancePayload>("voice:utterance", (payload) => {
      setChips((prev) =>
        [{ id: payload.id, text: payload.text, state: "heard" as const }, ...prev].slice(
          0,
          MAX_CHIPS,
        ),
      );
    });

    listen<VoiceTaskPayload>("voice:task", (payload) => {
      // Unknown ids (pruned chips, stale sessions) are ignored on purpose.
      setChips((prev) =>
        prev.map((chip) =>
          chip.id === payload.id ? { ...chip, state: payload.state } : chip,
        ),
      );
    });

    listen<VoiceModelDownloadPayload>("voice:model_download", (payload) => {
      setDownloadPct(payload.pct);
    });

    listen<VoiceErrorPayload>("voice:error", (payload) => {
      switch (payload.code) {
        case "model_missing":
          setModelMissing(true);
          break;
        case "mic_denied":
          setMicDenied(true);
          showNotice(
            "Microphone access is denied. Enable it in System Settings → Privacy & Security → Microphone (the terminal app during development), then relaunch.",
            true,
          );
          break;
        case "queue_full":
          // The chip is already marked failed via voice:task; stay quiet.
          console.warn("voice queue full:", payload.message);
          break;
        case "download":
          setDownloadPct(null);
          showNotice(payload.message);
          break;
        default:
          // idle_timeout, audio, stt, agent, no_device…
          showNotice(payload.message);
      }
    });

    // Seed mic/model state so the button renders correctly before first use.
    invoke<VoiceStatusPayload>("voice_get_status")
      .then((payload) => {
        if (cancelled) return;
        setStatus(payload.status);
        setModelMissing(!payload.modelPresent);
        setMicDenied(payload.micPermission === "denied");
      })
      .catch((e) => {
        console.error("voice_get_status failed:", e);
      });

    return () => {
      cancelled = true;
      for (const off of unlisteners) off();
    };
  }, [showNotice, writeLevel]);

  const start = useCallback(async () => {
    clearNotice();
    try {
      await invoke("voice_start_listening", optionsRef.current.getStartArgs());
      setMicDenied(false);
      return true;
    } catch (e) {
      // model_missing / mic_denied state flips arrive via voice:error.
      console.error("voice_start_listening failed:", e);
      return false;
    }
  }, [clearNotice]);

  const stop = useCallback(async () => {
    writeLevel(0);
    try {
      await invoke("voice_stop_listening");
    } catch (e) {
      console.error("voice_stop_listening failed:", e);
    }
  }, [writeLevel]);

  const downloadModel = useCallback(async () => {
    clearNotice();
    setDownloadPct(0);
    try {
      await invoke("voice_download_model");
      setDownloadPct(null);
      setModelMissing(false);
    } catch (e) {
      // voice:error already surfaced the message as a notice.
      setDownloadPct(null);
      console.error("voice_download_model failed:", e);
    }
  }, [clearNotice]);

  const active = status !== "idle";
  const busy = chips.some(
    (chip) =>
      chip.state === "queued" ||
      chip.state === "running" ||
      chip.state === "blocked_on_confirmation",
  );

  return {
    status,
    active,
    busy,
    micDenied,
    modelMissing,
    downloadPct,
    notice,
    chips,
    meterFillRef,
    start,
    stop,
    downloadModel,
  };
}
