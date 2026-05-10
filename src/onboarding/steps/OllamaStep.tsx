import { useEffect, useRef, useState } from "react";
import { Channel, invoke } from "@tauri-apps/api/core";
import { OllamaStatus, looksLikeVisionModel } from "../../settings/constants";

const isMac =
  typeof navigator !== "undefined" && navigator.userAgent.includes("Mac");

type InstallStatus =
  | { kind: "downloading"; percent: number }
  | { kind: "extracting" }
  | { kind: "installing" }
  | { kind: "launching" }
  | { kind: "done" };

type PullStatus =
  | { kind: "connecting" }
  | { kind: "pulling"; percent: number; phase: string }
  | { kind: "verifying" }
  | { kind: "done" };

export default function OllamaStep({
  hasCloudKey,
  onNext,
  onBack,
  onSkip,
}: {
  hasCloudKey: boolean;
  onNext: () => void;
  onBack: () => void;
  onSkip: () => void;
}) {
  const [installed, setInstalled] = useState<boolean | null>(null);
  const [status, setStatus] = useState<OllamaStatus>({ running: false, models: [] });
  const [installing, setInstalling] = useState<InstallStatus | null>(null);
  const [pulling, setPulling] = useState<PullStatus | null>(null);
  const [error, setError] = useState<string | null>(null);

  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const refresh = async () => {
    try {
      const [diskOk, st] = await Promise.all([
        invoke<boolean>("check_ollama_installed"),
        invoke<OllamaStatus>("check_ollama"),
      ]);
      if (!mountedRef.current) return;
      setInstalled(diskOk);
      setStatus(st);
    } catch {
      /* swallow — next poll retries */
    }
  };

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 2500);
    return () => clearInterval(id);
  }, []);

  const startInstall = async () => {
    setError(null);
    setInstalling({ kind: "downloading", percent: 0 });
    const ch = new Channel<InstallStatus>();
    ch.onmessage = (msg) => {
      if (mountedRef.current) setInstalling(msg);
    };
    try {
      await invoke("install_ollama", { onProgress: ch });
    } catch (e) {
      // Daemon-failed-to-start is now a real Err from Rust (previously the
      // installer silently marked Done). Surface it inline.
      if (mountedRef.current) setError(toMessage(e));
    } finally {
      if (mountedRef.current) setInstalling(null);
      refresh();
    }
  };

  const launch = async () => {
    setError(null);
    try {
      await invoke("launch_ollama");
    } catch (e) {
      if (mountedRef.current) setError(toMessage(e));
    }
    refresh();
  };

  const startPull = async () => {
    setError(null);
    setPulling({ kind: "connecting" });
    const ch = new Channel<PullStatus>();
    ch.onmessage = (msg) => {
      if (mountedRef.current) setPulling(msg);
    };
    try {
      await invoke("pull_ollama_model", {
        model: "llama3.2-vision",
        onProgress: ch,
      });
    } catch (e) {
      if (mountedRef.current) setError(toMessage(e));
    } finally {
      if (mountedRef.current) setPulling(null);
      refresh();
    }
  };

  const hasVisionModel = status.models.some(looksLikeVisionModel);
  const fullyReady = installed === true && status.running && hasVisionModel;
  const canContinue = hasCloudKey || fullyReady;

  const handleContinue = () => {
    if (fullyReady && !hasCloudKey) {
      localStorage.setItem("provider", "ollama");
      const visionModel = status.models.find(looksLikeVisionModel);
      if (visionModel) localStorage.setItem("ollama_model", visionModel);
    }
    if (canContinue) onNext();
  };

  const cta = ((): { label: string; action: () => void; busy?: boolean } | null => {
    if (installed === null) return null;
    if (installed === false) {
      return {
        label: installing ? installLabel(installing) : "Install Ollama",
        action: startInstall,
        busy: !!installing,
      };
    }
    if (!status.running) {
      return { label: "Launch Ollama", action: launch };
    }
    if (!hasVisionModel) {
      return {
        label: pulling ? pullLabel(pulling) : "Pull llama3.2-vision",
        action: startPull,
        busy: !!pulling,
      };
    }
    return null;
  })();

  const installPct = installing?.kind === "downloading" ? installing.percent : null;
  const pullPct = pulling?.kind === "pulling" ? pulling.percent : null;

  const visionRowLabel = hasVisionModel
    ? `Vision model · ${status.models.find(looksLikeVisionModel)}`
    : "Vision model not installed";

  return (
    <div className="onboarding-step-inner">
      <span className="onboarding-eyebrow">Local Ollama</span>

      <h1 className="onboarding-h1">Or run it locally.</h1>

      <p className="onboarding-subtitle">
        Ollama serves vision models from your own machine — fully private, no
        API costs. Skip if you'd rather use the cloud key.
      </p>

      <div className="onboarding-status" aria-live="polite">
        <StatusRow
          label={
            installed === null
              ? "Checking…"
              : installed
                ? isMac
                  ? "Ollama installed on this Mac"
                  : "Ollama installed on this PC"
                : "Ollama not installed"
          }
          good={installed === true}
        />
        <StatusRow
          label={status.running ? "Daemon running in the menu bar" : "Daemon not running"}
          good={status.running}
        />
        <StatusRow
          label={visionRowLabel}
          good={hasVisionModel}
        />
      </div>

      {(installing || pulling) && (
        <div className="onboarding-progress-block">
          <div className="onboarding-progress-meta">
            <span>{installing ? installLabel(installing) : pullLabel(pulling!)}</span>
            <span>{installPct ?? pullPct ?? "—"}%</span>
          </div>
          <div
            className={
              "onboarding-progress-bar" +
              (installPct == null && pullPct == null ? " indeterminate" : "")
            }
          >
            <div
              className="onboarding-progress-bar-fill"
              style={{ width: `${(installPct ?? pullPct ?? 0)}%` }}
            />
          </div>
        </div>
      )}

      {cta && !fullyReady && !installing && !pulling && (
        <div className="onboarding-cta">
          <button
            className="onboarding-btn"
            onClick={cta.action}
            disabled={cta.busy}
          >
            {cta.label}
            <span className="arrow" aria-hidden>→</span>
          </button>
        </div>
      )}

      {error && <div className="onboarding-error">{error}</div>}

      <div className="onboarding-actions">
        <button className="onboarding-link back" onClick={onBack}>
          <span className="arrow" aria-hidden>←</span>
          Back
        </button>
        <div className="onboarding-actions-right">
          {!fullyReady && hasCloudKey && (
            <button className="onboarding-link" onClick={onSkip}>
              Skip
            </button>
          )}
          <button
            className="onboarding-btn primary"
            onClick={handleContinue}
            disabled={!!installing || !!pulling || !canContinue}
          >
            {fullyReady
              ? "Continue · Ollama is ready"
              : hasCloudKey
                ? "Continue with cloud"
                : "Set up a provider to continue"}
            <span className="arrow" aria-hidden>→</span>
          </button>
        </div>
      </div>
    </div>
  );
}

function StatusRow({ label, good }: { label: string; good: boolean }) {
  return (
    <div className={"onboarding-status-row" + (good ? "" : " pending")}>
      <span className={"onboarding-status-marker" + (good ? " good" : "")} />
      <span className="onboarding-status-label">{label}</span>
      <span className={"onboarding-status-state " + (good ? "good" : "")}>
        {good ? "Ready" : "—"}
      </span>
    </div>
  );
}

function installLabel(s: InstallStatus): string {
  switch (s.kind) {
    case "downloading":
      return `Downloading Ollama`;
    case "extracting":
      return "Extracting";
    case "installing":
      return "Moving to Applications";
    case "launching":
      return "Launching";
    case "done":
      return "Done";
  }
}

function pullLabel(s: PullStatus): string {
  switch (s.kind) {
    case "connecting":
      return "Connecting to Ollama";
    case "pulling":
      return "Pulling llama3.2-vision";
    case "verifying":
      return "Verifying";
    case "done":
      return "Done";
  }
}

function toMessage(e: unknown): string {
  if (typeof e === "string") return e;
  if (e instanceof Error) return e.message;
  return String(e);
}
