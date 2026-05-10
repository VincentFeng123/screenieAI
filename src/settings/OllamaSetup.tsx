import { useEffect, useRef, useState } from "react";
import { Channel, invoke } from "@tauri-apps/api/core";
import { openUrl } from "@tauri-apps/plugin-opener";
import CopyCommand from "./CopyCommand";
import { VISION_MODEL_PULL } from "./constants";
import {
  hint,
  smallBtn,
  primaryBtn,
  stepRow,
  stepIndex,
  stepTitle,
  stepBody,
  progressTrack,
  progressFill,
} from "./styles";

type InstallStatus =
  | { kind: "downloading"; percent: number }
  | { kind: "extracting" }
  | { kind: "installing" }
  | { kind: "launching" }
  | { kind: "done" };

export default function OllamaSetup({
  onCheck,
  checking,
}: {
  onCheck: () => void;
  checking: boolean;
}) {
  const isMac =
    typeof navigator !== "undefined" && navigator.userAgent.includes("Mac");
  const [installing, setInstalling] = useState<InstallStatus | null>(null);
  const [installError, setInstallError] = useState<string | null>(null);

  // Guard against state writes after unmount. The component unmounts the
  // moment `ollama.running` flips true (auto-poll detected the daemon),
  // which can happen mid-install when an "already installed" early-return
  // path fires Launching → Done in milliseconds.
  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const startInstall = async () => {
    setInstallError(null);
    setInstalling({ kind: "downloading", percent: 0 });
    const channel = new Channel<InstallStatus>();
    channel.onmessage = (msg) => {
      if (mountedRef.current) setInstalling(msg);
    };
    try {
      await invoke("install_ollama", { onProgress: channel });
    } catch (e) {
      const msg = typeof e === "string" ? e : (e as Error).message ?? String(e);
      if (mountedRef.current) setInstallError(msg);
    } finally {
      if (mountedRef.current) setInstalling(null);
      onCheck();
    }
  };

  const installLabel = (s: InstallStatus | null): string => {
    if (!s) return "";
    switch (s.kind) {
      case "downloading":
        return `Downloading Ollama… ${s.percent}%`;
      case "extracting":
        return "Extracting…";
      case "installing":
        return "Moving to Applications…";
      case "launching":
        return "Launching Ollama…";
      case "done":
        return "Done";
    }
  };
  const downloadPct =
    installing?.kind === "downloading" ? installing.percent : null;

  return (
    <div style={{ display: "flex", flexDirection: "column", gap: 12 }}>
      <p style={{ ...hint, marginTop: 0 }}>
        Ollama runs vision models locally on your machine — fully private, no
        API keys, no costs. Two steps to set it up:
      </p>
      <div style={stepRow}>
        <div style={stepIndex}>1</div>
        <div style={{ flex: 1 }}>
          <div style={stepTitle}>Install Ollama</div>
          <div style={stepBody}>
            {isMac
              ? "We can install it for you (~700 MB download). macOS will ask for your password once so Ollama can finish its own setup."
              : "Download and run the installer for your platform."}
          </div>
          {isMac ? (
            <div style={{ marginTop: 8 }}>
              <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
                <button
                  onClick={startInstall}
                  disabled={installing !== null}
                  style={{
                    ...primaryBtn,
                    opacity: installing !== null ? 0.7 : 1,
                    cursor: installing !== null ? "not-allowed" : "pointer",
                  }}
                >
                  {installing ? "Installing…" : "Install for me"}
                </button>
                <button
                  onClick={() =>
                    openUrl("https://ollama.com/download").catch((e) =>
                      console.error("openUrl failed:", e),
                    )
                  }
                  style={smallBtn}
                  disabled={installing !== null}
                >
                  Manual install ↗
                </button>
              </div>
              {installing && (
                <div style={{ marginTop: 10 }}>
                  <div
                    style={{
                      fontSize: 11.5,
                      opacity: 0.7,
                      marginBottom: 4,
                    }}
                  >
                    {installLabel(installing)}
                  </div>
                  <div style={progressTrack}>
                    <div
                      style={{
                        ...progressFill,
                        width:
                          downloadPct != null
                            ? `${downloadPct}%`
                            : installing.kind === "extracting"
                            ? "65%"
                            : installing.kind === "installing"
                            ? "85%"
                            : installing.kind === "launching"
                            ? "97%"
                            : "100%",
                      }}
                    />
                  </div>
                </div>
              )}
              {installError && !installing && (
                <div
                  style={{
                    marginTop: 10,
                    padding: "8px 10px",
                    fontSize: 11.5,
                    background: "var(--error-bg)",
                    color: "var(--error-text)",
                    borderRadius: 8,
                  }}
                >
                  Auto-install failed: {installError}. Try the manual link.
                </div>
              )}
            </div>
          ) : (
            <button
              style={{ ...primaryBtn, marginTop: 8 }}
              onClick={() =>
                openUrl("https://ollama.com/download").catch((e) =>
                  console.error("openUrl failed:", e),
                )
              }
            >
              Open ollama.com/download ↗
            </button>
          )}
        </div>
      </div>
      <div style={stepRow}>
        <div style={stepIndex}>2</div>
        <div style={{ flex: 1 }}>
          <div style={stepTitle}>Pull a vision model</div>
          <div style={stepBody}>
            Run this once in a terminal — about 7 GB to download.
          </div>
          <CopyCommand command={VISION_MODEL_PULL} />
        </div>
      </div>
      <div
        style={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          gap: 8,
          marginTop: 4,
          fontSize: 11.5,
          opacity: 0.6,
        }}
      >
        <span>{checking ? "Checking…" : "Auto-detecting every few seconds"}</span>
        <button onClick={onCheck} style={smallBtn}>
          Check now
        </button>
      </div>
    </div>
  );
}
