import { useEffect, useRef, useState } from "react";
import { Channel, invoke } from "@tauri-apps/api/core";
import CopyCommand from "./CopyCommand";
import { VISION_MODEL_PULL } from "./constants";
import { hint, smallBtn, primaryBtn, progressTrack, progressFill } from "./styles";

type PullStatus =
  | { kind: "connecting" }
  | { kind: "pulling"; percent: number; phase: string }
  | { kind: "verifying" }
  | { kind: "done" };

export default function NeedsVisionModel({ onCheck }: { onCheck: () => void }) {
  const [pulling, setPulling] = useState<PullStatus | null>(null);
  const [pullError, setPullError] = useState<string | null>(null);
  const mountedRef = useRef(true);
  useEffect(() => {
    mountedRef.current = true;
    return () => {
      mountedRef.current = false;
    };
  }, []);

  const startPull = async () => {
    setPullError(null);
    setPulling({ kind: "connecting" });
    const channel = new Channel<PullStatus>();
    channel.onmessage = (msg) => {
      if (mountedRef.current) setPulling(msg);
    };
    try {
      await invoke("pull_ollama_model", {
        model: "llama3.2-vision",
        onProgress: channel,
      });
    } catch (e) {
      const msg = typeof e === "string" ? e : (e as Error).message ?? String(e);
      if (mountedRef.current) setPullError(msg);
    } finally {
      if (mountedRef.current) setPulling(null);
      onCheck();
    }
  };

  const pullLabel = (s: PullStatus | null): string => {
    if (!s) return "";
    switch (s.kind) {
      case "connecting":
        return "Connecting to Ollama…";
      case "pulling":
        return `Pulling llama3.2-vision… ${s.percent}%`;
      case "verifying":
        return "Verifying…";
      case "done":
        return "Done";
    }
  };
  const pullPct = pulling?.kind === "pulling" ? pulling.percent : null;

  return (
    <div style={{ marginTop: 10 }}>
      <p style={{ ...hint, marginTop: 0, marginBottom: 8 }}>
        No vision model installed yet. Ollama needs one to read screenshots —
        we can pull <code>llama3.2-vision</code> for you (~7 GB).
      </p>
      <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
        <button
          onClick={startPull}
          disabled={pulling !== null}
          style={{
            ...primaryBtn,
            opacity: pulling !== null ? 0.7 : 1,
            cursor: pulling !== null ? "not-allowed" : "pointer",
          }}
        >
          {pulling ? "Pulling…" : "Pull llama3.2-vision"}
        </button>
        <button onClick={onCheck} style={smallBtn} disabled={pulling !== null}>
          Recheck
        </button>
      </div>
      {pulling && (
        <div style={{ marginTop: 10 }}>
          <div style={{ fontSize: 11.5, opacity: 0.7, marginBottom: 4 }}>
            {pullLabel(pulling)}
          </div>
          <div style={progressTrack}>
            <div
              style={{
                ...progressFill,
                width:
                  pullPct != null
                    ? `${pullPct}%`
                    : pulling.kind === "verifying"
                    ? "97%"
                    : pulling.kind === "done"
                    ? "100%"
                    : "8%",
              }}
            />
          </div>
        </div>
      )}
      {pullError && !pulling && (
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
          Pull failed: {pullError}
        </div>
      )}
      <details style={{ marginTop: 12 }}>
        <summary style={{ ...hint, cursor: "pointer", marginTop: 0 }}>
          Or run it yourself in a terminal
        </summary>
        <div style={{ marginTop: 8 }}>
          <CopyCommand command={VISION_MODEL_PULL} />
        </div>
      </details>
    </div>
  );
}
