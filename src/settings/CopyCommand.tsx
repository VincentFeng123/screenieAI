import { useState } from "react";
import { smallBtn } from "./styles";

export default function CopyCommand({ command }: { command: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div
      style={{
        display: "flex",
        gap: 6,
        alignItems: "stretch",
      }}
    >
      <code
        style={{
          flex: 1,
          padding: "8px 10px",
          borderRadius: 8,
          background: "var(--ink)",
          color: "var(--bg)",
          fontSize: 12.5,
          fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, monospace",
          overflow: "auto",
          whiteSpace: "nowrap",
        }}
      >
        {command}
      </code>
      <button
        onClick={() => {
          navigator.clipboard.writeText(command).then(
            () => {
              setCopied(true);
              setTimeout(() => setCopied(false), 1400);
            },
            () => {
              /* clipboard unavailable */
            },
          );
        }}
        style={{
          ...smallBtn,
          minWidth: 70,
          color: copied ? "var(--status-ok-text)" : (smallBtn.color as string),
          borderColor: copied
            ? "var(--status-ok-text)"
            : (smallBtn.border as string),
        }}
      >
        {copied ? "Copied" : "Copy"}
      </button>
    </div>
  );
}
