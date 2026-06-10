import { Download, Loader2, Mic, MicOff } from "lucide-react";

export type VoiceMicState =
  | "idle"
  | "listening"
  | "transcribing"
  | "denied"
  | "model-missing";

export default function VoiceMicButton({
  state,
  onToggle,
}: {
  state: VoiceMicState;
  onToggle: () => void;
}) {
  const label =
    state === "listening" || state === "transcribing"
      ? "Stop listening"
      : state === "denied"
        ? "Microphone access denied"
        : state === "model-missing"
          ? "Download the speech model to enable voice"
          : "Start voice commands";
  return (
    <button
      type="button"
      className="quick-tooltip-voice-btn"
      data-state={state}
      onClick={onToggle}
      aria-label={label}
      title={label}
    >
      {state === "denied" ? (
        <MicOff size={16} strokeWidth={1.9} aria-hidden />
      ) : state === "transcribing" ? (
        <Loader2 size={16} strokeWidth={1.9} aria-hidden />
      ) : state === "model-missing" ? (
        <Download size={16} strokeWidth={1.9} aria-hidden />
      ) : (
        <Mic size={16} strokeWidth={1.9} aria-hidden />
      )}
    </button>
  );
}
