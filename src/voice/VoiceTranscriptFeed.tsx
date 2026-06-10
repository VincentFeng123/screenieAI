import type { VoiceChip, VoiceChipState } from "./types";

/** Newest chips first; older ones stay in memory but drop out of view. */
const MAX_RENDERED = 4;

const STATE_LABELS: Partial<Record<VoiceChipState, string>> = {
  heard: "heard",
  queued: "queued",
  running: "running",
  failed: "failed",
  killed: "killed",
  blocked_on_confirmation: "waiting for confirmation",
};

export default function VoiceTranscriptFeed({ chips }: { chips: VoiceChip[] }) {
  if (chips.length === 0) return null;
  return (
    <div className="quick-tooltip-voice-feed" aria-label="Voice commands">
      {chips.slice(0, MAX_RENDERED).map((chip) => (
        <div key={chip.id} className="quick-tooltip-voice-chip" data-state={chip.state}>
          <span className="quick-tooltip-voice-chip-dot" aria-hidden />
          <span className="quick-tooltip-voice-chip-text">{chip.text}</span>
          {STATE_LABELS[chip.state] && (
            <span className="quick-tooltip-voice-chip-state">
              {STATE_LABELS[chip.state]}
            </span>
          )}
        </div>
      ))}
    </div>
  );
}
