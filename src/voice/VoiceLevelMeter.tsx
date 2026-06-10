import type { RefObject } from "react";

/**
 * Thin RMS bar fed by `useVoiceSession`. The fill element updates via
 * ref-driven DOM writes (`transform: scaleX`), NOT React state — 10 Hz
 * through useState would re-render the whole tooltip and re-trigger the
 * frost-region MutationObserver. Never add useState for rms here.
 */
export default function VoiceLevelMeter({
  fillRef,
}: {
  fillRef: RefObject<HTMLDivElement | null>;
}) {
  return (
    <div className="quick-tooltip-voice-meter" aria-hidden>
      <div ref={fillRef} className="quick-tooltip-voice-meter-fill" />
    </div>
  );
}
