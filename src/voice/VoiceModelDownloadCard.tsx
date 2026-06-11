import type { VoiceModelName } from "./types";

/** Approximate download sizes, mirroring MODEL_SPECS in voice/model.rs. */
const MODEL_SIZES: Record<VoiceModelName, string> = {
  "tiny.en": "75 MB",
  "base.en": "142 MB",
  "small.en": "466 MB",
  "large-v3-turbo": "574 MB",
};

/**
 * First-use prompt shown inside the agent card while the whisper model is
 * missing. Progress comes from `voice:model_download` events; the mic
 * enables (no auto-start) once the download completes.
 */
export default function VoiceModelDownloadCard({
  pct,
  model,
  onDownload,
}: {
  pct: number | null;
  model: VoiceModelName | null;
  onDownload: () => void;
}) {
  const downloading = pct !== null;
  const size = model ? MODEL_SIZES[model] : null;
  return (
    <div className="quick-tooltip-voice-download">
      {downloading ? (
        <>
          <div className="quick-tooltip-voice-download-text">
            Downloading speech model… {pct}%
          </div>
          <div className="quick-tooltip-voice-progress" aria-hidden>
            <div
              className="quick-tooltip-voice-progress-fill"
              style={{ width: `${pct}%` }}
            />
          </div>
        </>
      ) : (
        <>
          <div className="quick-tooltip-voice-download-text">
            Voice needs a one-time speech model download
            {size ? ` (~${size})` : ""}. Transcription stays on this Mac.
          </div>
          <button
            type="button"
            className="quick-tooltip-confirm-btn primary"
            onClick={onDownload}
          >
            Download model
          </button>
        </>
      )}
    </div>
  );
}
