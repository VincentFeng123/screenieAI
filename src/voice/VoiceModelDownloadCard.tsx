/**
 * First-use prompt shown inside the agent card while the whisper model is
 * missing. Progress comes from `voice:model_download` events; the mic
 * enables (no auto-start) once the download completes.
 */
export default function VoiceModelDownloadCard({
  pct,
  onDownload,
}: {
  pct: number | null;
  onDownload: () => void;
}) {
  const downloading = pct !== null;
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
            Voice needs a one-time speech model download (~142&nbsp;MB).
            Transcription stays on this Mac.
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
