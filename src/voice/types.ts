// TypeScript mirrors of the Rust voice event/command payloads
// (src-tauri/src/voice/). Event names are the `voice:*` contract.

export type VoiceBackendStatus = "idle" | "listening" | "transcribing";

export type VoiceModelName = "tiny.en" | "base.en" | "small.en";

export type VoiceConfig = {
  silenceMs: number;
  model: VoiceModelName;
  autoStopS: number;
};

export type VoiceStatusPayload = {
  status: VoiceBackendStatus;
  modelPresent: boolean;
  micPermission: "authorized" | "denied" | "undetermined" | "unknown";
  config: VoiceConfig;
};

export type VoiceLevelPayload = { rms: number };

export type VoiceUtterancePayload = { id: string; text: string };

export type VoiceChipState =
  | "heard"
  | "queued"
  | "running"
  | "done"
  | "failed"
  | "blocked_on_confirmation"
  | "killed";

export type VoiceTaskPayload = { id: string; state: VoiceChipState };

export type VoiceModelDownloadPayload = { pct: number; model: VoiceModelName };

export type VoiceErrorPayload = {
  code:
    | "mic_denied"
    | "no_device"
    | "model_missing"
    | "audio"
    | "stt"
    | "download"
    | "queue_full"
    | "idle_timeout"
    | "agent";
  message: string;
  /** Present on queue_full so the rejected utterance's chip can be marked. */
  id?: string;
};

export type VoiceChip = {
  id: string;
  text: string;
  state: VoiceChipState;
};
