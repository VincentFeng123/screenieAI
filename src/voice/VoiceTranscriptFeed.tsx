import { useEffect, useRef, useState } from "react";
import { Eraser, Pause, Play, X } from "lucide-react";
import type { VoiceChip, VoiceChipState } from "./types";

/** Newest chips first; older ones stay in memory but drop out of view. */
const MAX_RENDERED = 4;

const STATE_LABELS: Partial<Record<VoiceChipState, string>> = {
  heard: "heard",
  queued: "queued",
  running: "running",
  failed: "failed",
  killed: "killed",
  removed: "removed",
  blocked_on_confirmation: "waiting for confirmation",
};

type Props = {
  chips: VoiceChip[];
  paused: boolean;
  onSetPaused: (paused: boolean) => void;
  onClear: () => void;
  onRemove: (id: string) => void;
  /** Resolves true when the edit landed before the task started running. */
  onEdit: (id: string, text: string) => Promise<boolean>;
  /** Claim/release tooltip keyboard routing around inline editing. */
  onEditFocus: () => void;
  onEditBlur: () => void;
};

export default function VoiceTranscriptFeed({
  chips,
  paused,
  onSetPaused,
  onClear,
  onRemove,
  onEdit,
  onEditFocus,
  onEditBlur,
}: Props) {
  const [editingId, setEditingId] = useState<string | null>(null);
  const [draft, setDraft] = useState("");
  const editInputRef = useRef<HTMLInputElement | null>(null);

  // Unmounting a focused input does NOT fire blur, so every edit exit path
  // must release the keyboard claim explicitly.
  const endEdit = () => {
    setEditingId(null);
    onEditBlur();
  };

  // A queued chip can start running (or get killed) mid-edit; drop the
  // editor rather than submitting against a task that already popped.
  const editingChip = chips.find((c) => c.id === editingId);
  const editingState = editingChip?.state;
  useEffect(() => {
    if (editingId && editingState !== "queued") {
      setEditingId(null);
      onEditBlur();
    }
  }, [editingId, editingState, onEditBlur]);

  useEffect(() => {
    if (editingId) {
      editInputRef.current?.focus();
      editInputRef.current?.select();
    }
  }, [editingId]);

  if (chips.length === 0) return null;

  const hasQueued = chips.some((chip) => chip.state === "queued");

  const commitEdit = async (chip: VoiceChip) => {
    const text = draft.trim();
    if (!text || text === chip.text) {
      endEdit();
      return;
    }
    if (await onEdit(chip.id, text)) {
      endEdit();
    }
  };

  return (
    <div className="quick-tooltip-voice-feed" aria-label="Voice tasks">
      <div className="quick-tooltip-voice-feed-header">
        <span className="quick-tooltip-voice-feed-title">Tasks</span>
        {paused && (
          <span className="quick-tooltip-voice-feed-paused-badge">Paused</span>
        )}
        <button
          type="button"
          className="quick-tooltip-voice-feed-btn"
          onClick={() => onSetPaused(!paused)}
          aria-label={paused ? "Resume queued tasks" : "Pause queued tasks"}
          title={paused ? "Resume queued tasks" : "Pause queued tasks"}
        >
          {paused ? (
            <Play size={12} strokeWidth={2} aria-hidden />
          ) : (
            <Pause size={12} strokeWidth={2} aria-hidden />
          )}
        </button>
        <button
          type="button"
          className="quick-tooltip-voice-feed-btn"
          onClick={onClear}
          disabled={!hasQueued}
          aria-label="Clear queued tasks"
          title="Clear queued tasks"
        >
          <Eraser size={12} strokeWidth={2} aria-hidden />
        </button>
      </div>
      {chips.slice(0, MAX_RENDERED).map((chip) => {
        const queued = chip.state === "queued";
        const editing = editingId === chip.id;
        return (
          <div
            key={chip.id}
            className="quick-tooltip-voice-chip"
            data-state={chip.state}
          >
            <span className="quick-tooltip-voice-chip-dot" aria-hidden />
            {editing ? (
              <input
                ref={editInputRef}
                className="quick-tooltip-voice-chip-edit"
                value={draft}
                onChange={(e) => setDraft(e.target.value)}
                onFocus={onEditFocus}
                onBlur={endEdit}
                onKeyDown={(e) => {
                  if (e.key === "Enter") {
                    e.preventDefault();
                    void commitEdit(chip);
                  } else if (e.key === "Escape") {
                    e.preventDefault();
                    e.stopPropagation();
                    endEdit();
                  }
                }}
                aria-label="Edit queued task"
              />
            ) : (
              <span
                className="quick-tooltip-voice-chip-text"
                data-editable={queued || undefined}
                title={queued ? "Click to edit" : undefined}
                onClick={() => {
                  if (!queued) return;
                  setDraft(chip.text);
                  setEditingId(chip.id);
                }}
              >
                {chip.text}
              </span>
            )}
            {!editing && STATE_LABELS[chip.state] && (
              <span className="quick-tooltip-voice-chip-state">
                {STATE_LABELS[chip.state]}
              </span>
            )}
            {queued && !editing && (
              <button
                type="button"
                className="quick-tooltip-voice-chip-remove"
                onClick={() => onRemove(chip.id)}
                aria-label="Remove queued task"
                title="Remove queued task"
              >
                <X size={11} strokeWidth={2.2} aria-hidden />
              </button>
            )}
          </div>
        );
      })}
    </div>
  );
}
