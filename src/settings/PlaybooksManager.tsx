import { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";

// Mirrors agent::PlaybookMeta (camelCase serde).
type PlaybookMeta = {
  name: string;
  apps: string[];
  triggers: string[];
  requiresScripting: boolean;
  builtin: boolean;
  overridden: boolean;
  enabled: boolean;
};

const NEW_PLAYBOOK_TEMPLATE = `---
name: my-playbook
apps: com.apple.Safari
triggers: example goal phrase
---
Guidance the agent sees when the app and goal match. Keep it short; split
longer guidance into ## sections so only the relevant ones are included.
`;

export default function PlaybooksManager() {
  const [playbooks, setPlaybooks] = useState<PlaybookMeta[]>([]);
  const [editing, setEditing] = useState<string | null>(null); // name, or "" for new
  const [draft, setDraft] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);
  const [confirmingDelete, setConfirmingDelete] = useState<string | null>(
    null,
  );

  const refresh = useCallback(() => {
    invoke<PlaybookMeta[]>("list_playbooks")
      .then(setPlaybooks)
      .catch((err) => setError(String(err)));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const openEditor = (name: string | null) => {
    setError(null);
    if (name === null) {
      setEditing("");
      setDraft(NEW_PLAYBOOK_TEMPLATE);
      return;
    }
    invoke<string>("read_playbook", { name })
      .then((content) => {
        setEditing(name);
        setDraft(content);
      })
      .catch((err) => setError(String(err)));
  };

  const save = () => {
    setBusy(true);
    setError(null);
    invoke<PlaybookMeta>("write_playbook", { content: draft })
      .then(() => {
        setEditing(null);
        setDraft("");
        refresh();
      })
      .catch((err) => setError(String(err)))
      .finally(() => setBusy(false));
  };

  // Deleting a user file is irreversible (a shadowed built-in reappears,
  // a user playbook is gone): the first click arms, the second executes.
  const remove = (name: string) => {
    if (confirmingDelete !== name) {
      setConfirmingDelete(name);
      return;
    }
    setConfirmingDelete(null);
    setBusy(true);
    setError(null);
    invoke("delete_playbook", { name })
      .then(refresh)
      .catch((err) => setError(String(err)))
      .finally(() => setBusy(false));
  };

  const toggle = (name: string, enabled: boolean) => {
    setError(null);
    // Optimistic flip; refresh reconciles.
    setPlaybooks((current) =>
      current.map((p) => (p.name === name ? { ...p, enabled } : p)),
    );
    invoke("set_playbook_enabled", { name, enabled })
      .then(refresh)
      .catch((err) => {
        setError(String(err));
        refresh();
      });
  };

  return (
    <div className="playbooks-manager">
      {playbooks.map((playbook) => (
        <div className="settings-row playbooks-row" key={playbook.name}>
          <div>
            <p className="settings-row-label">
              {playbook.name}
              {playbook.builtin && (
                <span className="playbooks-badge">Built-in</span>
              )}
              {playbook.overridden && (
                <span className="playbooks-badge" data-edited>
                  Edited
                </span>
              )}
              {playbook.requiresScripting && (
                <span className="playbooks-badge" data-scripting>
                  Needs scripting
                </span>
              )}
            </p>
            <p className="settings-row-help">
              {playbook.apps.join(", ") || "any app"}
              {playbook.triggers.length > 0 &&
                ` — ${playbook.triggers.slice(0, 4).join(", ")}${
                  playbook.triggers.length > 4 ? ", …" : ""
                }`}
            </p>
          </div>
          <div className="settings-control playbooks-actions">
            <button
              className="settings-button"
              onClick={() => openEditor(playbook.name)}
            >
              {playbook.builtin ? "View / edit" : "Edit"}
            </button>
            {!playbook.builtin && (
              <button
                className="settings-button settings-button-danger"
                disabled={busy}
                onClick={() => remove(playbook.name)}
                onBlur={() => setConfirmingDelete(null)}
              >
                {confirmingDelete === playbook.name
                  ? "Click to confirm"
                  : playbook.overridden
                    ? "Revert"
                    : "Delete"}
              </button>
            )}
            <button
              className="settings-toggle"
              data-active={playbook.enabled}
              role="switch"
              aria-checked={playbook.enabled}
              aria-label={`Enable playbook ${playbook.name}`}
              onClick={() => toggle(playbook.name, !playbook.enabled)}
            />
          </div>
        </div>
      ))}

      {editing !== null ? (
        <div className="playbooks-editor">
          <textarea
            className="playbooks-editor-textarea"
            value={draft}
            spellCheck={false}
            rows={14}
            onChange={(event) => setDraft(event.target.value)}
            aria-label="Playbook markdown"
          />
          <div className="playbooks-editor-footer">
            <span className="settings-muted">
              {draft.length.toLocaleString()} chars — sections beyond ~2,500
              chars are trimmed to the goal-relevant ones
            </span>
            <div className="playbooks-actions">
              <button
                className="settings-button"
                onClick={() => {
                  setEditing(null);
                  setError(null);
                }}
              >
                Cancel
              </button>
              <button
                className="settings-button settings-button-primary"
                disabled={busy || draft.trim().length === 0}
                onClick={save}
              >
                Save
              </button>
            </div>
          </div>
        </div>
      ) : (
        <div className="playbooks-footer">
          <button className="settings-button" onClick={() => openEditor(null)}>
            New playbook
          </button>
        </div>
      )}

      {error && <p className="playbooks-error">{error}</p>}
    </div>
  );
}
