import { useCallback, useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";

// Mirrors agent::MemoryEntry (camelCase serde).
type MemoryEntry = {
  id: string;
  goalSummary: string;
  appKey?: string | null;
  remember: string;
  outcome: string;
  steps: number;
  createdAtMs: number;
};

function age(createdAtMs: number): string {
  const days = Math.floor((Date.now() - createdAtMs) / 86_400_000);
  return days <= 0 ? "today" : `${days}d ago`;
}

export default function AgentMemoryList() {
  const [entries, setEntries] = useState<MemoryEntry[]>([]);
  const [error, setError] = useState<string | null>(null);

  const refresh = useCallback(() => {
    invoke<MemoryEntry[]>("list_agent_memory")
      .then(setEntries)
      .catch((err) => setError(String(err)));
  }, []);

  useEffect(() => {
    refresh();
  }, [refresh]);

  const remove = (id: string) => {
    invoke("delete_agent_memory", { id })
      .then(refresh)
      .catch((err) => setError(String(err)));
  };

  const clear = () => {
    invoke("clear_agent_memory")
      .then(refresh)
      .catch((err) => setError(String(err)));
  };

  if (entries.length === 0) {
    return (
      <p className="settings-row-help">
        Nothing saved yet. When the agent finishes a task with a takeaway
        worth keeping, it lands here (and shows a card you can dismiss or
        forget right away).
        {error && <span className="playbooks-error"> {error}</span>}
      </p>
    );
  }

  return (
    <div className="playbooks-manager">
      {entries.map((entry) => (
        <div className="settings-row playbooks-row" key={entry.id}>
          <div>
            <p className="settings-row-label">{entry.remember}</p>
            <p className="settings-row-help">
              {entry.goalSummary} — {entry.appKey || "any app"},{" "}
              {age(entry.createdAtMs)}
            </p>
          </div>
          <div className="settings-control playbooks-actions">
            <button
              className="settings-button settings-button-danger"
              onClick={() => remove(entry.id)}
            >
              Forget
            </button>
          </div>
        </div>
      ))}
      <div className="playbooks-footer">
        <button className="settings-button settings-button-danger" onClick={clear}>
          Forget all
        </button>
      </div>
      {error && <p className="playbooks-error">{error}</p>}
    </div>
  );
}
