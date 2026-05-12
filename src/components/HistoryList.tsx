import {
  useEffect,
  useRef,
  useState,
  type CSSProperties,
  type ReactNode,
} from "react";
import { Archive } from "lucide-react";
import {
  deleteHistoryEntry,
  deriveHistoryTitle,
  listHistory,
  loadHistoryImage,
  type HistoryEntry,
} from "../lib/history";

export type HistoryListProps = {
  /// Optional row-click handler. Receives the entry plus the full PNG bytes
  /// (base64). When omitted, rows aren't clickable — used by the Settings
  /// panel where there's no surface to "open" an entry into.
  onOpen?: (entry: HistoryEntry, fullPng: string) => void;
  /// Bump this number to force a re-fetch of the underlying list. The
  /// SettingsPanel uses this after Clear-All so the just-emptied list
  /// re-renders without a manual refresh.
  reloadKey?: number;
  /// Replaces the default "No captures yet" message. Renders inside the
  /// same scroll container so layout stays consistent.
  emptyState?: ReactNode;
  /// Override the container style. Defaults to a flex column tuned for the
  /// chat-panel embeds (fills remaining height, scrolls). The Settings
  /// panel hands in a no-flex variant since its outer card scrolls instead.
  containerStyle?: CSSProperties;
};

const DEFAULT_CONTAINER_STYLE: CSSProperties = {
  flex: 1,
  minHeight: 0,
  overflow: "auto",
  marginRight: 6,
  marginBottom: 8,
  padding: "10px 6px 6px 10px",
  display: "flex",
  flexDirection: "column",
  gap: 2,
  position: "relative",
  zIndex: 1,
};

export default function HistoryList({
  onOpen,
  reloadKey = 0,
  emptyState,
  containerStyle,
}: HistoryListProps) {
  const [entries, setEntries] = useState<HistoryEntry[] | null>(null);
  const [statusError, setStatusError] = useState<string | null>(null);
  const [retryKey, setRetryKey] = useState(0);
  // Two-step delete: archive arms an entry, "Confirm" commits it. Only one
  // row can be pending at a time. Cleared by: clicking Confirm, clicking the
  // archive on a different row, the 4s safety timer below, or any
  // pointerdown outside the active Confirm button (the window-level
  // listener two effects down). Once cleared, the regular archive icon is
  // immediately available on hover — no latch, no leftover red ring.
  const [pendingDeleteId, setPendingDeleteId] = useState<string | null>(null);
  // Mirror of pendingDeleteId so the click-outside listener (which only
  // re-binds when pending changes) and the timer can read the current value
  // without stale closures.
  const pendingRef = useRef<string | null>(null);
  pendingRef.current = pendingDeleteId;

  useEffect(() => {
    let cancelled = false;
    setEntries(null);
    setStatusError(null);
    listHistory()
      .then((list) => {
        if (!cancelled) setEntries(list);
      })
      .catch((e) => {
        if (cancelled) return;
        setEntries([]);
        setStatusError(`Couldn't load history: ${toMessage(e)}`);
      });
    return () => {
      cancelled = true;
    };
  }, [reloadKey, retryKey]);

  // Auto-disarm a stranded "Confirm" state after 4s. Belt-and-braces with
  // the click-outside dismissal — covers the case where the user walks away.
  useEffect(() => {
    if (!pendingDeleteId) return;
    const t = setTimeout(() => {
      setPendingDeleteId(null);
    }, 4000);
    return () => clearTimeout(t);
  }, [pendingDeleteId]);

  // Click-anywhere-else dismissal. Capture-phase listeners on `window` (not
  // `document`) so we catch the event before any React handler. `window` +
  // capture is the most reliable surface across the chat panel and detached
  // chat window — listening only on `document` was missing some clicks in
  // those contexts. We also subscribe to `pointerdown` in addition to
  // `mousedown` so trackpad / pen / synthetic click sources all dismiss.
  // The check uses `composedPath()` so a click that originated inside the
  // pending Confirm button (even through a nested span / icon) is correctly
  // recognized and skips the dismissal — letting the second click commit
  // the delete.
  useEffect(() => {
    if (!pendingDeleteId) return;
    const onDown = (e: Event) => {
      const ev = e as Event & { composedPath?: () => EventTarget[] };
      let path: EventTarget[];
      if (typeof ev.composedPath === "function") {
        path = ev.composedPath();
      } else if (e.target) {
        path = [e.target];
      } else {
        path = [];
      }
      const hitPending = path.some((node) => {
        if (!(node instanceof Element)) return false;
        return node.matches?.(
          '.screenie-history-archive[data-pending="true"]',
        );
      });
      if (hitPending) return;
      setPendingDeleteId(null);
    };
    window.addEventListener("mousedown", onDown, true);
    window.addEventListener("pointerdown", onDown, true);
    return () => {
      window.removeEventListener("mousedown", onDown, true);
      window.removeEventListener("pointerdown", onDown, true);
    };
  }, [pendingDeleteId]);

  const remove = async (id: string) => {
    try {
      setStatusError(null);
      await deleteHistoryEntry(id);
      setEntries((list) => (list ?? []).filter((e) => e.id !== id));
      setPendingDeleteId((current) => (current === id ? null : current));
    } catch (e) {
      console.error("delete_history_entry failed:", e);
      setStatusError(`Couldn't delete that entry: ${toMessage(e)}`);
    }
  };

  const open = async (entry: HistoryEntry) => {
    if (!onOpen) return;
    try {
      setStatusError(null);
      const full = await loadHistoryImage(entry.id);
      onOpen(entry, full);
    } catch (e) {
      console.error("load_history_image failed:", e);
      setStatusError(`Couldn't open that capture: ${toMessage(e)}`);
    }
  };

  return (
    <div
      style={containerStyle ?? DEFAULT_CONTAINER_STYLE}
      className="screenie-history-list"
    >
      {entries === null && (
        <div className="screenie-history-status">Loading…</div>
      )}
      {entries !== null && statusError && (
        <div className="screenie-history-status" role="alert">
          {statusError}
          <button
            type="button"
            className="screenie-action"
            style={{ marginTop: 10, fontSize: 12 }}
            onClick={() => setRetryKey((key) => key + 1)}
          >
            Retry
          </button>
        </div>
      )}
      {entries !== null && !statusError && entries.length === 0 &&
        (emptyState ?? (
          <div className="screenie-history-status">
            No captures yet — your history will show up here.
          </div>
        ))}
      {!statusError && entries?.map((entry) => {
        const title = deriveHistoryTitle(entry);
        const pending = pendingDeleteId === entry.id;
        const interactive = !!onOpen;
        return (
          <div
            key={entry.id}
            className="screenie-history-row"
            data-interactive={interactive}
            onClick={interactive ? () => open(entry) : undefined}
            onMouseDown={(e) => e.stopPropagation()}
          >
            <span className="screenie-history-title" title={title}>
              {title}
            </span>
            <button
              type="button"
              className="screenie-history-archive"
              data-pending={pending}
              onClick={(e) => {
                e.stopPropagation();
                if (pending) {
                  void remove(entry.id);
                } else {
                  setPendingDeleteId(entry.id);
                }
              }}
              aria-label={pending ? "Confirm delete" : "Archive history entry"}
            >
              {pending ? (
                "Confirm"
              ) : (
                <Archive size={13} strokeWidth={1.85} aria-hidden />
              )}
            </button>
          </div>
        );
      })}
    </div>
  );
}

function toMessage(e: unknown): string {
  if (typeof e === "string") return e;
  if (e instanceof Error) return e.message;
  return String(e);
}
