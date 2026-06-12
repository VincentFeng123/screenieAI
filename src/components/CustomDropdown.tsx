import {
  useCallback,
  useEffect,
  useLayoutEffect,
  useRef,
  useState,
  type ReactNode,
} from "react";
import { createPortal } from "react-dom";
import { Check, ChevronDown, Search } from "lucide-react";
import "./CustomDropdown.css";

export type CustomDropdownOption = {
  value: string;
  label: string;
  detail?: string;
};

const SEARCH_THRESHOLD = 8;
const MENU_BOTTOM_MARGIN = 24;
const MENU_GAP = 6;
const MENU_MIN_HEIGHT = 140;
/// Minimum gap between the portaled menu and the viewport edges. The webview
/// hard-clips anything outside the native window (the quick tooltip window is
/// barely wider than its trigger row), so the menu must stay inside it.
const VIEWPORT_EDGE_MARGIN = 8;
/// Fraction of the viewport the menu can occupy at most. 70% leaves the user
/// some context (the trigger + a glimpse of the surrounding UI) while still
/// letting long lists like the OpenAI model picker stretch on tall windows.
const MENU_MAX_VIEWPORT_FRACTION = 0.7;
/// Hard ceiling — even on a 4K display we don't need a 2700px-tall dropdown.
const MENU_HARD_MAX_HEIGHT = 720;

function viewportMenuMax(): number {
  const h = typeof window !== "undefined" ? window.innerHeight : 600;
  return Math.min(
    MENU_HARD_MAX_HEIGHT,
    Math.max(MENU_MIN_HEIGHT, Math.floor(h * MENU_MAX_VIEWPORT_FRACTION)),
  );
}

function optionParts(option: CustomDropdownOption | undefined, fallback: string) {
  const label = option?.label ?? fallback;
  if (option?.detail) return { main: label, detail: option.detail };

  const separator = " — ";
  const idx = label.indexOf(separator);
  if (idx === -1) return { main: label };
  return {
    main: label.slice(0, idx),
    detail: label.slice(idx + separator.length),
  };
}

function matchesQuery(option: CustomDropdownOption, query: string): boolean {
  const q = query.trim().toLowerCase();
  if (!q) return true;
  return (
    option.label.toLowerCase().includes(q) ||
    option.value.toLowerCase().includes(q) ||
    (option.detail?.toLowerCase().includes(q) ?? false)
  );
}

export default function CustomDropdown({
  value,
  options,
  onChange,
  ariaLabel,
  variant = "panel",
  disabled = false,
  triggerLabel,
  placement: placementPreference = "auto",
  onOpenChange,
}: {
  value: string;
  options: CustomDropdownOption[];
  onChange: (value: string) => void;
  ariaLabel: string;
  variant?: "panel" | "ghost";
  disabled?: boolean;
  triggerLabel?: ReactNode;
  placement?: "auto" | "below" | "above";
  onOpenChange?: (open: boolean) => void;
}) {
  const [open, setOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState("");
  const [menuMaxHeight, setMenuMaxHeight] = useState<number>(() => viewportMenuMax());
  // Both variants flip between below/above based on available space.
  // The menu is portaled to <body> so flipping never gets clipped by an
  // ancestor's `overflow: hidden` (e.g., the chat panel's rounded rect).
  const [resolvedPlacement, setResolvedPlacement] = useState<"below" | "above">("below");
  // Triggered position for the portaled menu. Updated on open + on
  // window resize / scroll while open so the menu stays anchored to the
  // (potentially-moved) trigger. `maxWidth` is set when the trigger lives
  // inside an `.screenie-chat-panel` — caps the menu at 70% of that panel
  // so it doesn't sprawl across the chat surface.
  const [menuPos, setMenuPos] = useState<{
    left: number;
    top: number;
    width: number;
    maxWidth?: number;
  } | null>(null);
  const rootRef = useRef<HTMLDivElement>(null);
  const triggerRef = useRef<HTMLButtonElement>(null);
  const menuRef = useRef<HTMLDivElement>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const selected = options.find((option) => option.value === value) ?? options[0];
  const selectedParts = optionParts(selected, value);
  const showSearch = options.length > SEARCH_THRESHOLD;
  const filteredOptions = options.filter((option) => matchesQuery(option, searchQuery));

  const updateOpen = useCallback((next: boolean | ((current: boolean) => boolean)) => {
    setOpen((current) => (typeof next === "function" ? next(current) : next));
  }, []);

  useEffect(() => {
    onOpenChange?.(open);
  }, [onOpenChange, open]);

  useEffect(() => {
    if (!open) return;

    const onPointerDown = (event: PointerEvent) => {
      const root = rootRef.current;
      const menu = menuRef.current;
      const target = event.target as Node;
      // The menu is portaled to <body> so it isn't a descendant of `root`.
      // Treat clicks inside either as inside the dropdown.
      if (root && root.contains(target)) return;
      if (menu && menu.contains(target)) return;
      updateOpen(false);
    };
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") updateOpen(false);
    };

    window.addEventListener("pointerdown", onPointerDown);
    window.addEventListener("keydown", onKeyDown);
    return () => {
      window.removeEventListener("pointerdown", onPointerDown);
      window.removeEventListener("keydown", onKeyDown);
    };
  }, [open, updateOpen]);

  useEffect(() => {
    if (!open) setSearchQuery("");
  }, [open]);

  // Recompute placement + position whenever the menu opens or the trigger
  // shifts. The menu is portaled to <body> so it can't be clipped by an
  // ancestor's `overflow: hidden` (e.g., the overlay's chat panel). Position
  // is recomputed on resize/scroll while open so it tracks the trigger.
  useLayoutEffect(() => {
    if (!open) return;
    const compute = () => {
      const trigger = triggerRef.current;
      if (!trigger) return;
      const rect = trigger.getBoundingClientRect();
      const viewportH = window.innerHeight;
      const availableBelow = viewportH - rect.bottom - MENU_BOTTOM_MARGIN - MENU_GAP;
      const availableAbove = rect.top - MENU_BOTTOM_MARGIN - MENU_GAP;
      // Flip up when there's more room above. The menu is portaled to
      // <body>, so flipping up never gets clipped by an ancestor's
      // `overflow: hidden`. This matters for the detached chat window
      // where the dropdown sits in the BOTTOM textfield area — opening
      // below would run straight off the bottom of the window.
      const useAbove =
        placementPreference === "above"
          ? true
          : placementPreference === "below"
            ? false
            : availableAbove > availableBelow;
      const available = useAbove ? availableAbove : availableBelow;
      setResolvedPlacement(useAbove ? "above" : "below");
      // What physically fits between the trigger and the window edge on the
      // chosen side: the webview hard-clips at the window bounds, so the
      // MENU_MIN_HEIGHT floor must never push past this.
      const sideCap = Math.max(
        60,
        Math.floor(
          (useAbove ? rect.top : viewportH - rect.bottom) -
            MENU_GAP -
            VIEWPORT_EDGE_MARGIN,
        ),
      );
      // Cap at 70% of viewport height (or the available side-room, whichever
      // is smaller). The min keeps the menu usable when the trigger sits in
      // a tight slot.
      setMenuMaxHeight(
        Math.min(
          viewportMenuMax(),
          Math.max(MENU_MIN_HEIGHT, Math.floor(available)),
          sideCap,
        ),
      );
      // Width cap: when the trigger is inside the overlay's chat panel, the
      // menu is bounded to 70% of that panel's width so it doesn't sprawl
      // across the entire chat surface. Outside a chat panel (e.g., the
      // settings UI) we let the menu match the trigger's width.
      const chatPanel = trigger.closest(".screenie-chat-panel") as HTMLElement | null;
      let maxWidth: number | undefined;
      let left = rect.left;
      if (chatPanel) {
        const panelRect = chatPanel.getBoundingClientRect();
        maxWidth = Math.floor(panelRect.width * 0.7);
        // If the menu would extend past the panel's right edge, shift it
        // left so the right edge aligns with the panel (8px inset). Don't
        // shift past the panel's left edge.
        const desiredRight = left + maxWidth;
        const panelRight = panelRect.right - 8;
        if (desiredRight > panelRight) {
          left = Math.max(panelRect.left + 8, panelRight - maxWidth);
        }
      }
      // Whatever the panel cap says, the menu can never be wider than the
      // window itself — the quick tooltip window is narrower than the ghost
      // menu's CSS max-width (480px).
      const viewportCap = window.innerWidth - 2 * VIEWPORT_EDGE_MARGIN;
      maxWidth = maxWidth === undefined ? viewportCap : Math.min(maxWidth, viewportCap);
      setMenuPos({
        left,
        // For above-placement we anchor the menu's BOTTOM edge to the trigger
        // top; the inline style below converts this into a `bottom` value.
        top: useAbove ? rect.top : rect.bottom,
        width: rect.width,
        maxWidth,
      });
    };
    compute();
    window.addEventListener("resize", compute);
    window.addEventListener("scroll", compute, true);
    return () => {
      window.removeEventListener("resize", compute);
      window.removeEventListener("scroll", compute, true);
    };
  }, [open, placementPreference]);

  // The menu anchors to the trigger's LEFT edge, so a trigger near the
  // window's right edge (the quick tooltip's autonomy select) pushes the
  // wider menu past the webview bounds, where it is hard-clipped. The ghost
  // menu is content-sized (`width: max-content`), so its width is only
  // knowable after render: measure and shift back into the viewport before
  // paint. Runs after every `compute()` since that resets `left`.
  useLayoutEffect(() => {
    if (!open) return;
    const menu = menuRef.current;
    if (!menu) return;
    const menuWidth = menu.getBoundingClientRect().width;
    const maxLeft = window.innerWidth - VIEWPORT_EDGE_MARGIN - menuWidth;
    setMenuPos((pos) => {
      if (!pos) return pos;
      const left = Math.max(VIEWPORT_EDGE_MARGIN, Math.min(pos.left, maxLeft));
      return Math.abs(left - pos.left) < 1 ? pos : { ...pos, left };
    });
  }, [open, menuPos]);

  return (
    <div
      ref={rootRef}
      className="screenie-select"
      data-open={open}
      data-variant={variant}
      data-placement={resolvedPlacement}
      onKeyDownCapture={(event) => {
        if (!open || event.key !== "Escape") return;
        event.stopPropagation();
        updateOpen(false);
      }}
    >
      <button
        ref={triggerRef}
        type="button"
        className="screenie-select-trigger"
        aria-label={ariaLabel}
        aria-haspopup="listbox"
        aria-expanded={open}
        disabled={disabled || options.length === 0}
        onClick={() => updateOpen((next) => !next)}
        onKeyDown={(event) => {
          if (event.key !== "ArrowDown" && event.key !== "ArrowUp") return;
          event.preventDefault();
          updateOpen(true);
        }}
      >
        <span className="screenie-select-label">
          {triggerLabel ?? (
            <span className="screenie-select-value">
              <span className="screenie-select-value-main">{selectedParts.main}</span>
              {selectedParts.detail && (
                <span className="screenie-select-value-detail">
                  {selectedParts.detail}
                </span>
              )}
            </span>
          )}
        </span>
        <ChevronDown
          className="screenie-select-chevron"
          size={14}
          strokeWidth={1.75}
          aria-hidden
        />
      </button>

      {open &&
        menuPos &&
        createPortal(
          <div
            ref={menuRef}
            className="screenie-select-menu screenie-select-menu-portal"
            data-variant={variant}
            data-placement={resolvedPlacement}
            role="listbox"
            aria-label={ariaLabel}
            style={{
              position: "fixed",
              left: menuPos.left,
              // Anchor below or above the trigger — `menuPos.top` is the
              // trigger's bottom for "below", or its top for "above". For
              // "above" we convert that to a `bottom` value (viewport-anchored)
              // so the menu's bottom edge sits MENU_GAP above the trigger.
              ...(resolvedPlacement === "above"
                ? {
                    bottom: window.innerHeight - menuPos.top + MENU_GAP,
                    top: "auto" as const,
                  }
                : { top: menuPos.top + MENU_GAP, bottom: "auto" as const }),
              minWidth: variant === "ghost" ? undefined : menuPos.width,
              width: variant === "ghost" ? undefined : menuPos.width,
              // 70%-of-chat-panel cap (only set inside .screenie-chat-panel).
              maxWidth: menuPos.maxWidth,
              maxHeight: menuMaxHeight,
              // Keep onMouseDown propagation to the body OUT — the overlay's
              // backdrop-click-to-close listens on the fullLayer element,
              // and the menu floats over the same surface.
            }}
            onMouseDown={(event) => event.stopPropagation()}
            onKeyDown={(event) => {
              if (
                event.key === "ArrowDown" ||
                event.key === "ArrowUp" ||
                event.key === "Home" ||
                event.key === "End"
              ) {
                event.preventDefault();
              }
            }}
          >
            {showSearch && (
              <div className="screenie-select-search-wrap">
                <Search
                  className="screenie-select-search-icon"
                  size={13}
                  strokeWidth={1.75}
                  aria-hidden
                />
                <input
                  ref={searchRef}
                  type="text"
                  value={searchQuery}
                  onChange={(event) => setSearchQuery(event.target.value)}
                  onKeyDown={(event) => {
                    if (event.key === "Enter") {
                      event.preventDefault();
                      const first = filteredOptions[0];
                      if (first) {
                        onChange(first.value);
                        updateOpen(false);
                      }
                    }
                  }}
                  className="screenie-select-search"
                  placeholder="Search…"
                  aria-label="Filter options"
                  spellCheck={false}
                />
              </div>
            )}

            <div className="screenie-select-options">
              {filteredOptions.length === 0 ? (
                <div className="screenie-select-empty">No matches</div>
              ) : (
                filteredOptions.map((option) => {
                  const parts = optionParts(option, option.value);
                  return (
                    <button
                      key={option.value}
                      type="button"
                      className="screenie-select-option"
                      role="option"
                      aria-selected={option.value === value}
                      data-selected={option.value === value}
                      onClick={() => {
                        onChange(option.value);
                        updateOpen(false);
                      }}
                    >
                      <span className="screenie-select-option-label">
                        <span className="screenie-select-option-main">{parts.main}</span>
                        {parts.detail && (
                          <span className="screenie-select-option-detail">
                            {parts.detail}
                          </span>
                        )}
                      </span>
                      <Check
                        className="screenie-select-check"
                        size={14}
                        strokeWidth={2}
                        aria-hidden
                      />
                    </button>
                  );
                })
              )}
            </div>
          </div>,
          document.body,
        )}
    </div>
  );
}
