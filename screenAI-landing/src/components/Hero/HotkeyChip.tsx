import { forwardRef } from "react";

type Props = { keys?: string[] };

export const HotkeyChip = forwardRef<HTMLDivElement, Props>(
  ({ keys = ["⌘", "⇧", "Space"] }, ref) => {
    return (
      <div className="sl-hotkey" ref={ref} aria-hidden="true">
        <div className="sl-hotkey__keys">
          {keys.map((k, i) => (
            <span key={i} className={`sl-keycap${k === "Space" ? " sl-keycap--wide" : ""}`}>
              {k}
            </span>
          ))}
        </div>
        <span className="sl-hotkey__label">Press to capture</span>
      </div>
    );
  }
);
HotkeyChip.displayName = "HotkeyChip";
