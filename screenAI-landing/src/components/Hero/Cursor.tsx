import { forwardRef } from "react";

export const Cursor = forwardRef<HTMLDivElement>((_, ref) => {
  return (
    <div className="sl-cursor" ref={ref} aria-hidden="true">
      <svg width="22" height="22" viewBox="0 0 22 22">
        <circle cx="11" cy="11" r="10" fill="rgba(0,0,0,0.4)" />
        <line x1="11" y1="2" x2="11" y2="20" stroke="#fff" strokeWidth="1" />
        <line x1="2" y1="11" x2="20" y2="11" stroke="#fff" strokeWidth="1" />
        <circle cx="11" cy="11" r="2.4" fill="#fff" />
      </svg>
    </div>
  );
});
Cursor.displayName = "Cursor";
