import { forwardRef, ReactNode } from "react";

type Props = { children?: ReactNode };

export const FakeDesktop = forwardRef<HTMLDivElement, Props>(({ children }, ref) => {
  return (
    <div className="sl-stage" ref={ref} aria-hidden="true">
      <div className="sl-stage__menubar">
        <span className="sl-stage__dot" />
        <span className="sl-stage__dot" />
        <span className="sl-stage__dot" />
        <div className="sl-stage__menubar-spacer" />
        <span className="sl-stage__menu-item">File</span>
        <span className="sl-stage__menu-item">Edit</span>
        <span className="sl-stage__menu-item">View</span>
        <div className="sl-stage__menubar-clock">
          <span className="sl-stage__clock-bar" />
          <span className="sl-stage__clock-bar" />
          <span className="sl-stage__clock-bar" />
        </div>
      </div>

      <div className="sl-stage__wallpaper">
        <div className="sl-stage__wallpaper-doc">
          <div className="sl-stage__doc-row sl-stage__doc-row--title" />
          <div className="sl-stage__doc-row" style={{ width: "78%" }} />
          <div className="sl-stage__doc-row" style={{ width: "92%" }} />
          <div className="sl-stage__doc-row" style={{ width: "65%" }} />
          <div className="sl-stage__doc-chart">
            <div className="sl-stage__doc-chart-grid" />
            <svg viewBox="0 0 220 110" className="sl-stage__doc-chart-svg" preserveAspectRatio="none">
              <path d="M 4 96 C 30 90, 56 78, 80 64 S 130 38, 160 26 S 200 14, 216 8" />
              <circle cx="56" cy="74" r="2.4" />
              <circle cx="118" cy="46" r="2.4" />
              <circle cx="172" cy="22" r="2.4" />
            </svg>
          </div>
          <div className="sl-stage__doc-row" style={{ width: "84%" }} />
          <div className="sl-stage__doc-row" style={{ width: "57%" }} />
        </div>
      </div>

      {children}
    </div>
  );
});

FakeDesktop.displayName = "FakeDesktop";
