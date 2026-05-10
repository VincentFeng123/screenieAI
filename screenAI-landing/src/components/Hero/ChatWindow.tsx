import { forwardRef } from "react";
import { X, ArrowUp, Image as ImageIcon } from "lucide-react";

type Props = {
  prompt: string;
  answerRef: React.RefObject<HTMLSpanElement | null>;
  caretRef: React.RefObject<HTMLSpanElement | null>;
};

export const ChatWindow = forwardRef<HTMLDivElement, Props>(
  ({ prompt, answerRef, caretRef }, ref) => {
    return (
      <div className="sl-chat" ref={ref} aria-hidden="true">
        <div className="sl-chat__hairline" />
        <div className="sl-chat__title">
          <button className="sl-chat__close" aria-label="Close" tabIndex={-1}>
            <X size={9} strokeWidth={2.4} />
          </button>
          <span className="sl-chat__title-label">
            <ImageIcon size={11} strokeWidth={1.8} />
            Selection · 482 × 218
          </span>
          <span className="sl-chat__title-spacer" />
          <span className="sl-chat__pill">CLOUD · CLAUDE</span>
        </div>

        <div className="sl-chat__body">
          <div className="sl-chat__capture">
            <div className="sl-chat__capture-shimmer" />
            <svg className="sl-chat__capture-chart" viewBox="0 0 220 110" preserveAspectRatio="none">
              <path d="M 4 96 C 30 90, 56 78, 80 64 S 130 38, 160 26 S 200 14, 216 8" />
              <circle cx="56" cy="74" r="2.4" />
              <circle cx="118" cy="46" r="2.4" />
              <circle cx="172" cy="22" r="2.4" />
            </svg>
          </div>

          <div className="sl-chat__msg sl-chat__msg--user">
            <span>{prompt}</span>
          </div>

          <div className="sl-chat__msg sl-chat__msg--ai">
            <span className="sl-chat__answer" ref={answerRef} />
            <span className="sl-chat__caret" ref={caretRef} aria-hidden="true" />
          </div>
        </div>

        <div className="sl-chat__prompt">
          <span className="sl-chat__prompt-text">Ask a follow-up…</span>
          <button className="sl-chat__send" aria-label="Send" tabIndex={-1}>
            <ArrowUp size={14} strokeWidth={2} />
          </button>
        </div>
      </div>
    );
  }
);
ChatWindow.displayName = "ChatWindow";
