export default function Welcome({ onNext }: { onNext: () => void }) {
  return (
    <div className="onboarding-step-inner welcome">
      <div className="onboarding-welcome-grid">
        <div className="onboarding-welcome-content">
          <span className="onboarding-eyebrow">Welcome</span>
          <h1 className="onboarding-h1">
            Ask AI about anything on your screen.
          </h1>
          <p className="onboarding-subtitle">
            Press a global hotkey from any app, drag a region, type a question.
            The answer streams back over your screenshot — Claude, OpenAI, or
            local Ollama, your choice.
          </p>
          <div className="onboarding-welcome-cta">
            <button className="onboarding-btn primary" onClick={onNext}>
              Get started
              <span className="arrow" aria-hidden>→</span>
            </button>
          </div>
        </div>

        {/* Placeholder product art — inline SVG so it inherits currentColor
            and themes for light/dark. Swap with <img src="/welcome.png" />
            (or similar) when there's a real asset. */}
        <div className="onboarding-welcome-image" aria-hidden>
          <svg
            xmlns="http://www.w3.org/2000/svg"
            viewBox="0 0 480 360"
            preserveAspectRatio="xMidYMid meet"
          >
            <g transform="translate(60, 80)">
              <rect
                width="360"
                height="200"
                rx="14"
                fill="currentColor"
                fillOpacity="0.06"
                stroke="currentColor"
                strokeOpacity="0.22"
                strokeWidth="1"
              />
              <circle cx="22" cy="22" r="4.5" fill="currentColor" opacity="0.3" />
              <circle cx="38" cy="22" r="4.5" fill="currentColor" opacity="0.3" />
              <circle cx="54" cy="22" r="4.5" fill="currentColor" opacity="0.3" />
              <rect x="22" y="60" width="190" height="6" rx="3" fill="currentColor" opacity="0.3" />
              <rect x="22" y="76" width="290" height="6" rx="3" fill="currentColor" opacity="0.16" />
              <rect x="22" y="92" width="170" height="6" rx="3" fill="currentColor" opacity="0.16" />
              <rect x="22" y="108" width="230" height="6" rx="3" fill="currentColor" opacity="0.16" />
              <rect
                x="178"
                y="118"
                width="146"
                height="64"
                fill="currentColor"
                fillOpacity="0.06"
                stroke="currentColor"
                strokeWidth="1.5"
              />
              <rect x="174" y="114" width="8" height="8" fill="currentColor" />
              <rect x="320" y="114" width="8" height="8" fill="currentColor" />
              <rect x="174" y="178" width="8" height="8" fill="currentColor" />
              <rect x="320" y="178" width="8" height="8" fill="currentColor" />
              <path
                d="M 240 146 L 240 162 L 244.5 158 L 248.5 164.5 L 250.5 163.5 L 246.5 157 L 252.5 157 Z"
                fill="currentColor"
              />
            </g>
          </svg>
        </div>
      </div>
    </div>
  );
}
