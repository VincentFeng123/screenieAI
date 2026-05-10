import { useState } from "react";
import Welcome from "./steps/Welcome";
import ApiKeys from "./steps/ApiKeys";
import OllamaStep from "./steps/OllamaStep";
import Tutorial from "./steps/Tutorial";
import "./onboarding.css";

export type OnboardingState = {
  hasCloudKey: boolean;
};

const STEPS = ["Welcome", "Keys", "Ollama", "Tutorial"] as const;

export default function Onboarding({ onComplete }: { onComplete: () => void | Promise<void> }) {
  const [step, setStep] = useState(0);
  const [state, setState] = useState<OnboardingState>({ hasCloudKey: false });

  const next = () => {
    if (step < STEPS.length - 1) setStep(step + 1);
    else onComplete();
  };
  const back = () => setStep((s) => Math.max(0, s - 1));

  return (
    <div className="onboarding-root">
      <header className="onboarding-header" data-tauri-drag-region>
        <span className="onboarding-mark">
          Screenie<span className="onboarding-mark-dim">AI</span>
        </span>
      </header>

      <main className="onboarding-step">
        {step === 0 && <Welcome onNext={next} />}
        {step === 1 && (
          <ApiKeys
            onNext={(hasCloudKey) => {
              setState((s) => ({ ...s, hasCloudKey }));
              next();
            }}
            onBack={back}
            onSkip={() => {
              setState((s) => ({ ...s, hasCloudKey: false }));
              next();
            }}
          />
        )}
        {step === 2 && (
          <OllamaStep
            hasCloudKey={state.hasCloudKey}
            onNext={next}
            onBack={back}
            onSkip={next}
          />
        )}
        {step === 3 && <Tutorial onComplete={onComplete} onBack={back} />}
      </main>

      <footer
        className="onboarding-progress-footer"
        aria-label={`Step ${step + 1} of ${STEPS.length}`}
      >
        <div className="onboarding-progress">
          {STEPS.map((label, i) => (
            <span
              key={label}
              className={
                "onboarding-progress-dot " +
                (i === step ? "active" : i < step ? "done" : "")
              }
            />
          ))}
        </div>
      </footer>
    </div>
  );
}
