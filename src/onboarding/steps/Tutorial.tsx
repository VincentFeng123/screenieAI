import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { Check } from "lucide-react";

// Interactive tutorial: enables tutorial mode in Rust on mount, prompts the
// user to press the global hotkey, and waits for the `tutorial-capture-complete`
// event from Rust (fired when the overlay closes after the tutorial capture).
// Rust hides this window before the screenshot fires and re-shows it after
// the overlay closes — see lib.rs trigger_capture_flow + overlay close
// handler.
export default function Tutorial({
  onComplete,
  onBack,
}: {
  onComplete: () => void | Promise<void>;
  onBack: () => void;
}) {
  const [done, setDone] = useState(false);
  const [hotkeyError, setHotkeyError] = useState<string | null>(null);
  const isMac =
    typeof navigator !== "undefined" && navigator.userAgent.includes("Mac");

  useEffect(() => {
    void invoke("set_tutorial_mode", { active: true });
    invoke<string | null>("get_hotkey_registration_error")
      .then((msg) => {
        if (msg) setHotkeyError(msg);
      })
      .catch(() => {});
    const unlistenP = listen("tutorial-capture-complete", () => {
      setDone(true);
      // Rust swaps tutorial_mode back to false when finishing the overlay
      // session. Re-arm it so a second hotkey press (before the user clicks
      // Done) still hides this onboarding window for the screenshot. Without
      // this, a retry would capture the onboarding window itself.
      void invoke("set_tutorial_mode", { active: true });
    });
    return () => {
      void invoke("set_tutorial_mode", { active: false });
      unlistenP.then((fn) => fn());
    };
  }, []);

  const handleBack = () => {
    void invoke("set_tutorial_mode", { active: false });
    onBack();
  };

  const handleDone = async () => {
    await invoke("set_tutorial_mode", { active: false }).catch(() => {});
    await onComplete();
  };

  return (
    <div className="onboarding-step-inner tutorial">
      <span className="onboarding-eyebrow">Try it</span>

      <h1 className="onboarding-h1">
        {done ? "You're set." : "Press the hotkey."}
      </h1>

      <p className="onboarding-subtitle">
        {hotkeyError
          ? "The global hotkey did not register. You can finish setup, then use the menu bar icon or open Settings to resolve the shortcut."
          : done
          ? "Press the hotkey from any app at any time. Screenie lives in the menu bar — left-click to capture, right-click for Settings or to quit."
          : "We'll get out of the way, capture the screen behind this window, and bring you back here when you close the overlay."}
      </p>

      {hotkeyError && <div className="onboarding-error">{hotkeyError}</div>}

      <div className="onboarding-tutorial-stage">
        {hotkeyError ? (
          <div className="onboarding-tutorial-hotkey" aria-label="Hotkey unavailable">
            <span className="onboarding-key wide">Menu bar</span>
          </div>
        ) : done ? (
          <div className="onboarding-check" aria-hidden>
            <Check size={42} strokeWidth={2.5} />
          </div>
        ) : (
          <div className="onboarding-tutorial-hotkey" aria-label="Press the hotkey now">
            {isMac ? (
              <>
                <span className="onboarding-key">⌘</span>
                <span className="onboarding-key">⇧</span>
                <span className="onboarding-key">A</span>
              </>
            ) : (
              <>
                <span className="onboarding-key wide">Ctrl</span>
                <span className="onboarding-key wide">Shift</span>
                <span className="onboarding-key">A</span>
              </>
            )}
          </div>
        )}
        <p className="onboarding-tutorial-prompt">
          {hotkeyError ? "Hotkey unavailable" : done ? "Capture complete" : "Press to continue"}
        </p>
      </div>

      <div className="onboarding-actions">
        <button className="onboarding-link back" onClick={handleBack}>
          <span className="arrow" aria-hidden>←</span>
          Back
        </button>
        <div className="onboarding-actions-right">
          {done || hotkeyError ? (
            <button className="onboarding-btn primary" onClick={handleDone}>
              Done
              <span className="arrow" aria-hidden>→</span>
            </button>
          ) : (
            <button
              className="onboarding-link"
              onClick={() => {
                // The user opted out of demoing the hotkey — disable
                // tutorial mode immediately so a stray hotkey press doesn't
                // continue to hide this window.
                void invoke("set_tutorial_mode", { active: false });
                setDone(true);
              }}
            >
              Skip the demo
            </button>
          )}
        </div>
      </div>
    </div>
  );
}
