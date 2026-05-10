import { useEffect, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import SettingsPanel from "./settings/SettingsPanel";
import Onboarding from "./onboarding/Onboarding";
import { applyStoredPreferences } from "./settings/preferences";
import "./App.css";

type View = "loading" | "onboarding" | "settings";

const ONBOARDING_KEY = "onboarding_complete";

export default function App() {
  const [view, setView] = useState<View>("loading");

  // Decide on first paint whether to run onboarding or stay hidden.
  // Rust always launches the main window with `visible: false`. If onboarding
  // is incomplete we ask Rust to show it (which also promotes to Regular so
  // a Dock icon appears). If complete, we leave the window hidden — the user
  // reaches Settings only via the tray menu.
  useEffect(() => {
    applyStoredPreferences();
    const done = localStorage.getItem(ONBOARDING_KEY) === "1";
    if (done) {
      setView("settings");
    } else {
      invoke("show_settings_window")
        .catch((e) => console.error("show_settings_window failed:", e))
        .finally(() => setView("onboarding"));
    }
  }, []);

  if (view === "loading") return null;

  if (view === "onboarding") {
    return (
      <Onboarding
        onComplete={async () => {
          localStorage.setItem(ONBOARDING_KEY, "1");
          try {
            await invoke("complete_onboarding");
          } catch (e) {
            console.error("complete_onboarding failed:", e);
          }
          setView("settings");
        }}
      />
    );
  }

  return (
    <SettingsPanel
      onRunOnboardingAgain={() => {
        // Re-runs in the same window. The window is already shown (the user
        // reached Settings via tray, which also promotes to Regular), so we
        // just swap the view.
        setView("onboarding");
      }}
    />
  );
}
