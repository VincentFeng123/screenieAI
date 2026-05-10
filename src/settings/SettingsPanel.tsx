import {
  Fragment,
  useEffect,
  useLayoutEffect,
  useMemo,
  useRef,
  useState,
  type CSSProperties,
  type MouseEvent as ReactMouseEvent,
  type ReactNode,
} from "react";
import { invoke } from "@tauri-apps/api/core";
import { listen } from "@tauri-apps/api/event";
import { getCurrentWindow } from "@tauri-apps/api/window";
import { Eye, EyeOff, Plus, RefreshCw, Trash2, X } from "lucide-react";
import OllamaSetup from "./OllamaSetup";
import NeedsVisionModel from "./NeedsVisionModel";
import CustomDropdown, {
  type CustomDropdownOption,
} from "../components/CustomDropdown";
import {
  Provider,
  OllamaStatus,
  ANTHROPIC_MODELS,
  OPENAI_MODELS,
  GEMINI_MODELS,
  looksLikeVisionModel,
} from "./constants";
import {
  applyStoredPreferences,
  readPreferences,
  subscribePreferences,
  writePreference,
  type ScreeniePreferences,
} from "./preferences";
import { debounce } from "../lib/debounce";
import {
  addTemplate,
  deleteTemplate,
  readTemplates,
  resetTemplates,
  subscribeTemplates,
  updateTemplate,
  type PromptTemplate,
} from "../lib/templates";
import { clearHistory } from "../lib/history";
import HistoryList from "../components/HistoryList";
import {
  clearAll as clearAllUsage,
  currentMonthKey,
  formatCostCents,
  formatTokens,
  readAllMonths,
  subscribeUsage,
  type ProviderTotals,
  type UsageStore,
} from "../lib/usage";
import "./settings.css";

type SectionId =
  | "overview"
  | "providers"
  | "overlay"
  | "ai-output"
  | "appearance"
  | "templates"
  | "history"
  | "maintenance";

type ProviderMeta = {
  id: Provider;
  label: string;
  detail: string;
  kind: "cloud" | "local";
};

const settingsSections: { id: SectionId; label: string }[] = [
  { id: "overview", label: "Overview" },
  { id: "providers", label: "Providers" },
  { id: "overlay", label: "Overlay" },
  { id: "ai-output", label: "AI Output" },
  { id: "templates", label: "Templates" },
  { id: "history", label: "History" },
  { id: "appearance", label: "Appearance" },
  { id: "maintenance", label: "Maintenance" },
];

const providerMetas: ProviderMeta[] = [
  { id: "anthropic", label: "Claude", detail: "Anthropic cloud", kind: "cloud" },
  { id: "openai", label: "OpenAI", detail: "OpenAI cloud", kind: "cloud" },
  { id: "gemini", label: "Gemini", detail: "Google cloud", kind: "cloud" },
  { id: "ollama", label: "Ollama", detail: "Local private models", kind: "local" },
];

const anthropicModelOptions: CustomDropdownOption[] = ANTHROPIC_MODELS.map((m) => ({
  value: m.id,
  label: m.label,
}));
const openaiModelOptions: CustomDropdownOption[] = OPENAI_MODELS.map((m) => ({
  value: m.id,
  label: m.label,
}));
const geminiModelOptions: CustomDropdownOption[] = GEMINI_MODELS.map((m) => ({
  value: m.id,
  label: m.label,
}));

function storedModelOrDefault(key: string, options: CustomDropdownOption[]): string {
  const saved = localStorage.getItem(key);
  return saved && options.some((option) => option.value === saved)
    ? saved
    : options[0].value;
}

const WINDOW_DRAG_BLOCKERS =
  'button, input, textarea, select, a, [role="button"], [role="switch"], [role="listbox"], .screenie-select';

const WINDOW_DRAG_SURFACES =
  ".settings-shell, .settings-sidebar, .settings-main, .settings-section, .settings-brand, .settings-nav, .settings-sidebar-footer, .settings-page-header, .settings-card, .settings-card-header";

function startWindowDragFromEmptySpace(event: ReactMouseEvent<HTMLElement>) {
  if (event.button !== 0) return;
  const target = event.target;
  if (!(target instanceof HTMLElement)) return;
  if (target.closest(WINDOW_DRAG_BLOCKERS)) return;
  if (!target.closest(WINDOW_DRAG_SURFACES)) return;

  getCurrentWindow()
    .startDragging()
    .catch(() => {
      /* browser preview or unavailable native window */
    });
}

export default function SettingsPanel({
  onRunOnboardingAgain,
}: {
  onRunOnboardingAgain?: () => void;
}) {
  const isMac =
    typeof navigator !== "undefined" && navigator.userAgent.includes("Mac");
  const hotkey = isMac ? "⌘ + Shift + A" : "Ctrl + Shift + A";

  const [activeSection, setActiveSection] = useState<SectionId>("overview");
  const [preferences, setPreferences] = useState<ScreeniePreferences>(() =>
    readPreferences(),
  );
  // The frosted-glass effect is now provided by the macOS native sidebar
  // window effect + CSS `backdrop-filter` on `.settings-shell` (settings.css).
  // Both update LIVE — the desktop content visible through the window
  // changes as the user moves windows around behind it. The earlier
  // BlurredBackdrop bitmap was captured once and never refreshed, so the
  // frost looked frozen.
  // Initialize from localStorage on first render so the UI doesn't flash
  // Anthropic for one frame before the useEffect catches up.
  const [provider, setProvider] = useState<Provider>(() => {
    const saved = (typeof window !== "undefined"
      ? localStorage.getItem("provider")
      : null) as Provider | null;
    return saved && ["anthropic", "openai", "gemini", "ollama"].includes(saved)
      ? saved
      : "anthropic";
  });
  const [anthropicKey, setAnthropicKey] = useState("");
  const [openaiKey, setOpenaiKey] = useState("");
  const [geminiKey, setGeminiKey] = useState("");
  const [revealKey, setRevealKey] = useState(false);
  const [ollamaModel, setOllamaModel] = useState(
    () => localStorage.getItem("ollama_model") || "llama3.2-vision",
  );
  const [anthropicModel, setAnthropicModel] = useState(
    () => storedModelOrDefault("anthropic_model", anthropicModelOptions),
  );
  const [openaiModel, setOpenaiModel] = useState(
    () => storedModelOrDefault("openai_model", openaiModelOptions),
  );
  const [geminiModel, setGeminiModel] = useState(
    () => storedModelOrDefault("gemini_model", geminiModelOptions),
  );
  const [ollama, setOllama] = useState<OllamaStatus>({ running: false, models: [] });
  const [checking, setChecking] = useState(false);
  const [saveError, setSaveError] = useState<string | null>(null);
  const [hotkeyError, setHotkeyError] = useState<string | null>(null);

  useEffect(() => {
    applyStoredPreferences();
    return subscribePreferences((next) => {
      setPreferences(next);
      applyStoredPreferences();
    });
  }, []);

  // Surface hotkey registration failures (commonly: missing Accessibility
  // permission, or another app owns the shortcut). Rust emits this once
  // at startup if either shortcut failed to register.
  useEffect(() => {
    invoke<string | null>("get_hotkey_registration_error")
      .then((msg) => {
        if (msg) setHotkeyError(msg);
      })
      .catch(() => {});
    const unlistenP = listen<string>("hotkey-registration-failed", (event) => {
      setHotkeyError(event.payload);
    });
    return () => {
      unlistenP.then((fn) => fn());
    };
  }, []);

  // Load API keys from the keychain on mount. Provider + model preferences
  // are already seeded from localStorage in the useState initializers above.
  useEffect(() => {
    (async () => {
      // One-time migration: if an Anthropic key is still in localStorage from
      // an earlier version, move it into the OS keyring and clear localStorage.
      const legacy = localStorage.getItem("anthropic_api_key");
      if (legacy) {
        try {
          await invoke("set_secret", {
            name: "anthropic_api_key",
            value: legacy,
          });
          localStorage.removeItem("anthropic_api_key");
        } catch {
          /* best effort */
        }
      }
      try {
        const aKey = await invoke<string | null>("get_secret", {
          name: "anthropic_api_key",
        });
        if (aKey) setAnthropicKey(aKey);
      } catch {
        /* keyring not reachable */
      }
      try {
        const oKey = await invoke<string | null>("get_secret", {
          name: "openai_api_key",
        });
        if (oKey) setOpenaiKey(oKey);
      } catch {
        /* keyring not reachable */
      }
      try {
        const gKey = await invoke<string | null>("get_secret", {
          name: "gemini_api_key",
        });
        if (gKey) setGeminiKey(gKey);
      } catch {
        /* keyring not reachable */
      }
    })();
  }, []);

  const ollamaModelRef = useRef(ollamaModel);
  ollamaModelRef.current = ollamaModel;

  const checkOllama = async () => {
    setChecking(true);
    try {
      const status = await invoke<OllamaStatus>("check_ollama");
      setOllama(status);
      if (
        status.running &&
        status.models.length &&
        !status.models.includes(ollamaModelRef.current)
      ) {
        const next = status.models.find(looksLikeVisionModel);
        if (next) {
          setOllamaModel(next);
          localStorage.setItem("ollama_model", next);
        }
      }
    } finally {
      setChecking(false);
    }
  };

  // Auto-check Ollama whenever the provider switches to it.
  useEffect(() => {
    if (provider === "ollama") checkOllama();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [provider]);

  // Auto-poll while the user is on the Ollama provider and the daemon is not
  // yet reachable, so installing it in another window updates the UI.
  useEffect(() => {
    if (provider !== "ollama") return;
    if (ollama.running) return;
    const id = setInterval(() => {
      checkOllama();
    }, 2500);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [provider, ollama.running]);

  const saveProvider = (p: Provider) => {
    setProvider(p);
    localStorage.setItem("provider", p);
  };

  // Debounce keychain writes so a paste-then-edit (or character-by-character
  // typing) doesn't fire one set_secret per keystroke. 350ms is short enough
  // that the user perceives the save as instant.
  const debouncedKeyringWrite = useMemo(
    () =>
      debounce((name: string, val: string) => {
        const op = val
          ? invoke("set_secret", { name, value: val })
          : invoke("delete_secret", { name });
        op.catch((e) => {
          const msg = typeof e === "string" ? e : (e && (e as Error).message) || String(e);
          console.error("keyring write failed:", e);
          setSaveError(`Couldn't save to the system keychain: ${msg}`);
        });
      }, 350),
    [],
  );

  // Flush any pending keychain write when the panel unmounts (e.g., the
  // user navigates to overlay or quits) so an in-flight debounce doesn't
  // drop the latest character.
  useEffect(() => {
    return () => debouncedKeyringWrite.flush();
  }, [debouncedKeyringWrite]);

  const saveSecret = (name: string, val: string) => {
    setSaveError(null);
    debouncedKeyringWrite(name, val);
  };

  const writeSecretNow = async (name: string, val: string) => {
    if (val) await invoke("set_secret", { name, value: val });
    else await invoke("delete_secret", { name });
  };

  const quitFromSettings = async () => {
    setSaveError(null);
    debouncedKeyringWrite.cancel();
    try {
      await Promise.all([
        writeSecretNow("anthropic_api_key", anthropicKey),
        writeSecretNow("openai_api_key", openaiKey),
        writeSecretNow("gemini_api_key", geminiKey),
      ]);
      await invoke("quit_app");
    } catch (e) {
      const msg = typeof e === "string" ? e : (e && (e as Error).message) || String(e);
      setSaveError(`Couldn't save to the system keychain: ${msg}`);
    }
  };

  const saveAnthropicKey = (val: string) => {
    setAnthropicKey(val);
    saveSecret("anthropic_api_key", val);
  };
  const saveOpenaiKey = (val: string) => {
    setOpenaiKey(val);
    saveSecret("openai_api_key", val);
  };
  const saveGeminiKey = (val: string) => {
    setGeminiKey(val);
    saveSecret("gemini_api_key", val);
  };

  const saveOllamaModel = (val: string) => {
    setOllamaModel(val);
    localStorage.setItem("ollama_model", val);
  };
  const saveAnthropicModel = (val: string) => {
    setAnthropicModel(val);
    localStorage.setItem("anthropic_model", val);
  };
  const saveOpenaiModel = (val: string) => {
    setOpenaiModel(val);
    localStorage.setItem("openai_model", val);
  };
  const saveGeminiModel = (val: string) => {
    setGeminiModel(val);
    localStorage.setItem("gemini_model", val);
  };

  const savePreference = <K extends keyof ScreeniePreferences>(
    name: K,
    value: ScreeniePreferences[K],
  ) => {
    writePreference(name, value);
    setPreferences(readPreferences());
  };

  const hasOllamaVisionModel = ollama.models.some(looksLikeVisionModel);
  const selectedOllamaModelLooksVisionCapable = looksLikeVisionModel(ollamaModel);
  const ollamaReady =
    ollama.running && hasOllamaVisionModel && selectedOllamaModelLooksVisionCapable;

  const ready =
    (provider === "anthropic" && anthropicKey.length > 0) ||
    (provider === "openai" && openaiKey.length > 0) ||
    (provider === "gemini" && geminiKey.length > 0) ||
    (provider === "ollama" && ollamaReady);

  const activeProviderMeta =
    providerMetas.find((meta) => meta.id === provider) ?? providerMetas[0];
  const activeModel = modelForProvider(
    provider,
    anthropicModel,
    openaiModel,
    geminiModel,
    ollamaModel,
  );
  const ollamaStatusLabel = getOllamaStatusLabel(
    checking,
    ollama.running,
    ollamaReady,
    hasOllamaVisionModel,
  );
  const activeProviderStatus =
    provider === "ollama" ? ollamaStatusLabel : ready ? "Ready" : "Needs key";

  const overviewStats = useMemo(
    () => [
      { label: "Status", value: ready ? "Ready" : "Not configured" },
      { label: "Provider", value: activeProviderMeta.label },
      { label: "Model", value: activeModel },
    ],
    [activeModel, activeProviderMeta.label, ready],
  );
  const activeNavIndex = Math.max(
    0,
    settingsSections.findIndex((section) => section.id === activeSection),
  );

  return (
    <div
      className="settings-shell"
      data-density={preferences.settingsDensity}
      onMouseDown={startWindowDragFromEmptySpace}
    >
      <button
        type="button"
        className="settings-close-btn"
        aria-label="Close settings"
        title="Close settings"
        data-tauri-drag-region="false"
        onMouseDown={(e) => e.stopPropagation()}
        onClick={(e) => {
          e.stopPropagation();
          invoke("hide_settings_window").catch((err) =>
            console.error("hide_settings_window failed:", err),
          );
        }}
      >
        <X size={11} strokeWidth={2.25} aria-hidden />
      </button>
      <aside className="settings-sidebar" aria-label="Settings navigation">
        {/* No separate BlurredBackdrop here — the sidebar is now a
            column on the unified chat-panel surface, separated from
            `main` only by a 1px hairline (see settings.css). The shell's
            own frost serves as the single backdrop for both columns. */}
        <div className="settings-brand">
          <h1 className="settings-brand-title">Screenie AI</h1>
          <p className="settings-brand-subtitle">
            Capture, ask, and tune how answers show up.
          </p>
        </div>

        <nav
          className="settings-nav"
          style={{ "--active-nav-index": activeNavIndex } as CSSProperties}
        >
          <span
            className="settings-nav-indicator"
            aria-hidden
          />
          {settingsSections.map((section) => (
            <button
              key={section.id}
              className="settings-nav-btn"
              data-active={activeSection === section.id}
              onClick={() => setActiveSection(section.id)}
            >
              <span>{section.label}</span>
            </button>
          ))}
        </nav>

        <div className="settings-sidebar-footer">
          <span className="settings-hotkey">{hotkey}</span>
          <span>{ready ? "Ready for the global shortcut." : "Finish provider setup to capture."}</span>
        </div>
      </aside>

      <main className="settings-main">
        {saveError && (
          <div className="settings-alert" role="alert">
            <span>{saveError}</span>
            <button onClick={() => setSaveError(null)} aria-label="Dismiss">
              ×
            </button>
          </div>
        )}
        {hotkeyError && (
          <div className="settings-alert" role="alert">
            <span>
              The global hotkey couldn't register ({hotkeyError}). On macOS this
              usually means Accessibility / Input Monitoring permission isn't
              granted yet — open System Settings → Privacy &amp; Security →
              Accessibility (and Input Monitoring), enable Screenie AI, then
              relaunch the app.
            </span>
            <button onClick={() => setHotkeyError(null)} aria-label="Dismiss">
              ×
            </button>
          </div>
        )}

        <SettingsSection
          id="overview"
          active={activeSection === "overview"}
          title="Overview"
          description="A quick read on the capture shortcut, active AI provider, and current model."
        >
          <SettingsCard
            title="Capture status"
            description="The global shortcut opens the region picker from any app."
            action={<StatusPill label={ready ? "Ready" : "Not configured"} good={ready} />}
          >
            <div className="settings-status-grid">
              {overviewStats.map((stat) => (
                <div className="settings-stat" key={stat.label}>
                  <span className="settings-stat-label">{stat.label}</span>
                  <span className="settings-stat-value">
                    {stat.value}
                  </span>
                </div>
              ))}
            </div>
            <p className="settings-note">
              Shortcut: <span className="settings-hotkey">{hotkey}</span>
            </p>
          </SettingsCard>

          <div className="settings-grid-2">
            <SettingsCard
              title="Provider"
              description={`${activeProviderMeta.label} is selected. ${activeProviderMeta.kind === "local" ? "Requests stay local." : "Screenshots are sent to the provider API."}`}
              action={
                <StatusPill
                  label={activeProviderStatus}
                  good={ready}
                  warning={!ready && provider === "ollama" && ollama.running}
                />
              }
            >
              <div className="settings-quick-actions">
                <button
                  className="settings-button settings-button-primary"
                  onClick={() => setActiveSection("providers")}
                >
                  Configure provider
                </button>
              </div>
            </SettingsCard>

            <SettingsCard
              title="Answer layout"
              description={`Responses are ${preferences.aiResponseStyle} and the panel density is ${preferences.aiRenderDensity}.`}
            >
              <div className="settings-quick-actions">
                <button
                  className="settings-button"
                  onClick={() => setActiveSection("ai-output")}
                >
                  Tune AI output
                </button>
                <button
                  className="settings-button"
                  onClick={() => setActiveSection("overlay")}
                >
                  Tune overlay
                </button>
              </div>
            </SettingsCard>
          </div>

          <UsageOverview />
        </SettingsSection>

        <SettingsSection
          id="providers"
          active={activeSection === "providers"}
          title="Providers"
          description="Choose the model backend and store API keys in the system keychain."
        >
          <SettingsCard
            title="AI provider"
            description="Cloud providers are usually fastest. Ollama keeps prompts and screenshots local."
            action={<StatusPill label={activeProviderStatus} good={ready} />}
          >
            <div className="settings-provider-grid">
              {providerMetas.map((meta) => (
                <button
                  key={meta.id}
                  className="settings-provider-option"
                  data-active={provider === meta.id}
                  onClick={() => saveProvider(meta.id)}
                >
                  <span className="settings-provider-label">{meta.label}</span>
                  <span className="settings-provider-detail">{meta.detail}</span>
                </button>
              ))}
            </div>
          </SettingsCard>

          {provider === "anthropic" && (
            <CloudProviderCard
              title="Claude"
              keyTitle="Anthropic API key"
              keyValue={anthropicKey}
              keyPlaceholder="sk-ant-..."
              keyWhere="console.anthropic.com"
              onKeyChange={saveAnthropicKey}
              revealKey={revealKey}
              setRevealKey={setRevealKey}
              modelValue={anthropicModel}
              modelOptions={anthropicModelOptions}
              onModelChange={saveAnthropicModel}
            />
          )}

          {provider === "openai" && (
            <CloudProviderCard
              title="OpenAI"
              keyTitle="OpenAI API key"
              keyValue={openaiKey}
              keyPlaceholder="sk-..."
              keyWhere="platform.openai.com"
              onKeyChange={saveOpenaiKey}
              revealKey={revealKey}
              setRevealKey={setRevealKey}
              modelValue={openaiModel}
              modelOptions={openaiModelOptions}
              onModelChange={saveOpenaiModel}
            />
          )}

          {provider === "gemini" && (
            <CloudProviderCard
              title="Gemini"
              keyTitle="Google AI Studio API key"
              keyValue={geminiKey}
              keyPlaceholder="AIza..."
              keyWhere="aistudio.google.com/apikey"
              onKeyChange={saveGeminiKey}
              revealKey={revealKey}
              setRevealKey={setRevealKey}
              modelValue={geminiModel}
              modelOptions={geminiModelOptions}
              onModelChange={saveGeminiModel}
            />
          )}

          {provider === "ollama" && (
            <SettingsCard
              title="Ollama"
              description="Local vision models run through the Ollama daemon."
              action={
                <StatusPill
                  label={ollamaStatusLabel}
                  good={ollamaReady}
                  warning={ollama.running && !ollamaReady}
                />
              }
            >
              {!ollama.running && (
                <OllamaSetup onCheck={checkOllama} checking={checking} />
              )}
              {ollama.running && (
                <>
                  <PreferenceRow
                    title="Vision model"
                    help="Choose an installed model that can read images."
                  >
                    <div className="settings-input-row">
                      {ollama.models.length > 0 ? (
                        <CustomDropdown
                          value={ollamaModel}
                          options={ollama.models.map((m) => ({ value: m, label: m }))}
                          onChange={saveOllamaModel}
                          ariaLabel="Ollama model"
                        />
                      ) : (
                        <input
                          value={ollamaModel}
                          onChange={(e) => saveOllamaModel(e.target.value)}
                          placeholder="llama3.2-vision"
                          className="settings-input"
                        />
                      )}
                      <button
                        className="settings-button settings-button-icon"
                        onClick={checkOllama}
                        aria-label="Refresh Ollama models"
                      >
                        <RefreshCw size={14} strokeWidth={1.75} aria-hidden />
                      </button>
                    </div>
                  </PreferenceRow>
                  {!hasOllamaVisionModel && (
                    <div style={{ marginTop: 12 }}>
                      <NeedsVisionModel onCheck={checkOllama} />
                    </div>
                  )}
                  {ollama.models.length > 0 &&
                    !selectedOllamaModelLooksVisionCapable && (
                      <p className="settings-note" style={{ color: "var(--error-text)" }}>
                        Heads up: <code>{ollamaModel}</code> may not support images.
                        Try <code>llava</code>, <code>llama3.2-vision</code>, or{" "}
                        <code>qwen2-vl</code>.
                      </p>
                    )}
                </>
              )}
            </SettingsCard>
          )}
        </SettingsSection>

        <SettingsSection
          id="overlay"
          active={activeSection === "overlay"}
          title="Overlay"
          description="Control the first capture panel state and small interaction details."
        >
          <SettingsCard>
            <PreferenceRow
              title="Default chat panel"
              help="Auto avoids covering the selected region when possible."
            >
              <SegmentedControl
                value={preferences.overlayChatDefault}
                options={[
                  { value: "auto", label: "Auto" },
                  { value: "open", label: "Open" },
                  { value: "hidden", label: "Hidden" },
                ]}
                onChange={(value) => savePreference("overlayChatDefault", value)}
                ariaLabel="Default chat panel"
              />
            </PreferenceRow>
            <PreferenceRow
              title="Prompt presets"
              help="Show Explain, Translate, OCR, and Summarize chips near the prompt."
            >
              <ToggleControl
                checked={preferences.overlayShowPresets}
                onChange={(value) => savePreference("overlayShowPresets", value)}
                ariaLabel="Show prompt presets"
              />
            </PreferenceRow>
            <PreferenceRow
              title="Backdrop click closes overlay"
              help="Turn this off if accidental clicks dismiss captures too often."
            >
              <ToggleControl
                checked={preferences.overlayCloseOnBackdrop}
                onChange={(value) => savePreference("overlayCloseOnBackdrop", value)}
                ariaLabel="Close overlay from backdrop"
              />
            </PreferenceRow>
            <PreferenceRow
              title="Allow blank send"
              help="Blank sends ask the AI to describe the selected region."
            >
              <ToggleControl
                checked={preferences.overlayAllowEmptySend}
                onChange={(value) => savePreference("overlayAllowEmptySend", value)}
                ariaLabel="Allow blank send"
              />
            </PreferenceRow>
          </SettingsCard>
        </SettingsSection>

        <SettingsSection
          id="ai-output"
          active={activeSection === "ai-output"}
          title="AI Output"
          description="Make streamed answers easier to scan inside the floating response panel."
        >
          <SettingsCard>
            <PreferenceRow
              title="Response style"
              help="Controls how much detail the model is asked to include."
            >
              <SegmentedControl
                value={preferences.aiResponseStyle}
                options={[
                  { value: "concise", label: "Concise" },
                  { value: "balanced", label: "Balanced" },
                  { value: "detailed", label: "Detailed" },
                ]}
                onChange={(value) => savePreference("aiResponseStyle", value)}
                ariaLabel="Response style"
              />
            </PreferenceRow>
            <PreferenceRow
              title="Panel density"
              help="Changes paragraph and equation spacing in the response panel."
            >
              <SegmentedControl
                value={preferences.aiRenderDensity}
                options={[
                  { value: "compact", label: "Compact" },
                  { value: "comfortable", label: "Comfortable" },
                  { value: "spacious", label: "Spacious" },
                ]}
                onChange={(value) => savePreference("aiRenderDensity", value)}
                ariaLabel="Panel density"
              />
            </PreferenceRow>
          </SettingsCard>
        </SettingsSection>

        <SettingsSection
          id="templates"
          active={activeSection === "templates"}
          title="Templates"
          description="Edit the preset chips that appear next to the prompt textfield in the overlay."
        >
          <TemplatesEditor />
        </SettingsSection>

        <SettingsSection
          id="history"
          active={activeSection === "history"}
          title="History"
          description="Browse past captures and their AI responses. Click a tile to revisit."
        >
          <HistoryBrowser />
        </SettingsSection>

        <SettingsSection
          id="appearance"
          active={activeSection === "appearance"}
          title="Appearance"
          description="Choose the app theme and how much room the Settings UI uses."
        >
          <SettingsCard>
            <PreferenceRow
              title="Theme"
              help="System follows macOS appearance."
            >
              <SegmentedControl
                value={preferences.theme}
                options={[
                  { value: "system", label: "System" },
                  { value: "light", label: "Light" },
                  { value: "dark", label: "Dark" },
                ]}
                onChange={(value) => savePreference("theme", value)}
                ariaLabel="Theme"
              />
            </PreferenceRow>
            <PreferenceRow
              title="Settings density"
              help="Compact keeps more controls visible without changing overlay responses."
            >
              <SegmentedControl
                value={preferences.settingsDensity}
                options={[
                  { value: "comfortable", label: "Comfortable" },
                  { value: "compact", label: "Compact" },
                ]}
                onChange={(value) => savePreference("settingsDensity", value)}
                ariaLabel="Settings density"
              />
            </PreferenceRow>
          </SettingsCard>
        </SettingsSection>

        <SettingsSection
          id="maintenance"
          active={activeSection === "maintenance"}
          title="Maintenance"
          description="Utility actions for setup, local model detection, and quitting the app."
        >
          <SettingsCard>
            <PreferenceRow
              title="First-run setup"
              help="Runs the onboarding flow again in this window."
            >
              <button
                className="settings-button"
                onClick={onRunOnboardingAgain}
                disabled={!onRunOnboardingAgain}
              >
                Run setup again
              </button>
            </PreferenceRow>
            <PreferenceRow
              title="Ollama detection"
              help="Refresh installed local models and daemon status."
            >
              <button className="settings-button" onClick={checkOllama}>
                <RefreshCw size={14} strokeWidth={1.75} aria-hidden />
                {checking ? "Checking…" : "Refresh Ollama"}
              </button>
            </PreferenceRow>
          </SettingsCard>

          <SettingsCard>
            <HotkeyEditor />
          </SettingsCard>

          <SettingsCard>
            <PreferenceRow
              title="Quit app"
              help="Stops the tray app and any hidden windows."
            >
              <button
                className="settings-button settings-button-danger"
                onClick={quitFromSettings}
              >
                Quit Screenie AI
              </button>
            </PreferenceRow>
          </SettingsCard>
        </SettingsSection>
      </main>
    </div>
  );
}

function modelForProvider(
  provider: Provider,
  anthropicModel: string,
  openaiModel: string,
  geminiModel: string,
  ollamaModel: string,
): string {
  if (provider === "anthropic") return anthropicModel;
  if (provider === "openai") return openaiModel;
  if (provider === "gemini") return geminiModel;
  return ollamaModel;
}

function getOllamaStatusLabel(
  checking: boolean,
  running: boolean,
  ready: boolean,
  hasVisionModel: boolean,
): string {
  if (checking && !running) return "Checking";
  if (ready) return "Ready";
  if (!running) return "Not installed";
  return hasVisionModel ? "Select vision model" : "Needs vision model";
}

function SettingsSection({
  id,
  active,
  title,
  description,
  children,
}: {
  id: SectionId;
  active: boolean;
  title: string;
  description: string;
  children: ReactNode;
}) {
  return (
    <section
      id={`settings-${id}`}
      className="settings-section"
      data-active={active}
      aria-hidden={!active}
    >
      <div className="settings-page-header">
        <div>
          <h2 className="settings-page-title">{title}</h2>
          <p className="settings-page-description">{description}</p>
        </div>
      </div>
      {children}
    </section>
  );
}

function SettingsCard({
  title,
  description,
  action,
  children,
}: {
  title?: string;
  description?: string;
  action?: ReactNode;
  children: ReactNode;
}) {
  const hasHeader = !!(title || description || action);
  return (
    <section className="settings-card">
      {hasHeader && (
        <div className="settings-card-header">
          <div>
            {title && <h3 className="settings-card-title">{title}</h3>}
            {description && (
              <p className="settings-card-description">{description}</p>
            )}
          </div>
          {action}
        </div>
      )}
      {children}
    </section>
  );
}

function PreferenceRow({
  title,
  help,
  children,
}: {
  title: string;
  help?: string;
  children: ReactNode;
}) {
  return (
    <div className="settings-row">
      <div>
        <p className="settings-row-label">{title}</p>
        {help && <p className="settings-row-help">{help}</p>}
      </div>
      <div className="settings-control">{children}</div>
    </div>
  );
}

function StatusPill({
  label,
  good = false,
  warning = false,
}: {
  label: string;
  good?: boolean;
  warning?: boolean;
}) {
  return (
    <span
      className="settings-status-pill"
      data-good={good}
      data-warning={warning}
    >
      {label}
    </span>
  );
}

function SegmentedControl<T extends string>({
  value,
  options,
  onChange,
  ariaLabel,
}: {
  value: T;
  options: { value: T; label: string }[];
  onChange: (value: T) => void;
  ariaLabel: string;
}) {
  const activeIndex = Math.max(
    0,
    options.findIndex((option) => option.value === value),
  );
  const segmentGap = 3;
  const indicatorStyle: CSSProperties = {
    width: `calc((100% - 6px - ${(options.length - 1) * segmentGap}px) / ${options.length})`,
    transform: `translateX(calc(${activeIndex * 100}% + ${
      activeIndex * segmentGap
    }px))`,
  };

  return (
    <div className="settings-segmented" role="group" aria-label={ariaLabel}>
      <span
        className="settings-segmented-indicator"
        style={indicatorStyle}
        aria-hidden
      />
      {options.map((option) => (
        <button
          key={option.value}
          className="settings-segmented-btn"
          data-active={value === option.value}
          aria-pressed={value === option.value}
          onClick={() => onChange(option.value)}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}

function ToggleControl({
  checked,
  onChange,
  ariaLabel,
}: {
  checked: boolean;
  onChange: (checked: boolean) => void;
  ariaLabel: string;
}) {
  return (
    <button
      className="settings-toggle"
      data-active={checked}
      role="switch"
      aria-label={ariaLabel}
      aria-checked={checked}
      onClick={() => onChange(!checked)}
    />
  );
}

function CloudProviderCard({
  title,
  keyTitle,
  keyValue,
  keyPlaceholder,
  keyWhere,
  onKeyChange,
  revealKey,
  setRevealKey,
  modelValue,
  modelOptions,
  onModelChange,
}: {
  title: string;
  keyTitle: string;
  keyValue: string;
  keyPlaceholder: string;
  keyWhere: string;
  onKeyChange: (value: string) => void;
  revealKey: boolean;
  setRevealKey: (fn: (value: boolean) => boolean) => void;
  modelValue: string;
  modelOptions: CustomDropdownOption[];
  onModelChange: (value: string) => void;
}) {
  return (
    <SettingsCard
      title={title}
      description="Stored API keys go to the system keychain and are never written to disk in plaintext."
      action={<StatusPill label={keyValue ? "Saved" : "Not set"} good={!!keyValue} />}
    >
      <PreferenceRow title={keyTitle} help={`Get one at ${keyWhere}.`}>
        <div className="settings-input-with-action">
          <input
            type={revealKey ? "text" : "password"}
            value={keyValue}
            placeholder={keyPlaceholder}
            onChange={(event) => onKeyChange(event.target.value)}
            className="settings-input"
            autoComplete="off"
            spellCheck={false}
          />
          <button
            type="button"
            className="settings-input-eye"
            onClick={() => setRevealKey((value) => !value)}
            aria-label={revealKey ? "Hide API key" : "Show API key"}
          >
            {revealKey ? (
              <EyeOff size={15} strokeWidth={1.75} aria-hidden />
            ) : (
              <Eye size={15} strokeWidth={1.75} aria-hidden />
            )}
          </button>
        </div>
      </PreferenceRow>
      <PreferenceRow title="Model" help="Used by both new captures and follow-up questions.">
        <CustomDropdown
          value={modelValue}
          options={modelOptions}
          onChange={onModelChange}
          ariaLabel={`${title} model`}
        />
      </PreferenceRow>
    </SettingsCard>
  );
}

/* ------------------------------------------------------------------ */
/* Templates editor                                                    */
/* ------------------------------------------------------------------ */

function TemplatesEditor() {
  const [list, setList] = useState<PromptTemplate[]>(() => readTemplates());
  // Inline two-click confirm for the destructive Restore action. `confirm()`
  // is unreliable in the Tauri WebView, and an inline pattern matches the
  // history archive flow already used in this app.
  const [resetPending, setResetPending] = useState(false);
  useEffect(() => subscribeTemplates(setList), []);

  // Auto-disarm the pending reset after 4s.
  useEffect(() => {
    if (!resetPending) return;
    const t = setTimeout(() => setResetPending(false), 4000);
    return () => clearTimeout(t);
  }, [resetPending]);

  // Click-anywhere-else dismissal — capture-phase so it can't be swallowed.
  // Skipped if the click landed on the pending button itself, so the second
  // click commits the reset.
  useEffect(() => {
    if (!resetPending) return;
    const onDown = (e: MouseEvent) => {
      const target = e.target;
      if (
        target instanceof HTMLElement &&
        target.closest('.settings-templates-reset[data-pending="true"]')
      ) {
        return;
      }
      setResetPending(false);
    };
    document.addEventListener("mousedown", onDown, true);
    return () => document.removeEventListener("mousedown", onDown, true);
  }, [resetPending]);

  const onResetClick = () => {
    if (resetPending) {
      resetTemplates();
      setResetPending(false);
    } else {
      setResetPending(true);
    }
  };

  return (
    <div className="settings-templates-section">
      <p className="settings-templates-hint">
        Click any label or prompt to edit — including the four built-in
        presets. Changes save automatically.
      </p>

      {list.length === 0 ? (
        <p className="settings-note">
          No templates yet — add one below or restore the defaults.
        </p>
      ) : (
        <ul className="settings-templates-list">
          {list.map((t) => (
            <TemplateRow
              key={t.id}
              template={t}
              onChange={(patch) => updateTemplate(t.id, patch)}
              onDelete={() => deleteTemplate(t.id)}
            />
          ))}
        </ul>
      )}

      <div className="settings-templates-footer">
        <button
          type="button"
          className="settings-templates-add"
          onClick={() =>
            addTemplate("New preset", "Describe what's shown in this image.")
          }
        >
          <Plus size={14} strokeWidth={2} aria-hidden />
          Add template
        </button>
        <button
          type="button"
          className="settings-templates-reset"
          data-pending={resetPending}
          onClick={onResetClick}
        >
          {resetPending ? "Confirm reset" : "Restore defaults"}
        </button>
      </div>
    </div>
  );
}

function TemplateRow({
  template,
  onChange,
  onDelete,
}: {
  template: PromptTemplate;
  onChange: (patch: Partial<Omit<PromptTemplate, "id">>) => void;
  onDelete: () => void;
}) {
  const [deletePending, setDeletePending] = useState(false);
  // Auto-grow the prompt textarea to fit its content. Resizing manually is
  // disabled (resize: none in CSS) — the textarea reads as flowing prose
  // on the page, not a form field.
  const promptRef = useRef<HTMLTextAreaElement>(null);
  useLayoutEffect(() => {
    const ta = promptRef.current;
    if (!ta) return;
    ta.style.height = "auto";
    ta.style.height = ta.scrollHeight + "px";
  }, [template.prompt]);
  useEffect(() => {
    if (!deletePending) return;
    const t = setTimeout(() => setDeletePending(false), 4000);
    return () => clearTimeout(t);
  }, [deletePending]);
  // Click-anywhere-else dismissal — same pattern used by the
  // TemplatesEditor's reset confirm. Capture-phase mousedown so it
  // can't be swallowed upstream; the pending button is excluded so
  // the second click on it actually commits the delete.
  useEffect(() => {
    if (!deletePending) return;
    const onDown = (e: MouseEvent) => {
      const target = e.target;
      if (
        target instanceof HTMLElement &&
        target.closest('.settings-template-delete[data-pending="true"]')
      ) {
        return;
      }
      setDeletePending(false);
    };
    document.addEventListener("mousedown", onDown, true);
    return () => document.removeEventListener("mousedown", onDown, true);
  }, [deletePending]);

  return (
    <li className="settings-template-row">
      <input
        className="settings-template-label"
        value={template.label}
        placeholder="Chip label"
        spellCheck={false}
        onChange={(e) => onChange({ label: e.target.value })}
      />
      <button
        type="button"
        className="settings-template-delete"
        data-pending={deletePending}
        onClick={() => {
          if (deletePending) {
            onDelete();
            setDeletePending(false);
          } else {
            setDeletePending(true);
          }
        }}
        aria-label={deletePending ? "Confirm delete template" : "Delete template"}
        title={deletePending ? "Confirm delete template" : "Delete template"}
      >
        {deletePending ? (
          "Confirm"
        ) : (
          <Trash2 size={14} strokeWidth={1.75} aria-hidden />
        )}
      </button>
      <textarea
        ref={promptRef}
        className="settings-template-prompt"
        value={template.prompt}
        placeholder="The prompt this chip sends with the captured image."
        rows={1}
        onChange={(e) => onChange({ prompt: e.target.value })}
      />
    </li>
  );
}

/* ------------------------------------------------------------------ */
/* History browser                                                     */
/* ------------------------------------------------------------------ */

function HistoryBrowser() {
  // Bumping `reloadKey` re-runs the HistoryList's listHistory effect, so
  // Clear-All wipes disk + visually empties the list without a manual
  // refresh hop.
  const [reloadKey, setReloadKey] = useState(0);
  const [clearPending, setClearPending] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const wipe = async () => {
    if (!clearPending) {
      setClearPending(true);
      return;
    }
    setError(null);
    try {
      await clearHistory();
      setClearPending(false);
      setReloadKey((k) => k + 1);
    } catch (e) {
      setError(`Couldn't clear history: ${toMessage(e)}`);
      setClearPending(false);
    }
  };
  useEffect(() => {
    if (!clearPending) return;
    const t = setTimeout(() => setClearPending(false), 4000);
    return () => clearTimeout(t);
  }, [clearPending]);
  // Click-anywhere-else dismissal — same pattern as the templates
  // reset confirm. Pending button is excluded so the second click
  // actually commits the wipe.
  useEffect(() => {
    if (!clearPending) return;
    const onDown = (e: MouseEvent) => {
      const target = e.target;
      if (
        target instanceof HTMLElement &&
        target.closest('.settings-history-clear[data-pending="true"]')
      ) {
        return;
      }
      setClearPending(false);
    };
    document.addEventListener("mousedown", onDown, true);
    return () => document.removeEventListener("mousedown", onDown, true);
  }, [clearPending]);

  // No SettingsCard chrome for this section — the page-level header already
  // says "History", and the user wanted just the list breathing on the
  // page background. A thin toolbar holds the disk-location hint + the
  // Clear All affordance.
  return (
    <div className="settings-history-section">
      <div className="settings-history-toolbar">
        <span className="settings-history-hint">
          Stored locally under{" "}
          <code>~/Library/Application Support/com.screenieai.app/history</code>.
        </span>
        <button
          className="settings-history-clear"
          data-pending={clearPending}
          onClick={wipe}
          aria-label={clearPending ? "Confirm clear all history" : "Clear all history"}
          title={clearPending ? "Confirm clear all history" : "Clear all history"}
        >
          {clearPending ? "Confirm clear" : <Trash2 size={15} strokeWidth={1.75} aria-hidden />}
        </button>
      </div>
      {error && (
        <p className="settings-note" role="alert" style={{ color: "var(--error-text)" }}>
          {error}
        </p>
      )}
      <HistoryList
        reloadKey={reloadKey}
        emptyState={
          <p className="settings-note">
            No captures yet. Press your capture hotkey to start filling this list.
          </p>
        }
        containerStyle={{
          display: "flex",
          flexDirection: "column",
          gap: 6,
          marginTop: 12,
        }}
      />
    </div>
  );
}

/* ------------------------------------------------------------------ */
/* Hotkey editor                                                       */
/* ------------------------------------------------------------------ */

type HotkeyConfigDto = {
  capture: string;
  repeat: string;
  settings: string;
};

function HotkeyEditor() {
  const [cfg, setCfg] = useState<HotkeyConfigDto | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [savingKey, setSavingKey] = useState<string | null>(null);

  useEffect(() => {
    invoke<HotkeyConfigDto>("get_hotkey_config")
      .then(setCfg)
      .catch((e) => setError(String(e)));
  }, []);

  const update = async (key: keyof HotkeyConfigDto, next: HotkeyConfigDto) => {
    setError(null);
    setSavingKey(key);
    try {
      await invoke("set_hotkey_config", next);
      setCfg(next);
    } catch (e) {
      setError(typeof e === "string" ? e : (e as Error).message ?? String(e));
    } finally {
      setSavingKey(null);
    }
  };

  if (!cfg) {
    return <p className="settings-note">Loading hotkeys…</p>;
  }

  return (
    <>
      <PreferenceRow title="Capture region" help="Open the screen-region selector.">
        <HotkeyRecorder
          value={cfg.capture}
          busy={savingKey === "capture"}
          onChange={(v) => update("capture", { ...cfg, capture: v })}
        />
      </PreferenceRow>
      <PreferenceRow title="Repeat last capture" help="Re-uses the previous rect on the same monitor — no drag step.">
        <HotkeyRecorder
          value={cfg.repeat}
          busy={savingKey === "repeat"}
          onChange={(v) => update("repeat", { ...cfg, repeat: v })}
        />
      </PreferenceRow>
      <PreferenceRow title="Open Settings" help="Bring this window forward from any app.">
        <HotkeyRecorder
          value={cfg.settings}
          busy={savingKey === "settings"}
          onChange={(v) => update("settings", { ...cfg, settings: v })}
        />
      </PreferenceRow>
      {error && (
        <p className="settings-note" style={{ color: "var(--error-text)" }}>
          {error}
        </p>
      )}
    </>
  );
}

function HotkeyRecorder({
  value,
  busy,
  onChange,
}: {
  value: string;
  busy: boolean;
  onChange: (v: string) => void;
}) {
  const [recording, setRecording] = useState(false);
  const [draft, setDraft] = useState<string | null>(null);
  const [hint, setHint] = useState<string | null>(null);

  useEffect(() => {
    if (!recording) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.preventDefault();
        setRecording(false);
        setDraft(null);
        setHint(null);
        return;
      }
      // Need at least one modifier + a non-modifier key.
      const mods: string[] = [];
      if (e.metaKey) mods.push("CommandOrControl");
      else if (e.ctrlKey) mods.push("CommandOrControl");
      if (e.altKey) mods.push("Alt");
      if (e.shiftKey) mods.push("Shift");
      const code = e.code;
      const isModifier =
        code === "MetaLeft" ||
        code === "MetaRight" ||
        code === "ControlLeft" ||
        code === "ControlRight" ||
        code === "AltLeft" ||
        code === "AltRight" ||
        code === "ShiftLeft" ||
        code === "ShiftRight";
      if (isModifier) {
        e.preventDefault();
        setHint("Now press a letter or symbol.");
        return;
      }
      if (mods.length === 0) {
        // Need at least one modifier. Show the recorded so-far so the user
        // sees what's happening.
        e.preventDefault();
        setHint("Use at least one modifier such as Command, Ctrl, Option, or Shift.");
        return;
      }
      e.preventDefault();
      const accel = [...mods, code].join("+");
      setDraft(accel);
      setRecording(false);
      setHint(null);
      onChange(accel);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [recording, onChange]);

  const display = draft ?? value;
  return (
    <div style={{ display: "flex", gap: 8, alignItems: "center" }}>
      <button
        type="button"
        className="settings-button"
        aria-pressed={recording}
        disabled={busy}
        onClick={() => {
          setDraft(null);
          setHint(null);
          setRecording((r) => !r);
        }}
        style={{ minWidth: 200, fontFamily: "ui-monospace, monospace", fontSize: 12.5 }}
      >
        {recording ? "Press a combination…" : busy ? "Saving…" : prettifyAccelerator(display)}
      </button>
      {hint && (
        <span
          className="settings-note"
          role="status"
          style={{ marginTop: 0, color: "var(--error-text)" }}
        >
          {hint}
        </span>
      )}
    </div>
  );
}

function prettifyAccelerator(accel: string): string {
  return accel
    .split("+")
    .map((part) => {
      if (part === "CommandOrControl") return navigator.userAgent.includes("Mac") ? "⌘" : "Ctrl";
      if (part === "Alt") return navigator.userAgent.includes("Mac") ? "⌥" : "Alt";
      if (part === "Shift") return "⇧";
      if (part === "Control") return "⌃";
      if (part.startsWith("Key")) return part.slice(3);
      if (part.startsWith("Digit")) return part.slice(5);
      if (part === "Comma") return ",";
      if (part === "Period") return ".";
      if (part === "Space") return "Space";
      return part;
    })
    .join(" ");
}

/* ------------------------------------------------------------------ */
/* Usage / cost overview                                                */
/* ------------------------------------------------------------------ */

const PROVIDER_DISPLAY: Record<string, string> = {
  anthropic: "Claude (Anthropic)",
  openai: "GPT (OpenAI)",
  gemini: "Gemini (Google)",
  ollama: "Ollama (local)",
};

function UsageOverview() {
  const [store, setStore] = useState<UsageStore>(() => readAllMonths());
  const [resetPending, setResetPending] = useState(false);
  useEffect(() => subscribeUsage(setStore), []);
  useEffect(() => {
    if (!resetPending) return;
    const t = setTimeout(() => setResetPending(false), 4000);
    return () => clearTimeout(t);
  }, [resetPending]);
  // Click-anywhere-else dismissal — see the matching pattern in
  // TemplatesEditor / HistoryBrowser / TemplateRow. Pending button
  // excluded so the second click commits the reset.
  useEffect(() => {
    if (!resetPending) return;
    const onDown = (e: MouseEvent) => {
      const target = e.target;
      if (
        target instanceof HTMLElement &&
        target.closest('.settings-usage-reset[data-pending="true"]')
      ) {
        return;
      }
      setResetPending(false);
    };
    document.addEventListener("mousedown", onDown, true);
    return () => document.removeEventListener("mousedown", onDown, true);
  }, [resetPending]);
  const monthKey = currentMonthKey();
  const month = store[monthKey] ?? {};
  const entries = Object.entries(month) as [string, ProviderTotals][];
  const totalCents = entries.reduce((acc, [, t]) => acc + t.costCents, 0);
  const totalCalls = entries.reduce((acc, [, t]) => acc + t.calls, 0);
  const months = Object.keys(store)
    .filter((k) => k !== monthKey)
    .sort()
    .reverse();

  return (
    <SettingsCard
      title="Usage this month"
      description={`Per-provider token + cost totals for ${formatMonthLabel(monthKey)}. Pricing is approximate; cloud providers' actual invoices are authoritative.`}
      action={
        entries.length > 0 ? (
          <span
            className="settings-status-pill settings-status-pill-light"
            data-good={false}
          >
            {formatCostCents(totalCents)} · {totalCalls} calls
          </span>
        ) : undefined
      }
    >
      {entries.length === 0 ? (
        <p className="settings-note">
          No AI calls this month yet. Token + cost will accumulate here as you
          send queries.
        </p>
      ) : (
        <div
          style={{
            display: "grid",
            gridTemplateColumns: "1fr 1fr 1fr 1fr",
            gap: 8,
            fontSize: 12.5,
          }}
        >
          <div className="settings-note" style={{ marginTop: 0, fontWeight: 600 }}>
            Provider
          </div>
          <div className="settings-note" style={{ marginTop: 0, fontWeight: 600 }}>
            Tokens
          </div>
          <div className="settings-note" style={{ marginTop: 0, fontWeight: 600 }}>
            Calls
          </div>
          <div className="settings-note" style={{ marginTop: 0, fontWeight: 600 }}>
            Cost
          </div>
          {entries
            .slice()
            .sort((a, b) => b[1].costCents - a[1].costCents)
            .map(([provider, t]) => (
              <Fragment key={provider}>
                <div>{PROVIDER_DISPLAY[provider] ?? provider}</div>
                <div>
                  {formatTokens(t.inputTokens)} in · {formatTokens(t.outputTokens)} out
                </div>
                <div>{t.calls}</div>
                <div>{formatCostCents(t.costCents)}</div>
              </Fragment>
            ))}
        </div>
      )}
      {months.length > 0 && (
        <details style={{ marginTop: 14 }}>
          <summary
            style={{
              cursor: "pointer",
              fontSize: 12,
              color: "var(--ink-3)",
            }}
          >
            Earlier months ({months.length})
          </summary>
          <div style={{ marginTop: 8, display: "flex", flexDirection: "column", gap: 6 }}>
            {months.map((key) => {
              const m = store[key];
              const c = Object.values(m).reduce((acc, t) => acc + t.costCents, 0);
              const calls = Object.values(m).reduce((acc, t) => acc + t.calls, 0);
              return (
                <div
                  key={key}
                  style={{
                    display: "flex",
                    justifyContent: "space-between",
                    fontSize: 12,
                    color: "var(--ink-2)",
                  }}
                >
                  <span>{formatMonthLabel(key)}</span>
                  <span>
                    {formatCostCents(c)} · {calls} calls
                  </span>
                </div>
              );
            })}
          </div>
        </details>
      )}
      <div style={{ marginTop: 14 }}>
        <button
          className="settings-button settings-usage-reset"
          data-pending={resetPending}
          onClick={() => {
            if (!resetPending) {
              setResetPending(true);
              return;
            }
            clearAllUsage();
            setResetPending(false);
          }}
          style={{ fontSize: 11.5, padding: "4px 10px" }}
        >
          {resetPending ? "Confirm reset" : "Reset usage tracker"}
        </button>
      </div>
    </SettingsCard>
  );
}

function formatMonthLabel(monthKey: string): string {
  const [year, month] = monthKey.split("-").map((s) => parseInt(s, 10));
  if (!year || !month) return monthKey;
  const d = new Date(year, month - 1, 1);
  return d.toLocaleString(undefined, { month: "long", year: "numeric" });
}

function toMessage(e: unknown): string {
  if (typeof e === "string") return e;
  if (e instanceof Error) return e.message;
  return String(e);
}
