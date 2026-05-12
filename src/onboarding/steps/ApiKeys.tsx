import { useEffect, useMemo, useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import { Eye, EyeOff } from "lucide-react";
import { debounce } from "../../lib/debounce";

type CloudProvider = "anthropic" | "openai" | "gemini";

export default function ApiKeys({
  onNext,
  onBack,
  onSkip,
}: {
  onNext: (hasCloudKey: boolean) => void;
  onBack: () => void;
  onSkip: () => void;
}) {
  const [anthropicKey, setAnthropicKey] = useState("");
  const [openaiKey, setOpenaiKey] = useState("");
  const [geminiKey, setGeminiKey] = useState("");
  const [reveal, setReveal] = useState(false);
  const [error, setError] = useState<string | null>(null);
  // Active provider — auto-picks the first key the user enters, but the user
  // can flip it after entering multiple keys (previously they had to go to
  // Settings → Providers afterwards to switch).
  const [activeProvider, setActiveProvider] = useState<CloudProvider | null>(null);
  const [activeProviderTouched, setActiveProviderTouched] = useState(false);
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    (async () => {
      try {
        const a = await invoke<string | null>("get_secret", { name: "anthropic_api_key" });
        if (a) setAnthropicKey(a);
      } catch {
        /* keyring not reachable */
      }
      try {
        const o = await invoke<string | null>("get_secret", { name: "openai_api_key" });
        if (o) setOpenaiKey(o);
      } catch {
        /* keyring not reachable */
      }
      try {
        const g = await invoke<string | null>("get_secret", { name: "gemini_api_key" });
        if (g) setGeminiKey(g);
      } catch {
        /* keyring not reachable */
      }
    })();
  }, []);

  // Debounced keychain writer — see SettingsPanel for the rationale.
  const debouncedSave = useMemo(
    () =>
      debounce((name: string, val: string) => {
        const op = val
          ? invoke("set_secret", { name, value: val })
          : invoke("delete_secret", { name });
        op.catch((e) => {
          const msg = typeof e === "string" ? e : (e as Error).message ?? String(e);
          setError(`Couldn't save to the system keychain: ${msg}`);
        });
      }, 350),
    [],
  );

  useEffect(() => () => debouncedSave.flush(), [debouncedSave]);

  const saveSecret = (name: string, val: string) => {
    setError(null);
    debouncedSave(name, val);
  };

  const onChangeAnthropic = (v: string) => {
    setAnthropicKey(v);
    saveSecret("anthropic_api_key", v);
  };
  const onChangeOpenai = (v: string) => {
    setOpenaiKey(v);
    saveSecret("openai_api_key", v);
  };
  const onChangeGemini = (v: string) => {
    setGeminiKey(v);
    saveSecret("gemini_api_key", v);
  };

  const hasAnthropic = anthropicKey.trim().length > 0;
  const hasOpenai = openaiKey.trim().length > 0;
  const hasGemini = geminiKey.trim().length > 0;

  // Auto-pick the active provider while the user hasn't manually chosen.
  // Order: Anthropic > OpenAI > Gemini.
  const autoActive: CloudProvider | null = hasAnthropic
    ? "anthropic"
    : hasOpenai
      ? "openai"
      : hasGemini
        ? "gemini"
        : null;
  const effectiveActive: CloudProvider | null = activeProviderTouched
    ? activeProvider
    : autoActive;

  const writeSecretNow = async (name: string, val: string) => {
    if (val) await invoke("set_secret", { name, value: val });
    else await invoke("delete_secret", { name });
  };

  const handleContinue = async () => {
    setSaving(true);
    setError(null);
    debouncedSave.cancel();
    try {
      await Promise.all([
        writeSecretNow("anthropic_api_key", anthropicKey),
        writeSecretNow("openai_api_key", openaiKey),
        writeSecretNow("gemini_api_key", geminiKey),
      ]);
      if (effectiveActive) localStorage.setItem("provider", effectiveActive);
      onNext(hasAnthropic || hasOpenai || hasGemini);
    } catch (e) {
      const msg = typeof e === "string" ? e : (e as Error).message ?? String(e);
      setError(`Couldn't save to the system keychain: ${msg}`);
    } finally {
      setSaving(false);
    }
  };

  const handleSkip = () => {
    debouncedSave.cancel();
    onSkip();
  };

  const pickProvider = (id: CloudProvider) => {
    setActiveProvider(id);
    setActiveProviderTouched(true);
  };

  const eyeAria = reveal ? "Hide API key" : "Show API key";
  const toggleReveal = () => setReveal((v) => !v);

  return (
    <div className="onboarding-step-inner">
      <span className="onboarding-eyebrow">Cloud providers</span>

      <h1 className="onboarding-h1">Add a cloud key.</h1>

      <p className="onboarding-subtitle">
        Optional — skip if you'd rather run locally with Ollama. Keys are
        stored in your system keychain, never written to disk in plaintext.
      </p>

      <div className="onboarding-fields">
        <div className="onboarding-field">
          <label className="onboarding-label">
            <span>Anthropic · Claude</span>
          </label>
          <div className="onboarding-input-row">
            <input
              type={reveal ? "text" : "password"}
              value={anthropicKey}
              placeholder="sk-ant-..."
              onChange={(e) => onChangeAnthropic(e.target.value)}
              className="onboarding-input"
              autoComplete="off"
              spellCheck={false}
            />
            <button
              type="button"
              className="onboarding-input-eye"
              onClick={toggleReveal}
              aria-label={eyeAria}
            >
              {reveal ? (
                <EyeOff size={16} strokeWidth={1.75} aria-hidden />
              ) : (
                <Eye size={16} strokeWidth={1.75} aria-hidden />
              )}
            </button>
          </div>
        </div>

        <div className="onboarding-field">
          <label className="onboarding-label">
            <span>OpenAI · GPT</span>
          </label>
          <div className="onboarding-input-row">
            <input
              type={reveal ? "text" : "password"}
              value={openaiKey}
              placeholder="sk-..."
              onChange={(e) => onChangeOpenai(e.target.value)}
              className="onboarding-input"
              autoComplete="off"
              spellCheck={false}
            />
            <button
              type="button"
              className="onboarding-input-eye"
              onClick={toggleReveal}
              aria-label={eyeAria}
            >
              {reveal ? (
                <EyeOff size={16} strokeWidth={1.75} aria-hidden />
              ) : (
                <Eye size={16} strokeWidth={1.75} aria-hidden />
              )}
            </button>
          </div>
        </div>

        <div className="onboarding-field">
          <label className="onboarding-label">
            <span>Google · Gemini</span>
          </label>
          <div className="onboarding-input-row">
            <input
              type={reveal ? "text" : "password"}
              value={geminiKey}
              placeholder="AIza..."
              onChange={(e) => onChangeGemini(e.target.value)}
              className="onboarding-input"
              autoComplete="off"
              spellCheck={false}
            />
            <button
              type="button"
              className="onboarding-input-eye"
              onClick={toggleReveal}
              aria-label={eyeAria}
            >
              {reveal ? (
                <EyeOff size={16} strokeWidth={1.75} aria-hidden />
              ) : (
                <Eye size={16} strokeWidth={1.75} aria-hidden />
              )}
            </button>
          </div>
        </div>
      </div>

      {/* When the user has entered more than one key, surface a radio so they
          can pick which provider gets activated on Continue. With one key
          (or zero), this row collapses — the auto-pick is fine. */}
      {[hasAnthropic, hasOpenai, hasGemini].filter(Boolean).length > 1 && (
        <div className="onboarding-active-row" role="radiogroup" aria-label="Active provider">
          <span className="onboarding-label" style={{ marginBottom: 4 }}>
            Use this provider on first capture
          </span>
          <div style={{ display: "flex", gap: 8, flexWrap: "wrap" }}>
            {hasAnthropic && (
              <button
                type="button"
                role="radio"
                aria-checked={effectiveActive === "anthropic"}
                className={
                  "onboarding-link" +
                  (effectiveActive === "anthropic" ? " active" : "")
                }
                onClick={() => pickProvider("anthropic")}
              >
                Claude
              </button>
            )}
            {hasOpenai && (
              <button
                type="button"
                role="radio"
                aria-checked={effectiveActive === "openai"}
                className={
                  "onboarding-link" +
                  (effectiveActive === "openai" ? " active" : "")
                }
                onClick={() => pickProvider("openai")}
              >
                OpenAI
              </button>
            )}
            {hasGemini && (
              <button
                type="button"
                role="radio"
                aria-checked={effectiveActive === "gemini"}
                className={
                  "onboarding-link" +
                  (effectiveActive === "gemini" ? " active" : "")
                }
                onClick={() => pickProvider("gemini")}
              >
                Gemini
              </button>
            )}
          </div>
        </div>
      )}

      {error && <div className="onboarding-error">{error}</div>}

      <div className="onboarding-actions">
        <button className="onboarding-link back" onClick={onBack}>
          <span className="arrow" aria-hidden>←</span>
          Back
        </button>
        <div className="onboarding-actions-right">
          <button className="onboarding-link" onClick={handleSkip} disabled={saving}>
            Use Ollama instead
          </button>
          <button className="onboarding-btn primary" onClick={handleContinue} disabled={saving}>
            {saving ? "Saving…" : "Continue"}
            <span className="arrow" aria-hidden>→</span>
          </button>
        </div>
      </div>
    </div>
  );
}
