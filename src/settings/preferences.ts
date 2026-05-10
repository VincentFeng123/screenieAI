export type ThemePreference = "system" | "light" | "dark";
export type SettingsDensityPreference = "comfortable" | "compact";
export type OverlayChatDefaultPreference = "auto" | "open" | "hidden";
export type AiResponseStylePreference = "concise" | "balanced" | "detailed";
export type AiRenderDensityPreference = "compact" | "comfortable" | "spacious";

export type ScreeniePreferences = {
  theme: ThemePreference;
  settingsDensity: SettingsDensityPreference;
  overlayChatDefault: OverlayChatDefaultPreference;
  overlayShowPresets: boolean;
  overlayCloseOnBackdrop: boolean;
  overlayAllowEmptySend: boolean;
  aiResponseStyle: AiResponseStylePreference;
  aiRenderDensity: AiRenderDensityPreference;
};

export const PREFERENCE_EVENT = "screenie-preferences-changed";

export const DEFAULT_PREFERENCES: ScreeniePreferences = {
  theme: "system",
  settingsDensity: "comfortable",
  overlayChatDefault: "auto",
  overlayShowPresets: true,
  overlayCloseOnBackdrop: true,
  overlayAllowEmptySend: true,
  aiResponseStyle: "concise",
  aiRenderDensity: "comfortable",
};

const STORAGE_KEYS: Record<keyof ScreeniePreferences, string> = {
  theme: "screenie.pref.theme",
  settingsDensity: "screenie.pref.settingsDensity",
  overlayChatDefault: "screenie.pref.overlayChatDefault",
  overlayShowPresets: "screenie.pref.overlayShowPresets",
  overlayCloseOnBackdrop: "screenie.pref.overlayCloseOnBackdrop",
  overlayAllowEmptySend: "screenie.pref.overlayAllowEmptySend",
  aiResponseStyle: "screenie.pref.aiResponseStyle",
  aiRenderDensity: "screenie.pref.aiRenderDensity",
};

const THEME_VALUES: readonly ThemePreference[] = ["system", "light", "dark"];
const SETTINGS_DENSITY_VALUES: readonly SettingsDensityPreference[] = [
  "comfortable",
  "compact",
];
const OVERLAY_CHAT_VALUES: readonly OverlayChatDefaultPreference[] = [
  "auto",
  "open",
  "hidden",
];
const AI_RESPONSE_STYLE_VALUES: readonly AiResponseStylePreference[] = [
  "concise",
  "balanced",
  "detailed",
];
const AI_RENDER_DENSITY_VALUES: readonly AiRenderDensityPreference[] = [
  "compact",
  "comfortable",
  "spacious",
];

type PreferenceName = keyof ScreeniePreferences;

function storage(): Storage | null {
  if (typeof window === "undefined") return null;
  try {
    return window.localStorage;
  } catch {
    return null;
  }
}

function readEnum<T extends string>(
  key: string,
  allowed: readonly T[],
  fallback: T,
): T {
  const raw = storage()?.getItem(key);
  return raw && allowed.includes(raw as T) ? (raw as T) : fallback;
}

function readBoolean(key: string, fallback: boolean): boolean {
  const raw = storage()?.getItem(key);
  if (raw === null || raw === undefined) return fallback;
  if (raw === "1" || raw === "true") return true;
  if (raw === "0" || raw === "false") return false;
  return fallback;
}

export function readPreferences(): ScreeniePreferences {
  return {
    theme: readEnum(
      STORAGE_KEYS.theme,
      THEME_VALUES,
      DEFAULT_PREFERENCES.theme,
    ),
    settingsDensity: readEnum(
      STORAGE_KEYS.settingsDensity,
      SETTINGS_DENSITY_VALUES,
      DEFAULT_PREFERENCES.settingsDensity,
    ),
    overlayChatDefault: readEnum(
      STORAGE_KEYS.overlayChatDefault,
      OVERLAY_CHAT_VALUES,
      DEFAULT_PREFERENCES.overlayChatDefault,
    ),
    overlayShowPresets: readBoolean(
      STORAGE_KEYS.overlayShowPresets,
      DEFAULT_PREFERENCES.overlayShowPresets,
    ),
    overlayCloseOnBackdrop: readBoolean(
      STORAGE_KEYS.overlayCloseOnBackdrop,
      DEFAULT_PREFERENCES.overlayCloseOnBackdrop,
    ),
    overlayAllowEmptySend: readBoolean(
      STORAGE_KEYS.overlayAllowEmptySend,
      DEFAULT_PREFERENCES.overlayAllowEmptySend,
    ),
    aiResponseStyle: readEnum(
      STORAGE_KEYS.aiResponseStyle,
      AI_RESPONSE_STYLE_VALUES,
      DEFAULT_PREFERENCES.aiResponseStyle,
    ),
    aiRenderDensity: readEnum(
      STORAGE_KEYS.aiRenderDensity,
      AI_RENDER_DENSITY_VALUES,
      DEFAULT_PREFERENCES.aiRenderDensity,
    ),
  };
}

export function readPreference<K extends PreferenceName>(
  name: K,
): ScreeniePreferences[K] {
  return readPreferences()[name];
}

export function writePreference<K extends PreferenceName>(
  name: K,
  value: ScreeniePreferences[K],
): void {
  const store = storage();
  if (!store) return;
  store.setItem(
    STORAGE_KEYS[name],
    typeof value === "boolean" ? (value ? "1" : "0") : value,
  );

  if (name === "theme") applyThemePreference(value as ThemePreference);
  if (name === "settingsDensity") {
    applySettingsDensity(value as SettingsDensityPreference);
  }

  window.dispatchEvent(
    new CustomEvent(PREFERENCE_EVENT, { detail: readPreferences() }),
  );
}

export function subscribePreferences(
  listener: (preferences: ScreeniePreferences) => void,
): () => void {
  if (typeof window === "undefined") return () => {};

  const notify = () => listener(readPreferences());
  const onPreferenceEvent = (event: Event) => {
    const detail = (event as CustomEvent<ScreeniePreferences>).detail;
    listener(detail ?? readPreferences());
  };

  window.addEventListener("storage", notify);
  window.addEventListener(PREFERENCE_EVENT, onPreferenceEvent);
  return () => {
    window.removeEventListener("storage", notify);
    window.removeEventListener(PREFERENCE_EVENT, onPreferenceEvent);
  };
}

export function applyThemePreference(
  theme: ThemePreference = readPreference("theme"),
): void {
  if (typeof document === "undefined") return;
  const root = document.documentElement;
  root.dataset.themePreference = theme;
  if (theme === "system") {
    delete root.dataset.theme;
    root.style.colorScheme = "";
  } else {
    root.dataset.theme = theme;
    root.style.colorScheme = theme;
  }
}

export function applySettingsDensity(
  density: SettingsDensityPreference = readPreference("settingsDensity"),
): void {
  if (typeof document === "undefined") return;
  document.documentElement.dataset.settingsDensity = density;
}

export function applyStoredPreferences(): void {
  const preferences = readPreferences();
  applyThemePreference(preferences.theme);
  applySettingsDensity(preferences.settingsDensity);
}
