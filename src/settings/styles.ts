import React from "react";

// All settings styles read from CSS variables defined in App.css so the
// panel adapts to the OS light/dark theme. Variable names match the
// onboarding scoped tokens, so cross-view styling stays consistent.

export const inputStyle: React.CSSProperties = {
  flex: 1,
  padding: "9px 12px",
  borderRadius: 8,
  border: "1px solid var(--border)",
  fontSize: 13,
  fontFamily: "ui-monospace, SFMono-Regular, Menlo, Monaco, monospace",
  background: "var(--bg)",
  color: "var(--ink)",
  outline: "none",
};

export const smallBtn: React.CSSProperties = {
  padding: "8px 12px",
  fontSize: 12,
  borderRadius: 8,
  background: "var(--surface)",
  color: "var(--ink)",
  cursor: "pointer",
  fontFamily: "inherit",
};

export const hint: React.CSSProperties = {
  fontSize: 11.5,
  color: "var(--ink-3)",
  marginTop: 10,
  marginBottom: 0,
  lineHeight: 1.55,
};

export const primaryBtn: React.CSSProperties = {
  padding: "8px 14px",
  fontSize: 12.5,
  fontWeight: 500,
  borderRadius: 8,
  background: "var(--primary-bg)",
  color: "var(--primary-text)",
  cursor: "pointer",
  fontFamily: "inherit",
};

export const stepRow: React.CSSProperties = {
  display: "flex",
  gap: 12,
  alignItems: "flex-start",
};

export const stepIndex: React.CSSProperties = {
  width: 22,
  height: 22,
  minWidth: 22,
  borderRadius: "50%",
  background: "var(--section-bg)",
  fontSize: 11.5,
  fontWeight: 600,
  display: "flex",
  alignItems: "center",
  justifyContent: "center",
  color: "var(--ink)",
  marginTop: 1,
};

export const stepTitle: React.CSSProperties = {
  fontSize: 13,
  fontWeight: 600,
  color: "var(--ink)",
};

export const stepBody: React.CSSProperties = {
  fontSize: 12,
  color: "var(--ink-2)",
  marginTop: 2,
  lineHeight: 1.5,
};

export const quitBtn: React.CSSProperties = {
  padding: "8px 18px",
  fontSize: 12,
  fontFamily: "inherit",
  color: "var(--ink-3)",
  background: "transparent",
  border: "1px solid var(--border)",
  borderRadius: 999,
  cursor: "pointer",
};

export const progressTrack: React.CSSProperties = {
  width: "100%",
  height: 6,
  borderRadius: 999,
  background: "var(--section-bg)",
  overflow: "hidden",
};

export const progressFill: React.CSSProperties = {
  height: "100%",
  background: "var(--primary-bg)",
  borderRadius: 999,
  transition: "width 200ms ease",
};
