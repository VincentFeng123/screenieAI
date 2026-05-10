import React from "react";
import ReactDOM from "react-dom/client";
import App from "./App";
import Chat from "./Chat";
import Overlay from "./Overlay";

const params = new URLSearchParams(window.location.search);
const hashMode = window.location.hash.replace(/^#/, "");
const mode = params.get("mode") || hashMode;

if (mode === "overlay") {
  document.documentElement.style.background = "transparent";
  document.body.style.background = "transparent";
  document.body.style.margin = "0";
  document.body.style.overflow = "hidden";
  document.body.style.userSelect = "none";
  document.body.style.color = "#fff";
  document.body.style.cursor = "default";
  document.documentElement.style.cursor = "default";
  const root = document.getElementById("root") as HTMLElement;
  if (root) {
    root.style.background = "transparent";
    root.style.cursor = "default";
  }
}

if (mode === "chat") {
  // Transparent body so the chat-panel's BlurredBackdrop fills the window
  // edge-to-edge (the panel itself provides the frosted base + tint).
  document.body.style.margin = "0";
  document.body.style.background = "transparent";
  document.documentElement.style.background = "transparent";
  document.body.style.overflow = "hidden";
}

// Overlay (and Chat) render without StrictMode because they rely on one-shot
// side-effects (consuming the pending capture / chat seed from Rust state).
// StrictMode's dev-only double-invocation of effects would consume the
// data twice and break first-frame rendering.
ReactDOM.createRoot(document.getElementById("root") as HTMLElement).render(
  mode === "overlay" ? (
    <Overlay />
  ) : mode === "chat" ? (
    <Chat />
  ) : (
    <React.StrictMode>
      <App />
    </React.StrictMode>
  ),
);
