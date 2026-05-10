# Screenie AI Landing Page Design Skill

You are designing and implementing a premium landing page for **Screenie AI**, a macOS-first desktop AI app.

Screenie AI lets users press a hotkey, drag-select any region of their screen, and ask a vision-capable AI about what they captured. The answer appears instantly in a floating result/chat window. It supports cloud AI providers and local privacy-first models.

Your job is to create a **unique, minimal, black-and-white, techy landing page** with polished GSAP animations, strong visual direction, and production-quality frontend code.

---

## Product Context

Screenie AI is a desktop app for:

- Capturing any screen region with a hotkey
- Asking AI about the selected image
- Receiving the answer in a floating desktop window
- Keeping a history of captures and answers
- Editing/annotating screenshots before sending
- Choosing between cloud providers and local Ollama models
- Storing API keys safely in the OS keyring

Tech stack:

- Tauri 2
- Rust backend
- React 19
- TypeScript
- Vite
- macOS-first
- Windows-second
- Vision AI providers:
  - Anthropic
  - OpenAI
  - Gemini
  - Ollama

Important internal app pieces:

- `src-tauri/src/capture/` — native screen-region capture
- `src-tauri/src/ai/` — modular AI provider layer
- `src-tauri/src/secrets.rs` — OS keyring API key storage
- `src-tauri/src/ollama_install.rs` — local model setup
- `src/settings/OllamaSetup.tsx` — guided Ollama setup
- `src-tauri/src/history.rs` — capture/answer history
- `src/components/HistoryList.tsx` — history UI
- `src/Overlay.tsx` — drag-to-select overlay
- `src/Chat.tsx` — floating result/chat window
- `src/components/EditCanvas.tsx` — screenshot annotation/editing
- `src/onboarding/` — first-run onboarding
- `src/settings/` — provider tiles, hotkeys, preferences
- `screenAI-landing/` — marketing site location

---

# Core Design Direction

The site should feel like:

> A precise AI lens for your screen.

Visual identity:

- Minimal
- Black and white
- Premium
- Technical
- Mac-native
- Slightly futuristic
- Calm, sharp, and focused
- Not colorful
- Not generic SaaS
- Not corporate
- Not overdecorated

The page should feel closer to:

- Linear
- Raycast
- Vercel
- x.ai
- Apple Pro app pages
- Arc Browser
- Notion Calendar
- Teenage Engineering minimalism

Avoid:

- Generic blue/purple gradients
- Emoji-heavy design
- Cartoon illustrations
- Stock photos
- Loud neon cyberpunk
- Overused AI sparkle visuals
- Fake 3D blobs
- Corporate dashboard clichés

---

# Color System

Use a strict black-and-white system.

Preferred palette:

```css
:root {
  --bg: #030303;
  --bg-soft: #080808;
  --panel: rgba(255, 255, 255, 0.045);
  --panel-strong: rgba(255, 255, 255, 0.075);
  --border: rgba(255, 255, 255, 0.12);
  --border-strong: rgba(255, 255, 255, 0.24);
  --text: #f5f5f5;
  --text-muted: rgba(255, 255, 255, 0.62);
  --text-faint: rgba(255, 255, 255, 0.38);
  --white: #ffffff;
  --black: #000000;
  --glow: rgba(255, 255, 255, 0.18);
  --scanline: rgba(255, 255, 255, 0.08);
  --selection: rgba(255, 255, 255, 0.16);
}