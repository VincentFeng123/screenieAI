export const heroHeadline: string[] = [
  "A precise AI lens",
  "for your screen.",
];

export const heroEyebrow = "macOS · Menu bar";

export const heroSubhead =
  "Press a hotkey. Drag any region. Ask anything. Screenie returns the answer in a floating window — without leaving the app you're in.";

export const heroPrompt = "What's on screen?";

export const heroAnswer = `That's a Pareto frontier with three plotted models. The frontier line is monotonically non-decreasing — \`gpt-class\` sits above it, suggesting a recent step change in cost-vs-accuracy.`;

export const featureItems: { icon: string; title: string; body: string }[] = [
  {
    icon: "MousePointer",
    title: "Drag-to-capture, anywhere",
    body: "A native overlay snaps to the region you draw. Same hotkey across spaces, full-screen apps, and external displays.",
  },
  {
    icon: "Eye",
    title: "Vision-grade models",
    body: "Pluggable providers: Claude, GPT, Gemini, and local Ollama. Swap on the fly — your last choice persists per session.",
  },
  {
    icon: "Lock",
    title: "Local privacy mode",
    body: "Run vision models on-device through Ollama. Captures never leave your Mac. A pill in the title bar tells you which mode you're in.",
  },
  {
    icon: "Key",
    title: "Keys in the keychain",
    body: "API tokens are stored in the macOS Keychain. They're never written to disk in plaintext, never sent to a server we control.",
  },
  {
    icon: "PenLine",
    title: "Annotate before sending",
    body: "An edit canvas lets you draw, highlight, and crop further. The model only sees what you intended.",
  },
  {
    icon: "History",
    title: "History you control",
    body: "Past captures and answers are kept locally. Export, search, or wipe — one keystroke, no cloud round-trip.",
  },
];

export const workflowNodes = [
  { id: "hotkey", label: "Hotkey", caption: "Global shortcut" },
  { id: "capture", label: "Capture", caption: "Native region" },
  { id: "encode", label: "Encode", caption: "Edit · compress" },
  { id: "provider", label: "Provider", caption: "Cloud or Local" },
  { id: "render", label: "Render", caption: "Floating window" },
];

export const providers = [
  { name: "Anthropic", model: "Claude · Vision", mode: "cloud" as const },
  { name: "OpenAI", model: "GPT · Vision", mode: "cloud" as const },
  { name: "Google", model: "Gemini · Vision", mode: "cloud" as const },
  { name: "Ollama", model: "On-device", mode: "local" as const },
];
