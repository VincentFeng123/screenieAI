import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// @ts-expect-error process is a nodejs global
const host = process.env.TAURI_DEV_HOST;

// https://vite.dev/config/
export default defineConfig(async () => ({
  plugins: [react()],

  // Vite options tailored for Tauri development and only applied in `tauri dev` or `tauri build`
  //
  // 1. prevent Vite from obscuring rust errors
  clearScreen: false,
  // 2. tauri expects a fixed port, fail if that port is not available
  server: {
    port: 1420,
    strictPort: true,
    host: host || false,
    hmr: host
      ? {
          protocol: "ws",
          host,
          port: 1421,
        }
      : undefined,
    watch: {
      // 3. tell Vite to ignore watching `src-tauri`
      ignored: ["**/src-tauri/**"],
    },
  },

  // P-E: split the chunky markdown/math/highlight chunks out of the main
  // entry. Each chunk is independently parsed and cached by the WebView, so
  // a single 1 MB monolith becomes several smaller parses on first paint
  // (and the Settings window's React/Tauri code path no longer waits on a
  // ~280 KB KaTeX parse it never uses). Still ships every chunk on first
  // load — full lazy loading is a follow-up that requires `React.lazy`
  // around the markdown component.
  build: {
    rollupOptions: {
      output: {
        manualChunks: {
          katex: ["katex", "rehype-katex", "remark-math"],
          highlight: ["highlight.js", "rehype-highlight"],
          markdown: ["react-markdown", "remark-gfm"],
        },
      },
    },
  },
}));
