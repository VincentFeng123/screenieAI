/* ------------------------------------------------------------------ */
/* BlurredBackdrop — platform-split frost recipe                       */
/*                                                                     */
/* macOS: returns null. Frost is drawn by an NSVisualEffectView        */
/* mounted as a sibling of the WKWebView (see macos_window.m's         */
/* vibrancy section + `useOverlayFrostRegions` in Overlay.tsx). The    */
/* native compositor blurs the live desktop every frame.               */
/*                                                                     */
/* Windows: renders a 3-layer stack inside the panel —                 */
/*   1. Blurred screenshot bitmap, positioned so the panel shows the   */
/*      part of the screen behind it.                                  */
/*   2. Tint color overlay.                                            */
/*   3. Fill color overlay.                                            */
/* The overlay window has no Mica (it's borderless + transparent +     */
/* topmost), and CSS `backdrop-filter: blur()` over a transparent      */
/* topmost window has nothing in the compositor stack to blur. So we   */
/* fall back to the original v1 bitmap-blur recipe: the screenshot     */
/* refresh path (`overlay-background-changed` → React reload) keeps    */
/* this bitmap in step with the actual desktop content underneath.     */
/* The detached Chat window doesn't need this — its window gets Mica   */
/* applied via DWM in `windows_window::configure_main_window`.         */
/* ------------------------------------------------------------------ */

import { useEffect, useRef } from "react";
import { listen } from "@tauri-apps/api/event";

const isWindowsPlatform =
  typeof navigator !== "undefined" && navigator.userAgent.includes("Windows");

export function BlurredBackdrop(props: {
  src: string;
  screenW: number;
  screenH: number;
  blurRadius?: number;
  zIndex?: number;
  imageBrightness?: number;
  tint?: string;
  fill?: string;
  persistImage?: boolean;
}) {
  // Mac path: native vibrancy does the work, this component renders nothing.
  if (!isWindowsPlatform) return null;
  return <WindowsBlurredBackdrop {...props} />;
}

// Caller blur radii are tuned for the macOS NSVisualEffectView path (which
// applies its own ~50-80px-equivalent blur on top in the system compositor).
// On Windows we render the same blur via CSS `filter: blur()` over a
// downsampled-and-pre-blurred screenshot bitmap pushed from Rust at 60fps.
//
// IMPORTANT: Chromium silently caps or downgrades `filter: blur()` past
// ~40px on layered/transparent windows when the compositor can't promote
// the element to GPU. With the previous 2.4× scale (effective 60–80px),
// the blur visibly stopped rendering on Win11 22H2+ WebView2 — the
// screenshot showed through faintly but completely sharp. Keeping the CSS
// blur in the 20–32px range, combined with a Gaussian pre-blur applied
// inside `windows_window::capture_continuous_backdrop`, produces the
// same visual heft without tripping Chromium's filter limit. `will-change`
// + `transform: translateZ(0)` below force GPU compositing, which is the
// other half of making this render reliably.
const WIN_BLUR_SCALE = 1.0;
const WIN_BLUR_MAX = 32;
// Saturation boost mirrors the slight saturation lift NSVisualEffectMaterial.sidebar
// applies — without it the static-bitmap blur reads as a gray mush instead of
// faintly tinted glass.
const WIN_SATURATION = 1.4;

function WindowsBlurredBackdrop({
  src,
  screenW,
  screenH,
  blurRadius = 24,
  imageBrightness = 0.7,
  tint = "rgba(34, 36, 35, 0.43)",
  fill = "rgba(18, 19, 18, 0.17)",
}: {
  src: string;
  screenW: number;
  screenH: number;
  blurRadius?: number;
  zIndex?: number;
  imageBrightness?: number;
  tint?: string;
  fill?: string;
  persistImage?: boolean;
}) {
  const ref = useRef<HTMLDivElement>(null);
  // Cap the CSS blur at WIN_BLUR_MAX so Chromium's compositor reliably
  // GPU-rasterizes it; the Rust capture path pre-blurs the bitmap so the
  // panels still look meaningfully frosted at this lower CSS radius.
  const effectiveBlur = Math.min(blurRadius * WIN_BLUR_SCALE, WIN_BLUR_MAX);

  // Track the parent panel's screen-relative origin so the bitmap inside
  // each panel shows the part of the screen directly behind it. Earlier
  // versions of this hook used MutationObserver on the parent's style
  // attribute, which created a feedback loop with the overlay's
  // `useOverlayInteractionRegions` body-subtree observer — every BlurredBackdrop
  // setState fired the regions IPC, which re-rendered React, which set new
  // styles, repeat. The rAF loop below is allocation-free, doesn't go
  // through React state, and only writes the `backgroundPosition` style
  // when the value actually changes, so it stays cheap even across 8
  // mounted instances during a chat-panel drag.
  // A naive always-on rAF kept ~5% idle CPU with 8 panels open (8×
  // getBoundingClientRect + scheduling overhead per frame). Instead: do a
  // sync update once on mount and on size changes (ResizeObserver), and
  // only spin a 60Hz rAF DURING user-driven motion (drags, transitions,
  // scrolls) with a short idle quiet period before it shuts off. Idle CPU
  // drops to ~0; visual fidelity is identical because rAF is back at 60Hz
  // the instant motion starts.
  useEffect(() => {
    const el = ref.current;
    const parent = el?.parentElement;
    if (!el || !parent) return;

    let raf = 0;
    let idleTimer = 0;
    let lastLeft = Number.NaN;
    let lastTop = Number.NaN;

    const updateOnce = () => {
      const rect = parent.getBoundingClientRect();
      const nextLeft = -rect.left;
      const nextTop = -rect.top;
      if (
        Math.abs(nextLeft - lastLeft) < 0.5 &&
        Math.abs(nextTop - lastTop) < 0.5
      ) {
        return false;
      }
      lastLeft = nextLeft;
      lastTop = nextTop;
      el.style.backgroundPosition = `${nextLeft}px ${nextTop}px`;
      return true;
    };

    const tick = () => {
      raf = 0;
      updateOnce();
      if (idleTimer !== 0) {
        raf = requestAnimationFrame(tick);
      }
    };

    // ~200ms of quiet after the last motion event before the rAF stops.
    // Long enough to coast through a CSS transition tail / momentum scroll.
    const QUIET_MS = 200;

    const wake = () => {
      if (idleTimer !== 0) {
        clearTimeout(idleTimer);
      }
      idleTimer = window.setTimeout(() => {
        idleTimer = 0;
      }, QUIET_MS);
      if (raf === 0) {
        raf = requestAnimationFrame(tick);
      }
    };

    updateOnce();

    const ro = new ResizeObserver(() => {
      updateOnce();
      wake();
    });
    ro.observe(parent);

    const onWinMotion = () => wake();
    window.addEventListener("scroll", onWinMotion, {
      capture: true,
      passive: true,
    });
    window.addEventListener("resize", onWinMotion, { passive: true });

    const onPointer = () => wake();
    window.addEventListener("pointerdown", onPointer, {
      capture: true,
      passive: true,
    });
    window.addEventListener("pointermove", onPointer, {
      capture: true,
      passive: true,
    });
    window.addEventListener("pointerup", onPointer, {
      capture: true,
      passive: true,
    });

    const onMotion = () => wake();
    parent.addEventListener("transitionrun", onMotion);
    parent.addEventListener("transitionend", onMotion);
    parent.addEventListener("animationstart", onMotion);
    parent.addEventListener("animationend", onMotion);

    return () => {
      if (raf !== 0) cancelAnimationFrame(raf);
      if (idleTimer !== 0) clearTimeout(idleTimer);
      ro.disconnect();
      window.removeEventListener("scroll", onWinMotion, { capture: true });
      window.removeEventListener("resize", onWinMotion);
      window.removeEventListener("pointerdown", onPointer, { capture: true });
      window.removeEventListener("pointermove", onPointer, { capture: true });
      window.removeEventListener("pointerup", onPointer, { capture: true });
      parent.removeEventListener("transitionrun", onMotion);
      parent.removeEventListener("transitionend", onMotion);
      parent.removeEventListener("animationstart", onMotion);
      parent.removeEventListener("animationend", onMotion);
    };
  }, []);

  // Drive the bitmap from the `src` prop AND from live 60fps refresh
  // events. Both paths write to `el.style.backgroundImage` directly so
  // there's no React reconciliation cost when the bitmap updates — at
  // 60fps that React diff is the difference between buttery-smooth blur
  // and a perceptible stutter every frame. Rust's
  // `windows_window::capture_continuous_backdrop` emits the live event
  // while the overlay is visible.
  useEffect(() => {
    const el = ref.current;
    if (el) {
      el.style.backgroundImage = `url(data:image/png;base64,${src})`;
    }
  }, [src]);
  useEffect(() => {
    const unlistenPromise = listen<string>(
      "overlay-backdrop-update",
      (event) => {
        const el = ref.current;
        if (!el) return;
        const next = event.payload;
        if (typeof next !== "string" || next.length === 0) return;
        el.style.backgroundImage = `url(data:image/png;base64,${next})`;
      },
    );
    return () => {
      unlistenPromise.then((fn) => fn()).catch(() => {});
    };
  }, []);

  // All three layers carry the `screenie-blurred-backdrop` class so the
  // `.screenie-chat-panel > :not(.screenie-blurred-backdrop)` rule in
  // overlay.css correctly excludes them from the z-index: 1 layer reserved
  // for actual panel content. Without this the tint/fill divs would land
  // on the same layer as the content, painting OVER buttons (they have
  // pointer-events: none so clicks still reach controls — but they read as
  // milky-glass on top of text, which is the wrong look).
  return (
    <>
      <div
        ref={ref}
        className="screenie-blurred-backdrop"
        aria-hidden
        style={{
          position: "absolute",
          inset: 0,
          // backgroundImage is written imperatively by the two useEffects
          // above (src prop + overlay-backdrop-update event) so the 60fps
          // refresh doesn't go through React reconciliation.
          backgroundSize: `${screenW}px ${screenH}px`,
          backgroundRepeat: "no-repeat",
          filter: `blur(${effectiveBlur}px) brightness(${imageBrightness}) saturate(${WIN_SATURATION})`,
          // `will-change: filter` promotes the element to its own
          // compositor layer so Chromium GPU-rasterizes the blur instead
          // of CPU-fallbacking and silently dropping it. The translateZ(0)
          // is a separate compositor-promotion hint that works on Chromium
          // versions where will-change alone isn't honored on layered
          // transparent windows. Together they're the reason the blur
          // actually renders on Win11 22H2+ WebView2.
          willChange: "filter",
          transform: "translateZ(0)",
          zIndex: 0,
          pointerEvents: "none",
        }}
      />
      <div
        className="screenie-blurred-backdrop"
        aria-hidden
        style={{
          position: "absolute",
          inset: 0,
          background: tint,
          zIndex: 0,
          pointerEvents: "none",
        }}
      />
      <div
        className="screenie-blurred-backdrop"
        aria-hidden
        style={{
          position: "absolute",
          inset: 0,
          background: fill,
          zIndex: 0,
          pointerEvents: "none",
        }}
      />
    </>
  );
}

/// Inset hairline that traces the parent's rounded edges. Originally an
/// SVG driven by ResizeObserver → setState, but the React re-render lag
/// during fast resize/expand animations made the SVG's rect drift inside
/// the wrapper for a frame or two, which the user saw as a "weird"
/// flickering border. CSS borders recalc synchronously with the parent,
/// so a single `<div>` with a real border stays pixel-stable at every
/// size — no observers, no state, no re-renders.
///
/// Stroke width defaults to a full integer pixel (1) with low alpha
/// rather than a fractional sub-pixel width with higher alpha. Sub-pixel
/// CSS borders anti-alias inconsistently around `border-radius` corners
/// (the curve distributes the same alpha across more pixels than the
/// straight sides), which shows up as visibly-thinner corners. Integer
/// pixel widths render uniformly on every edge.
export function SvgInsetBorder({
  radius,
  inset = 2,
  strokeWidth = 1,
  strokeAlpha = 0.22,
}: {
  radius: number;
  inset?: number;
  strokeWidth?: number;
  strokeAlpha?: number;
}) {
  return (
    <div
      className="screenie-inset-border"
      aria-hidden
      style={{
        position: "absolute",
        inset: `${inset}px`,
        borderRadius: `${Math.max(0, radius - inset)}px`,
        border: `${strokeWidth}px solid rgba(255, 255, 255, ${strokeAlpha})`,
        boxSizing: "border-box",
        pointerEvents: "none",
      }}
    />
  );
}

/// Standard frost recipe used by the prompt toolbar. Matches `.screenie-toolbar`'s
/// CSS recipe (background + backdrop-filter) and the inner BlurredBackdrop tuning
/// the toolbar uses. Reused by the edit pill so its background is identical.
export const TOOLBAR_FROST = {
  blurRadius: 26,
  imageBrightness: 0.64,
  tint: "rgba(34, 36, 35, 0.43)",
  fill: "rgba(18, 19, 18, 0.17)",
} as const;

/// Frost recipe matching `.screenie-chat-panel` — slightly cooler/darker.
export const CHAT_PANEL_FROST = {
  blurRadius: 32,
  imageBrightness: 0.64,
  tint: "rgba(34, 36, 35, 0.50)",
  fill: "rgba(18, 19, 18, 0.20)",
} as const;
