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
  useEffect(() => {
    let raf = 0;
    let lastLeft = Number.NaN;
    let lastTop = Number.NaN;
    const tick = () => {
      raf = requestAnimationFrame(tick);
      const el = ref.current;
      const parent = el?.parentElement;
      if (!el || !parent) return;
      const rect = parent.getBoundingClientRect();
      const nextLeft = -rect.left;
      const nextTop = -rect.top;
      if (
        Math.abs(nextLeft - lastLeft) < 0.5 &&
        Math.abs(nextTop - lastTop) < 0.5
      ) {
        return;
      }
      lastLeft = nextLeft;
      lastTop = nextTop;
      // Direct style write — bypasses React reconciliation. Each frame
      // becomes one getBoundingClientRect + (at most) one style assignment.
      el.style.backgroundPosition = `${nextLeft}px ${nextTop}px`;
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
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
          backgroundImage: `url(data:image/png;base64,${src})`,
          backgroundSize: `${screenW}px ${screenH}px`,
          backgroundRepeat: "no-repeat",
          filter: `blur(${blurRadius}px) brightness(${imageBrightness})`,
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
