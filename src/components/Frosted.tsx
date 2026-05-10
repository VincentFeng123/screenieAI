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

import { useEffect, useLayoutEffect, useRef, useState } from "react";

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
  // Background-position offset that anchors the bitmap so the panel shows
  // the screen content directly behind itself. The bitmap covers the
  // overlay's viewport; we shift it by the panel's screen-relative origin
  // so each panel sees a window into the same global frame.
  const [pos, setPos] = useState<{ left: number; top: number }>({
    left: 0,
    top: 0,
  });

  useLayoutEffect(() => {
    const el = ref.current;
    const parent = el?.parentElement;
    if (!el || !parent) return;
    const update = () => {
      const rect = parent.getBoundingClientRect();
      setPos((prev) => {
        const nextLeft = -rect.left;
        const nextTop = -rect.top;
        if (
          Math.abs(prev.left - nextLeft) < 0.5 &&
          Math.abs(prev.top - nextTop) < 0.5
        ) {
          return prev;
        }
        return { left: nextLeft, top: nextTop };
      });
    };
    update();
    // ResizeObserver catches CSS-driven size changes (edit-pill 36→280
    // expansion, textarea auto-grow). MutationObserver catches inline-
    // style changes (toolbar following the rect, chat panel drag). Between
    // them, every position/size change React triggers fires an update
    // without us needing a rAF loop per BlurredBackdrop instance.
    const ro = new ResizeObserver(update);
    ro.observe(parent);
    const mo = new MutationObserver(update);
    mo.observe(parent, {
      attributes: true,
      attributeFilter: ["style", "class"],
    });
    return () => {
      ro.disconnect();
      mo.disconnect();
    };
  }, []);

  useEffect(() => {
    // Window-level resize / scroll affect every panel's screen position, so
    // sync all instances when those fire.
    const onResize = () => {
      const parent = ref.current?.parentElement;
      if (!parent) return;
      const rect = parent.getBoundingClientRect();
      setPos({ left: -rect.left, top: -rect.top });
    };
    window.addEventListener("resize", onResize);
    return () => window.removeEventListener("resize", onResize);
  }, []);

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
          backgroundPosition: `${pos.left}px ${pos.top}px`,
          backgroundRepeat: "no-repeat",
          filter: `blur(${blurRadius}px) brightness(${imageBrightness})`,
          // Overshoot the blur radius so the blur kernel can sample pixels
          // from beyond the panel's edge — without this, the blurred edge
          // is darkened by the transparent border.
          margin: -blurRadius,
          zIndex: 0,
          pointerEvents: "none",
        }}
      />
      <div
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
