/* ------------------------------------------------------------------ */
/* BlurredBackdrop — no-op shim (vibrancy moved to native AppKit)      */
/*                                                                     */
/* The frost behind each overlay panel is now provided by an           */
/* NSVisualEffectView mounted as a sibling of the WKWebView (see       */
/* macos_window.m's vibrancy section + the `useOverlayFrostRegions`    */
/* hook in Overlay.tsx). The native compositor blurs the live desktop  */
/* every frame, so the overlay no longer relies on a captured-PNG      */
/* snapshot for frost.                                                 */
/*                                                                     */
/* This component used to render the bitmap + tint + fill stack inside */
/* every panel. Returning `null` here means callers don't have to be   */
/* changed — the JSX still references it but it just renders nothing.  */
/* The panel's CSS background still draws on top of the vibrancy view  */
/* (which is BEHIND the WebView), so a low-alpha bg color is what      */
/* tones the frost. Adjust panel CSS, not this component, to dial      */
/* darkness in or out.                                                 */
/* ------------------------------------------------------------------ */

export function BlurredBackdrop(_props: {
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
  void _props;
  return null;
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
