import { useEffect, useRef } from "react";
import { gsap } from "gsap";
import { useReducedMotion } from "../lib/useReducedMotion";

export function MagneticCrosshair() {
  const rootRef = useRef<HTMLDivElement>(null);
  const reduced = useReducedMotion();

  useEffect(() => {
    if (reduced) return;
    // Disable on touch devices
    const hover = window.matchMedia("(hover: hover) and (pointer: fine)");
    if (!hover.matches) return;

    const root = rootRef.current;
    if (!root) return;

    const xTo = gsap.quickTo(root, "x", { duration: 0.42, ease: "power3.out" });
    const yTo = gsap.quickTo(root, "y", { duration: 0.42, ease: "power3.out" });
    const sTo = gsap.quickTo(root, "scale", { duration: 0.32, ease: "power3.out" });

    let lastX = window.innerWidth / 2;
    let lastY = window.innerHeight / 2;

    const onMove = (e: PointerEvent) => {
      lastX = e.clientX;
      lastY = e.clientY;
      xTo(lastX);
      yTo(lastY);
    };

    const isInteractive = (el: Element | null): boolean => {
      while (el && el !== document.body) {
        if (el instanceof HTMLElement) {
          const tag = el.tagName.toLowerCase();
          if (tag === "a" || tag === "button" || el.role === "button") return true;
          if (el.classList.contains("sl-card") || el.classList.contains("sl-prov__tile")) return true;
        }
        el = (el as HTMLElement).parentElement;
      }
      return false;
    };

    const onOver = (e: PointerEvent) => {
      sTo(isInteractive(e.target as Element | null) ? 1.6 : 1);
    };

    const show = () => gsap.to(root, { autoAlpha: 1, duration: 0.2 });
    const hide = () => gsap.to(root, { autoAlpha: 0, duration: 0.2 });

    gsap.set(root, { autoAlpha: 0, x: lastX, y: lastY });

    window.addEventListener("pointermove", onMove);
    window.addEventListener("pointerover", onOver);
    document.documentElement.addEventListener("pointerenter", show);
    document.documentElement.addEventListener("pointerleave", hide);

    show();

    return () => {
      window.removeEventListener("pointermove", onMove);
      window.removeEventListener("pointerover", onOver);
      document.documentElement.removeEventListener("pointerenter", show);
      document.documentElement.removeEventListener("pointerleave", hide);
    };
  }, [reduced]);

  if (reduced) return null;

  return (
    <div className="sl-mag" ref={rootRef} aria-hidden="true">
      <span className="sl-mag__ring" />
      <span className="sl-mag__bar sl-mag__bar--v" />
      <span className="sl-mag__bar sl-mag__bar--h" />
      <span className="sl-mag__dot" />
    </div>
  );
}
