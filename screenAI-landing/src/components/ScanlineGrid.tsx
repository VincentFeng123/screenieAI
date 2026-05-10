import { useRef } from "react";
import { useGSAP } from "@gsap/react";
import { gsap } from "gsap";
import { useReducedMotion } from "../lib/useReducedMotion";

export function ScanlineGrid() {
  const rootRef = useRef<HTMLDivElement>(null);
  const reduced = useReducedMotion();

  useGSAP(
    () => {
      if (reduced) return;
      const grid = rootRef.current?.querySelector<HTMLElement>(".sl-scanline__grid");
      const lines = rootRef.current?.querySelector<HTMLElement>(".sl-scanline__lines");
      if (!grid || !lines) return;

      gsap.to(grid, {
        backgroundPositionY: "+=56",
        duration: 18,
        ease: "none",
        repeat: -1,
      });
      gsap.to(lines, {
        backgroundPositionY: "+=120",
        duration: 32,
        ease: "none",
        repeat: -1,
      });

      // Periodic global scan beam — sweeps the page every ~16s
      const beam = rootRef.current?.querySelector<HTMLElement>(".sl-scanline__beam");
      if (beam) {
        const beamTl = gsap.timeline({ repeat: -1, repeatDelay: 7 });
        beamTl
          .fromTo(beam, { y: "-10%", autoAlpha: 0 }, { autoAlpha: 1, duration: 0.4 })
          .to(beam, { y: "110vh", duration: 4.0, ease: "power1.in" }, "<")
          .to(beam, { autoAlpha: 0, duration: 0.4 }, "-=0.5");
      }
    },
    { scope: rootRef, dependencies: [reduced] }
  );

  return (
    <div className="sl-scanline" ref={rootRef} aria-hidden="true">
      <div className="sl-scanline__halo" />
      <div className="sl-scanline__grid" />
      <div className="sl-scanline__lines" />
      <div className="sl-scanline__beam" />
    </div>
  );
}
