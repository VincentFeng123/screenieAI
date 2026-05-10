import { useRef } from "react";
import { gsap } from "gsap";
import { useGSAP } from "@gsap/react";
import { useReducedMotion } from "../lib/useReducedMotion";

const ITEMS = [
  "Code blocks",
  "Charts",
  "Diagrams",
  "Dense math",
  "UI screenshots",
  "Forms",
  "Receipts",
  "Foreign signage",
  "Spreadsheets",
  "Whiteboards",
  "Maps",
  "Hand-drawn notes",
];

export function Marquee() {
  const rootRef = useRef<HTMLDivElement>(null);
  const reduced = useReducedMotion();

  useGSAP(
    () => {
      if (reduced) return;
      const track = rootRef.current?.querySelector<HTMLElement>(".sl-marquee__track");
      if (!track) return;
      const half = track.scrollWidth / 2;
      gsap.to(track, {
        x: -half,
        ease: "none",
        duration: 32,
        repeat: -1,
      });
    },
    { scope: rootRef, dependencies: [reduced] }
  );

  return (
    <div className="sl-marquee" ref={rootRef} aria-hidden="true">
      <div className="sl-marquee__edge sl-marquee__edge--l" />
      <div className="sl-marquee__edge sl-marquee__edge--r" />
      <div className="sl-marquee__track">
        {[...ITEMS, ...ITEMS].map((it, i) => (
          <span className="sl-marquee__item" key={i}>
            <span className="sl-marquee__dot" />
            {it}
          </span>
        ))}
      </div>
    </div>
  );
}
