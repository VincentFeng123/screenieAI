import { useRef } from "react";
import { gsap } from "gsap";
import { useGSAP } from "@gsap/react";
import { useReducedMotion } from "../lib/useReducedMotion";

const STATS: { label: string; from: number; to: number; suffix: string; precision?: number }[] = [
  { label: "Vision providers", from: 0, to: 4, suffix: "" },
  { label: "Hotkey latency", from: 0, to: 22, suffix: "ms" },
  { label: "Capture overlay", from: 0, to: 60, suffix: "Hz" },
  { label: "Binary size", from: 0, to: 12, suffix: "MB" },
];

export function StatsTicker() {
  const rootRef = useRef<HTMLDivElement>(null);
  const reduced = useReducedMotion();

  useGSAP(
    () => {
      const values = rootRef.current?.querySelectorAll<HTMLElement>(".sl-stat__v");
      if (!values) return;

      values.forEach((el, i) => {
        const stat = STATS[i];
        if (!stat) return;
        if (reduced) {
          el.textContent = `${stat.to}${stat.suffix}`;
          return;
        }
        const o = { v: stat.from };
        gsap.to(o, {
          v: stat.to,
          duration: 1.6 + i * 0.15,
          ease: "expo.out",
          delay: 0.6 + i * 0.1,
          onUpdate() {
            el.textContent = `${Math.round(o.v)}${stat.suffix}`;
          },
        });
      });

      // Idle: low-pulse the dot
      if (!reduced) {
        const dot = rootRef.current?.querySelector<HTMLElement>(".sl-stats__live");
        if (dot) {
          gsap.to(dot, {
            opacity: 0.3,
            duration: 1.0,
            ease: "sine.inOut",
            yoyo: true,
            repeat: -1,
          });
        }
      }
    },
    { scope: rootRef, dependencies: [reduced] }
  );

  return (
    <div className="sl-stats" ref={rootRef} aria-hidden="true">
      <div className="sl-stats__head">
        <span className="sl-stats__live" />
        Live
      </div>
      {STATS.map((s) => (
        <div className="sl-stat" key={s.label}>
          <span className="sl-stat__v">0{s.suffix}</span>
          <span className="sl-stat__k">{s.label}</span>
        </div>
      ))}
    </div>
  );
}
