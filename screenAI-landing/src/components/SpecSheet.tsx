import { useRef } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { useGSAP } from "@gsap/react";
import { useReducedMotion } from "../lib/useReducedMotion";
import { scrambleAt } from "../lib/scramble";

gsap.registerPlugin(ScrollTrigger);

const SPECS: { k: string; v: string }[] = [
  { k: "Build", v: "screenie-ai · 0.1.0-alpha.7" },
  { k: "Engine", v: "Tauri 2 · Rust · WebView" },
  { k: "Capture", v: "Native CGImage · 60Hz overlay" },
  { k: "Encoder", v: "PNG · Q90 JPEG · sub-100ms" },
  { k: "Models", v: "Claude · GPT · Gemini · Ollama" },
  { k: "Storage", v: "Keychain · sandboxed FS" },
  { k: "Telemetry", v: "None · zero pings · zero ids" },
  { k: "Footprint", v: "12 MB binary · 38 MB RAM idle" },
];

export function SpecSheet() {
  const rootRef = useRef<HTMLElement>(null);
  const reduced = useReducedMotion();

  useGSAP(
    () => {
      const lines = rootRef.current?.querySelectorAll<HTMLElement>(".sl-spec__row");
      if (!lines) return;

      // Initial: blank values
      lines.forEach((row) => {
        const v = row.querySelector<HTMLElement>(".sl-spec__v");
        if (v) v.textContent = "";
        gsap.set(row, { autoAlpha: 0, x: -8 });
      });

      if (reduced) {
        lines.forEach((row, i) => {
          const v = row.querySelector<HTMLElement>(".sl-spec__v");
          if (v) v.textContent = SPECS[i]?.v ?? "";
          gsap.set(row, { autoAlpha: 1, x: 0 });
        });
        return;
      }

      const tl = gsap.timeline({
        scrollTrigger: { trigger: rootRef.current, start: "top 70%" },
      });

      lines.forEach((row, i) => {
        const v = row.querySelector<HTMLElement>(".sl-spec__v");
        const target = SPECS[i]?.v ?? "";
        tl.to(row, { autoAlpha: 1, x: 0, duration: 0.32, ease: "power2.out" }, i * 0.12);
        if (v) {
          const o = { p: 0 };
          tl.to(
            o,
            {
              p: 1,
              duration: 0.55 + target.length * 0.012,
              ease: "none",
              onUpdate() {
                v.textContent = scrambleAt(target, o.p, 4);
              },
              onComplete() { v.textContent = target; },
            },
            i * 0.12 + 0.1
          );
        }
      });
    },
    { scope: rootRef, dependencies: [reduced] }
  );

  return (
    <section
      className="sl-section sl-spec-sec"
      ref={rootRef}
      aria-labelledby="spec-title"
    >
      <div className="sl-container">
        <div className="sl-section__head">
          <span className="sl-eyebrow"><span className="sl-section-num">04</span>Spec sheet</span>
          <h2 className="sl-section__title" id="spec-title">
            The instrument, in numbers.
          </h2>
          <p className="sl-section__lede">
            What ships. Read like a part-list, because that's what it is.
          </p>
        </div>

        <div className="sl-spec">
          <div className="sl-spec__corners">
            <span /><span /><span /><span />
          </div>
          {SPECS.map((s, i) => (
            <div className="sl-spec__row" key={s.k}>
              <span className="sl-spec__idx">{String(i + 1).padStart(2, "0")}</span>
              <span className="sl-spec__k">{s.k}</span>
              <span className="sl-spec__sep" />
              <span className="sl-spec__v" />
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
