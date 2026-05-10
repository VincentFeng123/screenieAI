import { useRef } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { useGSAP } from "@gsap/react";
import { Apple } from "lucide-react";
import { useReducedMotion } from "../lib/useReducedMotion";

gsap.registerPlugin(ScrollTrigger);

export function CTA() {
  const rootRef = useRef<HTMLElement>(null);
  const reduced = useReducedMotion();

  useGSAP(
    () => {
      if (reduced) return;
      const els = rootRef.current?.querySelectorAll<HTMLElement>(".sl-cta__rev");
      if (!els) return;
      gsap.set(els, { y: 16, autoAlpha: 0 });
      gsap.to(els, {
        y: 0,
        autoAlpha: 1,
        duration: 0.7,
        ease: "expo.out",
        stagger: 0.08,
        scrollTrigger: { trigger: rootRef.current, start: "top 75%" },
      });
    },
    { scope: rootRef, dependencies: [reduced] }
  );

  return (
    <section
      className="sl-section sl-cta-sec"
      id="download"
      ref={rootRef}
      aria-labelledby="cta-title"
    >
      <div className="sl-container">
        <div className="sl-cta">
          <span className="sl-eyebrow sl-cta__rev"><span className="sl-section-num">06</span>Download</span>
          <h2 className="sl-cta__title sl-cta__rev" id="cta-title">
            See your screen, answered.
          </h2>
          <p className="sl-cta__lede sl-cta__rev">
            Free during alpha. Apple Silicon and Intel builds. macOS 13 and up.
          </p>
          <div className="sl-cta__row sl-cta__rev">
            <a href="#" className="sl-btn sl-btn--lg">
              <Apple size={16} strokeWidth={1.6} />
              Download for macOS
            </a>
            <button className="sl-btn sl-btn--lg sl-btn--ghost" disabled>
              Windows — coming soon
            </button>
          </div>
          <span className="sl-cta__meta sl-cta__rev">
            Universal binary · Notarized · No telemetry
          </span>
        </div>
      </div>
    </section>
  );
}
