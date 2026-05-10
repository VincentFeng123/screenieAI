import { useRef } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { useGSAP } from "@gsap/react";
import { providers } from "../lib/copy";
import { useReducedMotion } from "../lib/useReducedMotion";

gsap.registerPlugin(ScrollTrigger);

export function Providers() {
  const rootRef = useRef<HTMLElement>(null);
  const reduced = useReducedMotion();

  useGSAP(
    () => {
      if (reduced) return;
      const tiles = rootRef.current?.querySelectorAll<HTMLElement>(".sl-prov__tile");
      if (!tiles) return;
      gsap.set(tiles, { autoAlpha: 0, y: 14 });
      gsap.to(tiles, {
        autoAlpha: 1,
        y: 0,
        duration: 0.6,
        ease: "expo.out",
        stagger: 0.06,
        scrollTrigger: { trigger: rootRef.current, start: "top 80%" },
      });
    },
    { scope: rootRef, dependencies: [reduced] }
  );

  return (
    <section
      className="sl-section sl-prov"
      id="providers"
      ref={rootRef}
      aria-labelledby="prov-title"
    >
      <div className="sl-container">
        <div className="sl-section__head">
          <span className="sl-eyebrow"><span className="sl-section-num">05</span>Providers</span>
          <h2 className="sl-section__title" id="prov-title">
            Bring your own model.
          </h2>
          <p className="sl-section__lede">
            Cloud or on-device. Switch at any time. Keys never leave the macOS Keychain.
          </p>
        </div>

        <div className="sl-prov__grid">
          {providers.map((p) => (
            <div className="sl-prov__tile" key={p.name} data-mode={p.mode}>
              <div className="sl-prov__head">
                <span className="sl-prov__mode">{p.mode === "local" ? "On-device" : "Cloud"}</span>
                <span className={`sl-prov__dot sl-prov__dot--${p.mode}`} />
              </div>
              <span className="sl-prov__name">{p.name}</span>
              <span className="sl-prov__model">{p.model}</span>
            </div>
          ))}
        </div>
      </div>
    </section>
  );
}
