import { useRef } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { useGSAP } from "@gsap/react";
import { FeatureCard } from "./FeatureCard";
import { featureItems } from "../lib/copy";
import { useReducedMotion } from "../lib/useReducedMotion";

gsap.registerPlugin(ScrollTrigger);

export function Features() {
  const rootRef = useRef<HTMLElement>(null);
  const reduced = useReducedMotion();

  useGSAP(
    () => {
      const cards = rootRef.current?.querySelectorAll<HTMLElement>(".sl-card");
      if (!cards || cards.length === 0) return;

      gsap.set(cards, { y: 32, autoAlpha: 0, rotateX: 8 });

      if (reduced) {
        gsap.set(cards, { y: 0, autoAlpha: 1, rotateX: 0 });
        return;
      }

      gsap.to(cards, {
        y: 0,
        autoAlpha: 1,
        rotateX: 0,
        duration: 0.8,
        ease: "expo.out",
        stagger: 0.08,
        scrollTrigger: { trigger: rootRef.current, start: "top 78%" },
      });

      // 3D mouse-tilt per card
      const hover = window.matchMedia("(hover: hover) and (pointer: fine)");
      if (!hover.matches) return;

      const handlers: { el: HTMLElement; move: (e: PointerEvent) => void; leave: () => void }[] = [];
      cards.forEach((card) => {
        const inner = card.querySelector<HTMLElement>(".sl-card__inner") || card;
        const onMove = (e: PointerEvent) => {
          const r = card.getBoundingClientRect();
          const dx = (e.clientX - r.left - r.width / 2) / r.width;
          const dy = (e.clientY - r.top - r.height / 2) / r.height;
          gsap.to(inner, {
            rotateX: -dy * 6,
            rotateY: dx * 8,
            duration: 0.5,
            ease: "power2.out",
            transformPerspective: 800,
            transformOrigin: "center",
          });
          gsap.to(card.querySelector(".sl-card__shine"), {
            "--x": `${(e.clientX - r.left) / r.width * 100}%`,
            "--y": `${(e.clientY - r.top) / r.height * 100}%`,
            duration: 0.4,
            ease: "power2.out",
          });
        };
        const onLeave = () => {
          gsap.to(inner, { rotateX: 0, rotateY: 0, duration: 0.6, ease: "power3.out" });
        };
        card.addEventListener("pointermove", onMove);
        card.addEventListener("pointerleave", onLeave);
        handlers.push({ el: card, move: onMove, leave: onLeave });
      });

      return () => {
        handlers.forEach(({ el, move, leave }) => {
          el.removeEventListener("pointermove", move);
          el.removeEventListener("pointerleave", leave);
        });
      };
    },
    { scope: rootRef, dependencies: [reduced] }
  );

  return (
    <section
      className="sl-section sl-features"
      id="features"
      ref={rootRef}
      aria-labelledby="features-title"
    >
      <div className="sl-container">
        <div className="sl-section__head">
          <span className="sl-eyebrow"><span className="sl-section-num">02</span>Features</span>
          <h2 className="sl-section__title" id="features-title">
            Built for precision, not hype.
          </h2>
          <p className="sl-section__lede">
            Six things Screenie does well. Nothing else demanding your attention.
          </p>
        </div>

        <div className="sl-features__grid">
          {featureItems.map((f, i) => (
            <FeatureCard
              key={f.title}
              icon={f.icon}
              title={f.title}
              body={f.body}
              index={i + 1}
            />
          ))}
        </div>
      </div>
    </section>
  );
}
