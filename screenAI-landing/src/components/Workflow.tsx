import { useRef } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { useGSAP } from "@gsap/react";
import { Command, Crop, Code2, Cloud, MessageSquare } from "lucide-react";
import { useReducedMotion } from "../lib/useReducedMotion";

gsap.registerPlugin(ScrollTrigger);

const NODES = [
  { id: "hotkey", label: "Hotkey", caption: "Global shortcut", Icon: Command, ms: "0ms" },
  { id: "capture", label: "Capture", caption: "Native region", Icon: Crop, ms: "12ms" },
  { id: "encode", label: "Encode", caption: "Edit · compress", Icon: Code2, ms: "38ms" },
  { id: "provider", label: "Provider", caption: "Cloud or Local", Icon: Cloud, ms: "180ms", badge: true },
  { id: "render", label: "Render", caption: "Floating window", Icon: MessageSquare, ms: "412ms" },
];

export function Workflow() {
  const rootRef = useRef<HTMLElement>(null);
  const pinRef = useRef<HTMLDivElement>(null);
  const svgRef = useRef<SVGSVGElement>(null);
  const reduced = useReducedMotion();

  useGSAP(
    () => {
      const paths = svgRef.current?.querySelectorAll<SVGPathElement>(".sl-wf__connector");
      const nodes = rootRef.current?.querySelectorAll<HTMLElement>(".sl-wf__node");
      const dots = svgRef.current?.querySelectorAll<SVGCircleElement>(".sl-wf__dot");
      const ringPaths = svgRef.current?.querySelectorAll<SVGPathElement>(".sl-wf__halo");

      if (!paths || !nodes) return;

      paths.forEach((path) => {
        const len = path.getTotalLength();
        gsap.set(path, { strokeDasharray: len, strokeDashoffset: len });
      });
      gsap.set(nodes, { autoAlpha: 0, scale: 0.95, y: 14 });
      if (dots) gsap.set(dots, { autoAlpha: 0 });
      if (ringPaths) {
        ringPaths.forEach((path) => {
          const len = path.getTotalLength();
          gsap.set(path, { strokeDasharray: `${len * 0.18} ${len * 0.82}`, strokeDashoffset: len });
        });
      }

      if (reduced) {
        paths.forEach((path) => gsap.set(path, { strokeDashoffset: 0 }));
        gsap.set(nodes, { autoAlpha: 1, scale: 1, y: 0 });
        return;
      }

      // SCRUB-on-scroll: pin the section, draw paths and reveal nodes as user scrolls.
      const tl = gsap.timeline({
        scrollTrigger: {
          trigger: pinRef.current,
          start: "top 12%",
          end: "+=120%",
          pin: true,
          pinSpacing: true,
          scrub: 0.4,
          anticipatePin: 1,
        },
      });

      // Reveal nodes one by one, then draw connectors between them.
      nodes.forEach((node, i) => {
        tl.to(node, { autoAlpha: 1, scale: 1, y: 0, duration: 0.6, ease: "power2.out" }, i * 0.6);
        if (i < paths.length) {
          const path = paths[i];
          tl.to(path, { strokeDashoffset: 0, duration: 0.7, ease: "none" }, i * 0.6 + 0.3);
        }
      });

      // After all paths are drawn, persistent halo loop across all paths.
      const haloTl = gsap.timeline({
        repeat: -1,
        scrollTrigger: { trigger: rootRef.current, start: "top bottom", end: "bottom top", toggleActions: "play pause resume pause" },
      });
      if (ringPaths) {
        ringPaths.forEach((path, i) => {
          haloTl.to(
            path,
            { strokeDashoffset: -path.getTotalLength(), duration: 2.4, ease: "none" },
            i * 0.5
          );
        });
      }

      // Pulse dots traverse continuously
      if (dots && dots.length === paths.length) {
        const dotsTl = gsap.timeline({
          repeat: -1,
          repeatDelay: 0.6,
          scrollTrigger: { trigger: rootRef.current, start: "top bottom", end: "bottom top", toggleActions: "play pause resume pause" },
        });
        paths.forEach((path, i) => {
          const dot = dots[i];
          const len = path.getTotalLength();
          const t = { p: 0 };
          dotsTl.to(
            t,
            {
              p: 1,
              duration: 1.2,
              ease: "power2.inOut",
              onUpdate() {
                const pt = path.getPointAtLength(len * t.p);
                gsap.set(dot, { autoAlpha: 1, attr: { cx: pt.x, cy: pt.y } });
              },
              onComplete() {
                gsap.to(dot, { autoAlpha: 0, duration: 0.3 });
              },
              onStart() { t.p = 0; },
            },
            i * 0.45
          );
        });
      }
    },
    { scope: rootRef, dependencies: [reduced] }
  );

  return (
    <section
      className="sl-section sl-wf"
      id="workflow"
      ref={rootRef}
      aria-labelledby="wf-title"
    >
      <div className="sl-container">
        <div className="sl-section__head">
          <span className="sl-eyebrow"><span className="sl-section-num">03</span>How it works</span>
          <h2 className="sl-section__title" id="wf-title">
            Five steps. About four hundred milliseconds.
          </h2>
          <p className="sl-section__lede">
            From hotkey to floating answer — the path the bytes take.
          </p>
        </div>

        <div className="sl-wf__pin" ref={pinRef}>
          <div className="sl-wf__board">
            <svg
              className="sl-wf__svg"
              ref={svgRef}
              viewBox="0 0 1000 220"
              preserveAspectRatio="none"
              aria-hidden="true"
            >
              <defs>
                <marker id="sl-wf-arrow" viewBox="0 0 8 8" refX="7" refY="4" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
                  <path d="M0 0 L8 4 L0 8" fill="none" stroke="rgba(255,255,255,0.55)" strokeWidth="1" />
                </marker>
              </defs>

              <path className="sl-wf__connector" d="M 100 110 C 160 60, 240 60, 300 110" />
              <path className="sl-wf__connector" d="M 300 110 C 360 160, 440 160, 500 110" />
              <path className="sl-wf__connector" d="M 500 110 C 560 60, 640 60, 700 110" />
              <path className="sl-wf__connector" d="M 700 110 C 760 160, 840 160, 900 110" />

              {/* Halo overlay paths — same geometry, animated via dashoffset (continuous) */}
              <path className="sl-wf__halo" d="M 100 110 C 160 60, 240 60, 300 110" />
              <path className="sl-wf__halo" d="M 300 110 C 360 160, 440 160, 500 110" />
              <path className="sl-wf__halo" d="M 500 110 C 560 60, 640 60, 700 110" />
              <path className="sl-wf__halo" d="M 700 110 C 760 160, 840 160, 900 110" />

              <circle className="sl-wf__dot" r="3.6" cx="0" cy="0" />
              <circle className="sl-wf__dot" r="3.6" cx="0" cy="0" />
              <circle className="sl-wf__dot" r="3.6" cx="0" cy="0" />
              <circle className="sl-wf__dot" r="3.6" cx="0" cy="0" />
            </svg>

            <ol className="sl-wf__nodes" aria-label="Workflow steps">
              {NODES.map((n, i) => {
                const Icon = n.Icon;
                return (
                  <li className="sl-wf__node" key={n.id}>
                    <span className="sl-wf__step">{String(i + 1).padStart(2, "0")}</span>
                    <span className="sl-wf__circle">
                      <Icon size={18} strokeWidth={1.4} />
                    </span>
                    <span className="sl-wf__label">{n.label}</span>
                    <span className="sl-wf__caption">{n.caption}</span>
                    <span className="sl-wf__ms">{n.ms}</span>
                    {n.badge && (
                      <span className="sl-wf__badge">
                        <span>Cloud</span>
                        <span className="sl-wf__badge-divider" />
                        <span>Local</span>
                      </span>
                    )}
                  </li>
                );
              })}
            </ol>
          </div>

          <p className="sl-wf__note">
            API keys stay in the macOS Keychain. Local mode never sends data off-device.
          </p>
        </div>
      </div>
    </section>
  );
}
