import { useRef } from "react";
import { gsap } from "gsap";
import { useGSAP } from "@gsap/react";
import { FakeDesktop } from "./Hero/FakeDesktop";
import { HotkeyChip } from "./Hero/HotkeyChip";
import { Cursor } from "./Hero/Cursor";
import { SelectionRect } from "./Hero/SelectionRect";
import { ChatWindow } from "./Hero/ChatWindow";
import { StatsTicker } from "./StatsTicker";
import { Marquee } from "./Marquee";
import { useReducedMotion } from "../lib/useReducedMotion";
import { scrambleAt } from "../lib/scramble";
import {
  heroEyebrow,
  heroHeadline,
  heroSubhead,
  heroPrompt,
  heroAnswer,
} from "../lib/copy";

export function Hero() {
  const rootRef = useRef<HTMLElement>(null);
  const stageWrapRef = useRef<HTMLDivElement>(null);
  const stageRef = useRef<HTMLDivElement>(null);
  const eyebrowRef = useRef<HTMLSpanElement>(null);
  const eyebrowTextRef = useRef<HTMLSpanElement>(null);
  const headlineRef = useRef<HTMLHeadingElement>(null);
  const subheadRef = useRef<HTMLParagraphElement>(null);
  const ctaRef = useRef<HTMLDivElement>(null);
  const hotkeyRef = useRef<HTMLDivElement>(null);
  const cursorRef = useRef<HTMLDivElement>(null);
  const hintRef = useRef<HTMLSpanElement>(null);
  const hintTextRef = useRef<HTMLSpanElement>(null);
  const selectionRef = useRef<HTMLDivElement>(null);
  const flashRef = useRef<HTMLDivElement>(null);
  const chatRef = useRef<HTMLDivElement>(null);
  const answerRef = useRef<HTMLSpanElement>(null);
  const caretRef = useRef<HTMLSpanElement>(null);
  const dimRef = useRef<HTMLDivElement>(null);
  const connectorRef = useRef<SVGPathElement>(null);
  const scanBeamRef = useRef<HTMLDivElement>(null);
  const timelineRef = useRef<gsap.core.Timeline | null>(null);

  const reduced = useReducedMotion();

  useGSAP(
    () => {
      const stage = stageRef.current;
      if (!stage) return;

      const stageW = stage.clientWidth;
      const stageH = stage.clientHeight;
      const startX = stageW * 0.22;
      const startY = stageH * 0.40;
      const endX = stageW * 0.74;
      const endY = stageH * 0.78;
      const selW = endX - startX;
      const selH = endY - startY;

      const chatW = Math.min(360, Math.max(240, stageW * 0.4));
      const chatX = Math.max(8, Math.min(stageW - chatW - 8, endX - chatW));
      const chatY = Math.min(stageH - 12, endY + 10);

      // Connector line: from selection's bottom-right to chat's top-left.
      const connFrom = { x: endX - 1, y: endY - 1 };
      const connTo = { x: chatX + 18, y: chatY + 6 };
      if (connectorRef.current) {
        const svg = connectorRef.current.ownerSVGElement;
        if (svg) svg.setAttribute("viewBox", `0 0 ${stageW} ${stageH}`);
        connectorRef.current.setAttribute(
          "d",
          `M ${connFrom.x} ${connFrom.y} L ${connTo.x} ${connTo.y}`
        );
        const len = connectorRef.current.getTotalLength();
        gsap.set(connectorRef.current, {
          strokeDasharray: len,
          strokeDashoffset: len,
          autoAlpha: 0,
        });
      }

      // ----- INITIAL STATES -----
      const lines = headlineRef.current?.querySelectorAll<HTMLElement>(".sl-line__inner");
      gsap.set(lines || [], { y: "110%", filter: "blur(10px)" });
      gsap.set(eyebrowRef.current, { autoAlpha: 0 });
      if (eyebrowTextRef.current) eyebrowTextRef.current.textContent = "";
      gsap.set(subheadRef.current, { autoAlpha: 0, y: 14 });
      gsap.set(ctaRef.current, { autoAlpha: 0, y: 12 });

      gsap.set(hotkeyRef.current, { autoAlpha: 0, scale: 0.94, yPercent: -50, xPercent: -50 });
      gsap.set(cursorRef.current, { autoAlpha: 0, x: stageW * 0.18, y: stageH * 0.30 });
      gsap.set(hintRef.current, { autoAlpha: 0 });
      if (hintTextRef.current) hintTextRef.current.textContent = "";
      gsap.set(selectionRef.current, {
        autoAlpha: 0,
        x: startX,
        y: startY,
        width: 0,
        height: 0,
      });
      gsap.set(flashRef.current, { autoAlpha: 0 });
      gsap.set(chatRef.current, {
        autoAlpha: 0,
        x: chatX,
        y: chatY + 10,
        scale: 0.985,
        width: chatW,
      });
      gsap.set(dimRef.current, { autoAlpha: 0 });
      gsap.set(scanBeamRef.current, { autoAlpha: 0, y: -20 });
      if (answerRef.current) answerRef.current.textContent = "";
      gsap.set(caretRef.current, { autoAlpha: 0 });

      // Reduced motion: jump to end-state composition and don't build the timeline.
      if (reduced) {
        gsap.set(lines || [], { y: "0%", filter: "blur(0px)" });
        gsap.set(eyebrowRef.current, { autoAlpha: 1 });
        if (eyebrowTextRef.current) eyebrowTextRef.current.textContent = heroEyebrow;
        gsap.set(subheadRef.current, { autoAlpha: 1, y: 0 });
        gsap.set(ctaRef.current, { autoAlpha: 1, y: 0 });
        gsap.set(hotkeyRef.current, { autoAlpha: 0 });
        gsap.set(dimRef.current, { autoAlpha: 0 });
        gsap.set(selectionRef.current, { autoAlpha: 1, width: selW, height: selH });
        gsap.set(chatRef.current, { autoAlpha: 1, scale: 1, y: chatY });
        if (connectorRef.current) gsap.set(connectorRef.current, { autoAlpha: 1, strokeDashoffset: 0 });
        if (answerRef.current) answerRef.current.innerHTML = renderInlineCode(heroAnswer);
        gsap.set(caretRef.current, { autoAlpha: 0 });
        document.documentElement.dataset.rm = "1";
        return;
      }

      // ----- MASTER TIMELINE -----
      const tl = gsap.timeline({ paused: false, repeat: -1, repeatDelay: 1.6 });
      timelineRef.current = tl;

      // Eyebrow scramble in
      tl.to(eyebrowRef.current, { autoAlpha: 1, duration: 0.2 }, 0);
      const eyebrowProg = { p: 0 };
      tl.to(
        eyebrowProg,
        {
          p: 1,
          duration: 0.55,
          ease: "none",
          onUpdate() {
            if (eyebrowTextRef.current)
              eyebrowTextRef.current.textContent = scrambleAt(heroEyebrow, eyebrowProg.p, 4);
          },
          onComplete() {
            if (eyebrowTextRef.current) eyebrowTextRef.current.textContent = heroEyebrow;
          },
        },
        0
      );

      // (1) Hero text reveal — clip-rise + blur clear
      tl.to(
        lines || [],
        { y: "0%", filter: "blur(0px)", duration: 0.95, ease: "expo.out", stagger: 0.1 },
        0.25
      );
      tl.to(subheadRef.current, { autoAlpha: 1, y: 0, duration: 0.55, ease: "power2.out" }, 0.85);
      tl.to(ctaRef.current, { autoAlpha: 1, y: 0, duration: 0.45, ease: "power2.out" }, 1.0);

      // hotkey enter
      tl.to(hotkeyRef.current, { autoAlpha: 1, scale: 1, duration: 0.36, ease: "power3.out" }, 1.4);

      // press pulse + dim
      tl.to(hotkeyRef.current, { scale: 0.92, filter: "brightness(1.4)", duration: 0.09, ease: "power1.in" }, 1.85);
      tl.to(hotkeyRef.current, { scale: 1, filter: "brightness(1)", duration: 0.18, ease: "power2.out" }, 1.94);
      tl.to(dimRef.current, { autoAlpha: 1, duration: 0.32, ease: "power2.out" }, 1.85);

      // Scan beam sweeps the dim
      tl.to(scanBeamRef.current, { autoAlpha: 1, duration: 0.1 }, 1.95);
      tl.to(scanBeamRef.current, { y: stageH + 20, duration: 0.7, ease: "power2.in" }, 1.95);
      tl.to(scanBeamRef.current, { autoAlpha: 0, duration: 0.15 }, 2.55);

      tl.to(hotkeyRef.current, { autoAlpha: 0, y: -6, duration: 0.36, ease: "power2.in" }, 2.3);

      // Selection appears at size 0; box-shadow takes over the dim
      tl.set(selectionRef.current, { autoAlpha: 1 }, 2.35);
      tl.to(dimRef.current, { autoAlpha: 0, duration: 0.25, ease: "power1.out" }, 2.35);

      // cursor + hint scramble
      tl.to(cursorRef.current, { autoAlpha: 1, duration: 0.18 }, 2.15);
      tl.to(hintRef.current, { autoAlpha: 1, duration: 0.18 }, 2.25);
      const hintProg = { p: 0 };
      tl.to(
        hintProg,
        {
          p: 1,
          duration: 0.4,
          ease: "none",
          onUpdate() {
            if (hintTextRef.current)
              hintTextRef.current.textContent = scrambleAt("Drag to select", hintProg.p, 4);
          },
          onComplete() {
            if (hintTextRef.current) hintTextRef.current.textContent = "Drag to select";
          },
        },
        2.25
      );

      // Drag (animation 2)
      tl.to(cursorRef.current, { x: endX, y: endY, duration: 1.05, ease: "power2.inOut" }, 2.4);
      tl.to(
        selectionRef.current,
        { x: startX, y: startY, width: selW, height: selH, duration: 1.05, ease: "power2.inOut" },
        2.4
      );
      tl.to(
        hintRef.current,
        { x: () => endX - startX + 14, y: () => endY - startY - 6, duration: 1.05, ease: "power2.inOut" },
        2.4
      );
      tl.to(hintRef.current, { autoAlpha: 0, duration: 0.2 }, 3.35);

      // Crop snap
      tl.to(flashRef.current, { autoAlpha: 0.22, duration: 0.06, ease: "power1.out" }, 3.45);
      tl.to(flashRef.current, { autoAlpha: 0, duration: 0.18, ease: "power2.out" }, 3.51);
      tl.to(selectionRef.current, { autoAlpha: 0, duration: 0.18 }, 3.46);
      tl.to(cursorRef.current, { autoAlpha: 0, duration: 0.2 }, 3.46);

      // (3) chat window spawn — parent app's recipe (260ms cubic-bezier ≈ power3.out)
      tl.to(
        chatRef.current,
        { autoAlpha: 1, y: chatY, scale: 1, duration: 0.32, ease: "power3.out" },
        3.6
      );

      // Connector line draws from selection to chat
      if (connectorRef.current) {
        tl.set(connectorRef.current, { autoAlpha: 1 }, 3.62);
        tl.to(
          connectorRef.current,
          { strokeDashoffset: 0, duration: 0.45, ease: "power2.out" },
          3.62
        );
      }

      // (4) AI answer scramble-type
      tl.set(caretRef.current, { autoAlpha: 1 }, 3.95);
      const ans = { p: 0 };
      tl.to(
        ans,
        {
          p: 1,
          duration: 2.6,
          ease: "none",
          onUpdate() {
            if (answerRef.current) {
              const text = scrambleAt(heroAnswer, ans.p, 7);
              answerRef.current.innerHTML = renderInlineCode(text);
            }
          },
          onComplete() {
            if (answerRef.current) answerRef.current.innerHTML = renderInlineCode(heroAnswer);
          },
        },
        4.05
      );

      // Subtle exhale
      tl.to(chatRef.current, { y: chatY - 2, duration: 0.6, ease: "sine.inOut", yoyo: true, repeat: 1 }, 6.7);

      // Hover / focus pause
      const wrap = stageWrapRef.current;
      const onPause = () => tl.pause();
      const onResume = () => tl.resume();
      if (wrap) {
        wrap.addEventListener("mouseenter", onPause);
        wrap.addEventListener("mouseleave", onResume);
        wrap.addEventListener("focusin", onPause);
        wrap.addEventListener("focusout", onResume);
      }

      // Viewport pause
      const io = new IntersectionObserver(
        (entries) => {
          entries.forEach((e) => {
            if (e.isIntersecting) tl.resume();
            else tl.pause();
          });
        },
        { threshold: 0.1 }
      );
      if (wrap) io.observe(wrap);

      // Idle parallax: stage tilts subtly with mouse position
      const onMove = (e: PointerEvent) => {
        if (!wrap) return;
        const r = wrap.getBoundingClientRect();
        const dx = (e.clientX - r.left - r.width / 2) / r.width;
        const dy = (e.clientY - r.top - r.height / 2) / r.height;
        gsap.to(wrap, { rotateX: -dy * 1.2, rotateY: dx * 1.6, duration: 0.6, ease: "power2.out" });
      };
      const onLeave = () => {
        if (!wrap) return;
        gsap.to(wrap, { rotateX: 0, rotateY: 0, duration: 0.8, ease: "power3.out" });
      };
      const hover = window.matchMedia("(hover: hover) and (pointer: fine)");
      if (hover.matches && wrap) {
        wrap.addEventListener("pointermove", onMove);
        wrap.addEventListener("pointerleave", onLeave);
      }

      return () => {
        if (wrap) {
          wrap.removeEventListener("mouseenter", onPause);
          wrap.removeEventListener("mouseleave", onResume);
          wrap.removeEventListener("focusin", onPause);
          wrap.removeEventListener("focusout", onResume);
          wrap.removeEventListener("pointermove", onMove);
          wrap.removeEventListener("pointerleave", onLeave);
        }
        io.disconnect();
      };
    },
    { scope: rootRef, dependencies: [reduced] }
  );

  return (
    <section className="sl-hero" id="top" ref={rootRef} aria-labelledby="hero-title">
      <div className="sl-container">
        <div className="sl-hero__top">
          <div className="sl-hero__copy">
            <span className="sl-hero__eyebrow" ref={eyebrowRef}>
              <span className="sl-tickmark" />
              <span ref={eyebrowTextRef}>{heroEyebrow}</span>
            </span>
            <h1 className="sl-hero__title" ref={headlineRef} id="hero-title">
              {heroHeadline.map((line, i) => (
                <span className="sl-line" key={i}>
                  <span className="sl-line__inner">
                    {i === heroHeadline.length - 1 ? (
                      <>
                        <span className="sl-hero__title-em">for</span> your screen.
                      </>
                    ) : (
                      line
                    )}
                  </span>
                </span>
              ))}
            </h1>
            <p className="sl-hero__sub" ref={subheadRef}>
              {heroSubhead}
            </p>
            <div className="sl-hero__cta" ref={ctaRef}>
              <a href="#download" className="sl-btn sl-btn--lg">
                Download for macOS
              </a>
              <span className="sl-hero__cta-meta">Apple Silicon · Universal</span>
            </div>
          </div>

          <div className="sl-hero__sidecol" aria-hidden="true">
            <StatsTicker />
          </div>
        </div>

        <div
          className="sl-hero__stage-wrap"
          ref={stageWrapRef}
          role="img"
          aria-label="Animated demonstration: pressing the hotkey, dragging a selection across the screen, and an AI answer appearing in a floating window."
        >
          <span className="sl-hero__crosshair sl-hero__crosshair--tl" aria-hidden="true" />
          <span className="sl-hero__crosshair sl-hero__crosshair--tr" aria-hidden="true" />
          <span className="sl-hero__crosshair sl-hero__crosshair--bl" aria-hidden="true" />
          <span className="sl-hero__crosshair sl-hero__crosshair--br" aria-hidden="true" />

          <FakeDesktop ref={stageRef}>
            <div className="sl-stage-dim" ref={dimRef} aria-hidden="true" />
            <div className="sl-stage-scanbeam" ref={scanBeamRef} aria-hidden="true" />
            <HotkeyChip ref={hotkeyRef} />
            <Cursor ref={cursorRef} />
            <SelectionRect ref={selectionRef} />
            <span className="sl-hint" ref={hintRef} aria-hidden="true">
              <span className="sl-hint__dot" />
              <span ref={hintTextRef}>Drag to select</span>
            </span>
            <div className="sl-flash" ref={flashRef} aria-hidden="true" />

            <svg className="sl-stage-conn" aria-hidden="true">
              <path
                ref={connectorRef}
                d=""
                stroke="rgba(255,255,255,0.55)"
                strokeWidth="1"
                strokeDasharray="2 3"
                fill="none"
              />
            </svg>

            <ChatWindow
              ref={chatRef}
              prompt={heroPrompt}
              answerRef={answerRef}
              caretRef={caretRef}
            />
          </FakeDesktop>

          <button
            className="sl-replay"
            type="button"
            onClick={() => timelineRef.current?.restart()}
            aria-label="Replay the demo"
          >
            ↻ Replay
          </button>

          <div className="sl-hero__stage-meta" aria-hidden="true">
            <span>capture · region 482×218</span>
            <span>provider · claude-sonnet-4-6</span>
          </div>
        </div>

        <p className="sl-sr-only">
          Demonstration sequence: press hotkey, drag a selection across the document chart,
          a floating window appears at the selection origin and types the AI's answer.
        </p>

        <div className="sl-hero__stage-caption" aria-hidden="true">
          <div className="sl-hero__stage-caption-row">
            <span><em>01</em> Hotkey</span>
            <span className="sl-hero__stage-caption-arrow" />
            <span><em>02</em> Drag-select</span>
            <span className="sl-hero__stage-caption-arrow" />
            <span><em>03</em> Answer</span>
          </div>
        </div>

        <Marquee />
      </div>
    </section>
  );
}

function renderInlineCode(s: string): string {
  const esc = s.replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;");
  return esc.replace(/`([^`]+)`/g, "<code>$1</code>");
}
