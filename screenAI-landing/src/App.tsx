import { useEffect } from "react";
import { gsap } from "gsap";
import { ScrollTrigger } from "gsap/ScrollTrigger";
import { Nav } from "./components/Nav";
import { ScanlineGrid } from "./components/ScanlineGrid";
import { MagneticCrosshair } from "./components/MagneticCrosshair";
import { Hero } from "./components/Hero";
import { Features } from "./components/Features";
import { Workflow } from "./components/Workflow";
import { SpecSheet } from "./components/SpecSheet";
import { Providers } from "./components/Providers";
import { CTA } from "./components/CTA";
import { Footer } from "./components/Footer";

gsap.registerPlugin(ScrollTrigger);

export default function App() {
  useEffect(() => {
    const refresh = () => ScrollTrigger.refresh();
    const idle =
      (window as unknown as { requestIdleCallback?: (cb: () => void) => number }).requestIdleCallback ??
      ((cb: () => void) => window.setTimeout(cb, 1));
    const id = idle(refresh);
    return () => {
      if (typeof id === "number") {
        const cancel =
          (window as unknown as { cancelIdleCallback?: (id: number) => void }).cancelIdleCallback ??
          window.clearTimeout;
        cancel(id);
      }
    };
  }, []);

  return (
    <div className="sl-shell">
      <a href="#main" className="sl-skip">Skip to content</a>
      <ScanlineGrid />
      <MagneticCrosshair />
      <Nav />
      <main id="main">
        <Hero />
        <Features />
        <Workflow />
        <SpecSheet />
        <Providers />
        <CTA />
      </main>
      <Footer />
    </div>
  );
}
