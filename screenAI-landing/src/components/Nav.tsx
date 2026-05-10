import { useEffect, useState } from "react";

export function Nav() {
  const [scrolled, setScrolled] = useState(false);

  useEffect(() => {
    const onScroll = () => setScrolled(window.scrollY > 8);
    onScroll();
    window.addEventListener("scroll", onScroll, { passive: true });
    return () => window.removeEventListener("scroll", onScroll);
  }, []);

  return (
    <header className="sl-nav" data-scrolled={scrolled ? "1" : "0"}>
      <div className="sl-container sl-nav__inner">
        <a href="#top" className="sl-brand" aria-label="Screenie AI home">
          <span className="sl-brand__mark" aria-hidden="true" />
          <span className="sl-brand__name">
            <b>Screenie</b>
            <span>AI</span>
          </span>
        </a>

        <nav className="sl-nav__links" aria-label="Primary">
          <a href="#features">Features</a>
          <a href="#workflow">How it works</a>
          <a href="#providers">Providers</a>
        </nav>

        <a href="#download" className="sl-btn">Download</a>
      </div>
    </header>
  );
}
