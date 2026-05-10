export function Footer() {
  const year = new Date().getFullYear();
  return (
    <footer className="sl-footer">
      <div className="sl-container">
        <div className="sl-footer__row">
          <span>Screenie AI · © {year}</span>
          <div className="sl-footer__links">
            <a href="#">GitHub</a>
            <a href="#">Privacy</a>
            <a href="#">Contact</a>
          </div>
        </div>
      </div>
    </footer>
  );
}
