// Tiny glyph-scramble utility: each character cycles through random glyphs
// before settling on the target. Pure DOM-text, no DOM mutation per char.

const SCRAMBLE_CHARS = "01░▒▓▌▐■□●◐◑◒◓◢◣◤◥▲▼◀▶/\\|-_=*+~?#@&$%";

function rand(): string {
  return SCRAMBLE_CHARS[Math.floor(Math.random() * SCRAMBLE_CHARS.length)]!;
}

/**
 * Build a snapshot string at progress p (0..1) from `target`.
 * - Characters before the threshold are settled (target glyph).
 * - Characters near the threshold are scrambling (random glyph).
 * - Characters past the scramble window are blank space (preserves layout for pre-text).
 */
export function scrambleAt(target: string, p: number, scrambleWindow = 6): string {
  const len = target.length;
  const settled = Math.floor(p * len);
  let out = "";
  for (let i = 0; i < len; i++) {
    const ch = target[i]!;
    if (i < settled) {
      out += ch;
    } else if (i < settled + scrambleWindow) {
      // Preserve whitespace so wrapping doesn't shift
      out += /\s/.test(ch) ? ch : rand();
    } else {
      // Past the scramble window — empty
      out += "";
    }
  }
  return out;
}

/**
 * Scramble-reveal: animate `el.textContent` from "" → target with a scramble window.
 * `progress` is a [0..1] number you tween externally.
 */
export function applyScramble(
  el: HTMLElement,
  target: string,
  progress: number,
  scrambleWindow = 6
): void {
  el.textContent = scrambleAt(target, progress, scrambleWindow);
}
