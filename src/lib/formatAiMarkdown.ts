/**
 * v2 AI markdown / LaTeX → KaTeX pipeline.
 *
 * Replaces the 518-line v1 normalizer with a small, deterministic pass that
 * trusts the system prompt to produce KaTeX-clean output and only fixes the
 * three or four mistakes a well-prompted model still makes:
 *   • `\(...\)` and `\[...\]` (the React/markdown stack expects `$...$` /
 *     `$$...$$`).
 *   • `\begin{equation}...\end{equation}` (KaTeX has no equation env).
 *   • Inline `$...$` that actually contains a real equation rather than a
 *     bare symbol — promoted to a display block on its own line so the
 *     equation isn't run-on with prose. Single-token references like `$x$`
 *     or `$\pi$` stay inline.
 *   • Single-line `$$...$$` embedded mid-prose — broken out so each
 *     display block lives on its own line with blank padding above/below.
 *
 * Public API is unchanged: `SCREENIE_KATEX_OPTIONS` and `formatAiMarkdown`
 * keep their v1 names + signatures so callers (Overlay.tsx + Chat.tsx)
 * don't move.
 */

export const SCREENIE_KATEX_OPTIONS = {
  strict: false as const,
  throwOnError: false,
  errorColor: "currentColor",
  trust: false,
  // Shorthand macros carried over from v1. The system prompt advertises
  // these so AI responses can lean on them without rendering errors.
  macros: {
    "\\R": "\\mathbb{R}",
    "\\N": "\\mathbb{N}",
    "\\Z": "\\mathbb{Z}",
    "\\Q": "\\mathbb{Q}",
    "\\C": "\\mathbb{C}",
    "\\F": "\\mathbb{F}",
    "\\E": "\\mathbb{E}",
    "\\P": "\\mathbb{P}",
    "\\eps": "\\varepsilon",
    "\\veps": "\\varepsilon",
    "\\phi": "\\varphi",
    "\\norm": "\\lVert #1 \\rVert",
    "\\abs": "\\lvert #1 \\rvert",
    "\\set": "\\{ #1 \\}",
    "\\inner": "\\langle #1 \\rangle",
    "\\ip": "\\langle #1, #2 \\rangle",
    "\\dd": "\\,\\mathrm{d}",
    "\\diff": "\\,\\mathrm{d}",
    "\\del": "\\partial",
    "\\argmin": "\\operatorname*{arg\\,min}",
    "\\argmax": "\\operatorname*{arg\\,max}",
    "\\Tr": "\\operatorname{Tr}",
    "\\tr": "\\operatorname{tr}",
    "\\rank": "\\operatorname{rank}",
    "\\diag": "\\operatorname{diag}",
    "\\sign": "\\operatorname{sign}",
    "\\Var": "\\operatorname{Var}",
    "\\Cov": "\\operatorname{Cov}",
    "\\vec": "\\mathbf{#1}",
    "\\mat": "\\mathbf{#1}",
    "\\T": "^{\\mathsf{T}}",
    "\\qty": "\\left( #1 \\right)",
  },
};

/**
 * Streaming-safe + idempotent. Run on every render of an assistant bubble.
 * If the input is mid-stream and ends inside an open `$$` block, that block
 * stays open in the output — KaTeX renders nothing for that node until the
 * close arrives, which is what we want.
 */
export function formatAiMarkdown(input: string): string {
  let s = input.replace(/\r\n?/g, "\n");

  // ASCII-fy quotes. KaTeX's prime parser only accepts ASCII apostrophes,
  // so `y’(t)`-style derivatives need the smart quote rewritten first.
  s = s.replace(/[‘’]/g, "'").replace(/[“”]/g, '"');

  // `\( ... \)` → `$ ... $` (inline math).
  s = s.replace(/\\\(([\s\S]*?)\\\)/g, (_m, body) => `$${body.trim()}$`);

  // `\[ ... \]` → display block on its own line.
  s = s.replace(
    /\\\[([\s\S]*?)\\\]/g,
    (_m, body) => `\n\n$$\n${body.trim()}\n$$\n\n`,
  );

  // `\begin{equation}` / `\begin{equation*}` → `$$ ... $$` block.
  s = s.replace(
    /\\begin\{equation\*?\}([\s\S]*?)\\end\{equation\*?\}/g,
    (_m, body) => `\n\n$$\n${body.trim()}\n$$\n\n`,
  );

  // Promote inline `$...$` that looks like an actual equation into a
  // display block. Bare single-token references (`$x$`, `$\pi$`, `$a_n$`)
  // fail looksLikeEquation and stay inline so prose still reads naturally.
  s = s.replace(
    /(?<!\$)\$([^$\n]+?)\$(?!\$)/g,
    (match, body: string) =>
      looksLikeEquation(body) ? `\n\n$$\n${body.trim()}\n$$\n\n` : match,
  );

  // Pull any single-line `$$...$$` patterns embedded in prose out into
  // their own bare display block. Multi-line `$$\n...\n$$` blocks (already
  // proper display) aren't matched because the regex disallows newlines
  // inside the delimiters.
  s = breakOutInlineDisplay(s);

  // Walk every `$$ … $$` block and (a) trim blank lines INSIDE it (a blank
  // line inside math is treated by remark-math as a paragraph break, which
  // ends the math block early and causes KaTeX to render nothing), (b)
  // ensure a single blank line on the OUTSIDE of each delimiter so the
  // block is recognized as block-level. Replaces v1's two-regex
  // "force blank lines around $$" pair, which fired inside blocks too.
  s = repairDisplayBlocks(s);

  // Strip orphan punctuation that ends up adrift after equations get
  // promoted to display blocks. Two cases:
  //   • "The energy is $E$." → "." lands on its own line.
  //   • "The energy is $E$, where m is mass." → ", where..." starts the
  //     next paragraph.
  // Both read as floating punctuation and look broken in the rendered
  // bubble.
  s = stripOrphanPunctuation(s);

  // Collapse runs of 4+ newlines to 3 so the bubble doesn't grow huge gaps.
  return s.replace(/\n{4,}/g, "\n\n\n");
}

/**
 * Drop orphan punctuation that the promote step pushes adrift:
 *   1. A line whose content (after trim) is purely terminal punctuation
 *      (`.,;:!?`) — happens when the AI ended a sentence with an inline
 *      equation and the period got separated by the promote.
 *   2. A line that starts with punctuation + whitespace right after a
 *      `$$` close — happens when the AI continued the sentence after
 *      the equation. The leading punctuation+space is stripped; the
 *      rest of the line stays.
 */
function stripOrphanPunctuation(s: string): string {
  // Step 1: drop pure-punctuation lines.
  const lines = s.split("\n");
  const filtered = lines.filter(
    (line) => !/^[.,;:!?]+\s*$/.test(line.trim()),
  );
  const joined = filtered.join("\n");

  // Step 2: strip leading punctuation+whitespace right after a `$$` close.
  // `\$\$\n+` consumes the closing delimiter and any blank lines between
  // it and the next paragraph; `[.,;:!?]+[ \t]+` is the orphan prefix to
  // drop. The replacement keeps the math close + blank lines intact.
  return joined.replace(/(\$\$\n+)[.,;:!?]+[ \t]+/g, "$1");
}

/**
 * Walk lines looking for bare `$$` delimiter pairs. For each block:
 *   - Trim blank lines inside the body (so remark-math doesn't end the
 *     math early on a paragraph break).
 *   - Ensure one blank line above the opening `$$` and below the closing
 *     `$$`, but only when the surrounding context isn't already blank.
 * Unclosed blocks (mid-stream) pass through unchanged so partial responses
 * still render the prose around the unfinished math.
 */
function repairDisplayBlocks(s: string): string {
  const lines = s.split("\n");
  const out: string[] = [];
  const isBareDelim = (line: string) => /^\s*\$\$\s*$/.test(line);
  let i = 0;

  while (i < lines.length) {
    if (!isBareDelim(lines[i])) {
      out.push(lines[i]);
      i++;
      continue;
    }

    // Find the matching close. Bound the search at the document end.
    let close = -1;
    for (let j = i + 1; j < lines.length; j++) {
      if (isBareDelim(lines[j])) {
        close = j;
        break;
      }
    }

    if (close === -1) {
      // Mid-stream / unclosed — let it pass and keep going.
      out.push(lines[i]);
      i++;
      continue;
    }

    // Trim blank lines on either side of the body.
    let bodyStart = i + 1;
    let bodyEnd = close - 1;
    while (bodyStart <= bodyEnd && lines[bodyStart].trim() === "") bodyStart++;
    while (bodyEnd >= bodyStart && lines[bodyEnd].trim() === "") bodyEnd--;

    // Pre-pad: ensure a blank line above the opening `$$`.
    if (out.length > 0 && out[out.length - 1].trim() !== "") {
      out.push("");
    }
    out.push("$$");
    for (let k = bodyStart; k <= bodyEnd; k++) out.push(lines[k]);
    out.push("$$");

    // Post-pad: ensure a blank line below the closing `$$`. We push the
    // pad now; the next loop iteration will push the real next line on
    // top of it.
    i = close + 1;
    if (i < lines.length && lines[i].trim() !== "") {
      out.push("");
    }
  }

  return out.join("\n");
}

/**
 * Heuristic: does the body of an inline `$...$` look like a real equation
 * (relations, fractions, sums, multi-token operators) rather than a bare
 * symbol reference?
 *
 * Rule:
 *   - Contains a relation (`= < > ≈ ≤ ≥`) → equation.
 *   - Contains an equation-shaped LaTeX command (`\frac`, `\sum`, `\int`,
 *     `\sqrt`, `\lim`, `\cdot`, …) → equation.
 *   - Has an arithmetic operator (`+`, `-`, `*`, `/`) between non-trivial
 *     content AND is more than a few characters long → equation.
 *   - Otherwise → bare symbol, leave inline.
 */
function looksLikeEquation(body: string): boolean {
  if (/[=<>≈≤≥]/.test(body)) return true;
  if (
    /\\(?:frac|dfrac|tfrac|sum|prod|int|sqrt|lim|cdot|times|div|to|approx|leq|geq|begin)\b/.test(
      body,
    )
  ) {
    return true;
  }
  if (body.length > 4 && /[+\-*/]/.test(body) && /[a-zA-Z0-9]/.test(body)) {
    // Avoid promoting hyphenated words like "$x$-axis" — the inner body
    // would be just `x`, which fails the length check anyway.
    return true;
  }
  return false;
}

/**
 * Pull every single-line `$$...$$` (anywhere on the line) out into its own
 * bare `$$ ... $$` display block. Handles:
 *   - `$$x$$`                 → bare display block
 *   - `Use $$x$$ here`        → "Use\n\n$$\nx\n$$\n\nhere"
 *   - `$$x$$ and $$y$$`       → two consecutive blocks
 * Multi-line `$$\n...\n$$` blocks (already proper display) aren't matched
 * because the inner regex disallows newlines.
 */
function breakOutInlineDisplay(s: string): string {
  return s
    .split("\n")
    .map((line) => {
      // Bare delimiter line — leave alone.
      if (/^\s*\$\$\s*$/.test(line)) return line;

      const re = /\$\$([^\n$]+?)\$\$/g;
      const matches = [...line.matchAll(re)];
      if (matches.length === 0) return line;

      const segments: string[] = [];
      let last = 0;
      for (const m of matches) {
        const idx = m.index ?? 0;
        const before = line.slice(last, idx);
        if (before.trim()) segments.push(before.trimEnd());
        segments.push(`$$\n${m[1].trim()}\n$$`);
        last = idx + m[0].length;
      }
      const tail = line.slice(last);
      if (tail.trim()) segments.push(tail.trimStart());

      return segments.join("\n\n");
    })
    .join("\n");
}

/* =============================================================================
 * Legacy v1 pipeline — kept for reference. The v2 pipeline above replaces it.
 * If v2 misbehaves in production, swap the active path back by uncommenting
 * the v1 functions below and commenting out the v2 implementations. The v1
 * code is preserved verbatim from before the rewrite.
 * =============================================================================
 */

/*
export const SCREENIE_KATEX_OPTIONS_V1 = {
  strict: false as const,
  throwOnError: false,
  errorColor: "currentColor",
  trust: false,
  macros: {
    "\\R": "\\mathbb{R}",
    "\\N": "\\mathbb{N}",
    "\\Z": "\\mathbb{Z}",
    "\\Q": "\\mathbb{Q}",
    "\\C": "\\mathbb{C}",
    "\\F": "\\mathbb{F}",
    "\\E": "\\mathbb{E}",
    "\\P": "\\mathbb{P}",
    "\\eps": "\\varepsilon",
    "\\veps": "\\varepsilon",
    "\\phi": "\\varphi",
    "\\norm": "\\lVert #1 \\rVert",
    "\\abs": "\\lvert #1 \\rvert",
    "\\set": "\\{ #1 \\}",
    "\\inner": "\\langle #1 \\rangle",
    "\\ip": "\\langle #1, #2 \\rangle",
    "\\dd": "\\,\\mathrm{d}",
    "\\diff": "\\,\\mathrm{d}",
    "\\del": "\\partial",
    "\\argmin": "\\operatorname*{arg\\,min}",
    "\\argmax": "\\operatorname*{arg\\,max}",
    "\\Tr": "\\operatorname{Tr}",
    "\\tr": "\\operatorname{tr}",
    "\\rank": "\\operatorname{rank}",
    "\\diag": "\\operatorname{diag}",
    "\\sign": "\\operatorname{sign}",
    "\\Var": "\\operatorname{Var}",
    "\\Cov": "\\operatorname{Cov}",
    "\\vec": "\\mathbf{#1}",
    "\\mat": "\\mathbf{#1}",
    "\\T": "^{\\mathsf{T}}",
    "\\qty": "\\left( #1 \\right)",
  },
};

export function formatAiMarkdownV1(markdown: string): string {
  return normalizeAndRepairV1(markdown).replace(/\n{4,}/g, "\n\n\n");
}

function normalizeAndRepairV1(s: string): string {
  let input = s.replace(/\r\n/g, "\n").replace(/\r/g, "\n");

  input = input
    .replace(/[‘’]/g, "'")
    .replace(/[“”]/g, '"')
    .replace(/\\\[([\s\S]+?)\\\]/g, (_, body) =>
      `\n\n$$\n${body.trim()}\n$$\n\n`,
    )
    .replace(/\\\(([\s\S]+?)\\\)/g, (_, body) => `$${body.trim()}$`)
    .replace(
      /\\begin\{align\*?\}([\s\S]+?)\\end\{align\*?\}/g,
      (_, body) =>
        `\n\n$$\n\\begin{aligned}${body.trim()}\\end{aligned}\n$$\n\n`,
    )
    .replace(
      /\\begin\{alignat\*?\}\{?\d*\}?([\s\S]+?)\\end\{alignat\*?\}/g,
      (_, body) =>
        `\n\n$$\n\\begin{aligned}${body.trim()}\\end{aligned}\n$$\n\n`,
    )
    .replace(
      /\\begin\{equation\*?\}([\s\S]+?)\\end\{equation\*?\}/g,
      (_, body) => `\n\n$$\n${body.trim()}\n$$\n\n`,
    )
    .replace(
      /\\begin\{eqnarray\*?\}([\s\S]+?)\\end\{eqnarray\*?\}/g,
      (_, body) => {
        const fixed = (body as string)
          .split("\n")
          .map((line: string) => line.replace(/&\s*([=<>])\s*&/g, "$1 &"))
          .join("\n");
        return `\n\n$$\n\\begin{aligned}${fixed.trim()}\\end{aligned}\n$$\n\n`;
      },
    )
    .replace(
      /\\begin\{multline\*?\}([\s\S]+?)\\end\{multline\*?\}/g,
      (_, body) =>
        `\n\n$$\n\\begin{aligned}${body.trim()}\\end{aligned}\n$$\n\n`,
    )
    .replace(
      /\\begin\{gather\*?\}([\s\S]+?)\\end\{gather\*?\}/g,
      (_, body) =>
        `\n\n$$\n\\begin{gathered}${body.trim()}\\end{gathered}\n$$\n\n`,
    )
    .replace(/\\(?:notag|nonumber)\b\s* /g, "")
    .replace(/\\label\{[^}]*\}/g, "")
    .replace(/\\eqref\{([^}]*)\}/g, "($1)")
    .replace(/\\ref\{([^}]*)\}/g, "$1")
    .replace(/\\mathds\b/g, "\\mathbb")
    .replace(/\\(?:mbox|hbox)\{/g, "\\text{")
    .replace(/\\bm\b/g, "\\boldsymbol")
    .replace(/\\overbracket\b/g, "\\overbrace")
    .replace(/\\underbracket\b/g, "\\underbrace")
    .replace(/\\\\\\\\ /g, "\\\\");

  // … remaining v1 helpers (wrapBareAlignedBlocks, rescueDollarBlocks,
  // breakOutDisplayMath, promoteEquationLines, repairDisplayMathBlocks)
  // were ~400 additional lines and are intentionally elided from this
  // legacy block. Recover them from git history at commit before v2 if
  // a full revert is needed.
  return input;
}
*/
