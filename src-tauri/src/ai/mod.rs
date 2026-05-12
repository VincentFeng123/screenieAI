use serde::{Deserialize, Serialize, Serializer};
use std::sync::atomic::AtomicBool;
use std::sync::Arc;
use std::time::Duration;

/// Event emitted to the frontend during a streaming AI request. The
/// `Chunk` variant carries response text fragments; `Usage` is emitted
/// once at end-of-stream when the provider reports token counts.
///
/// The renderer's chat panel parses this discriminated union so it can
/// (a) accumulate streamed text and (b) attach the final usage tuple to
/// the assistant message bubble for display + monthly-total bookkeeping.
#[derive(Debug, Clone, Serialize)]
#[serde(tag = "type", rename_all = "camelCase")]
pub enum AskEvent {
    Chunk { text: String },
    Usage {
        input_tokens: u64,
        output_tokens: u64,
    },
}

pub mod anthropic;
pub mod gemini;
pub mod ollama;
pub mod openai;

/// Shared cancellation flag passed into each provider's stream function.
/// `lib.rs` replaces this on every new `ask_ai` (cancelling any predecessor)
/// and trips it on `close_overlay`. Stream loops poll it between chunks.
pub type CancelFlag = Arc<AtomicBool>;

// v3 system prompt — math formatting is OPT-IN, not the default. The earlier
// v2 prompt led the model to inject equations into non-math answers because
// the math rules dominated the prompt. v3 reframes: plain prose is the
// default, math syntax is unlocked only when the content is genuinely
// mathematical. The KaTeX-specific rules + forbidden constructs + macro
// list are unchanged in substance, but they're gated behind "when math is
// actually warranted".
const BASE_RESPONSE_FORMAT_INSTRUCTIONS: &str = "\
The output is rendered as Markdown with KaTeX math support.

DEFAULT MODE — plain prose. Short paragraphs (1–3 sentences). Standard Markdown for lists, **bold**, *italic*, `code`, and headings. NO math notation, NO LaTeX delimiters, NO equations.

DO NOT introduce math, formulas, equations, or LaTeX when the question is not about math. Examples that need NO math: explaining English text, summarizing a screenshot, identifying objects, writing/proofreading prose, casual questions, opinions, descriptions, instructions. Answer those in normal prose.

USE math formatting ONLY when the user's question or the screenshot's content is genuinely mathematical or quantitative — e.g. a math problem, a physics/stats/finance derivation, a chemistry equation, a code snippet involving formulas. If you are unsure, default to plain prose.

When math IS warranted, follow these rules strictly (the renderer will fail otherwise):

Math delimiters:
- $...$ ONLY for a single bare symbol or variable in running prose (e.g. $x$, $\\pi$, $a_n$, $E$). ONE token, no operators, no relations.
- $$...$$ for everything else: any equation, formula, calculation, derivation step, numerical result, substitution, or expression with operators.
- Every $$...$$ block lives on its OWN LINE, with one BLANK LINE before AND one BLANK LINE after.
- NEVER put two equations on the same line.
- A multi-step derivation = one $$...$$ block per step, each on its own line, each separated by a blank line.
- For grouped multi-line equations, use:
$$
\\begin{aligned}
... \\\\
... \\\\
\\end{aligned}
$$
(use `aligned`, not `align`; rows separated by \\\\).
- Always brace multi-character super/subscripts: write $x^{12}$ not $x^12$, $a_{ij}$ not $a_ij$.

Forbidden when emitting math — KaTeX rejects these:
- \\(...\\) and \\[...\\] — use $...$ and $$...$$ instead.
- \\begin{equation}, eqnarray, multline, gather, align, alignat — use $$...$$ with aligned/gathered instead.
- \\label, \\ref, \\eqref, \\notag, \\nonumber.
- \\documentclass, \\usepackage, TikZ, raw HTML, document-level commands.
- Do not wrap LaTeX in fenced code blocks unless the user explicitly asks for raw LaTeX source.

Available shorthand macros (predefined — use freely when emitting math):
\\R \\N \\Z \\Q \\C \\F \\E \\P (blackboard letters), \\eps and \\veps (varepsilon), \\norm{x}, \\abs{x}, \\inner{x}, \\ip{x}{y}, \\set{x}, \\dd and \\diff (upright d), \\del (partial), \\argmin, \\argmax, \\Tr, \\tr, \\rank, \\diag, \\sign, \\Var, \\Cov, \\vec{x} (bold), \\mat{X} (bold), \\T (transpose), \\qty{...} (auto-sized parens). Anything else must be standard KaTeX-supported syntax.";

pub(crate) fn response_format_instructions(profile: &str) -> String {
    let detail = match profile {
        "balanced" => {
            "Response detail: use enough explanation to be useful, but keep paragraphs short and avoid filler."
        }
        "detailed" => {
            "Response detail: include fuller step-by-step reasoning when it helps, while keeping each paragraph and equation block easy to scan."
        }
        _ => {
            "Response detail: answer directly and briefly. Prefer the minimum explanation that solves the user's prompt."
        }
    };

    format!("{BASE_RESPONSE_FORMAT_INSTRUCTIONS}\n\n{detail}")
}

/* =============================================================================
 * Legacy v1 prompt — kept for reference. The v2 prompt above replaces it.
 * If v2 misbehaves in production, swap the active path back by uncommenting
 * the v1 constant below and commenting out the v2 constant above. The v1
 * text is preserved verbatim from before the rewrite.
 * =============================================================================
 *
 * const BASE_RESPONSE_FORMAT_INSTRUCTIONS_V1: &str = "\
 * Format answers for a small screen overlay rendered with KaTeX. The renderer does almost no post-processing — emit KaTeX-clean markdown directly.
 *
 * Math formatting (mandatory — these are not suggestions):
 * - Use $...$ ONLY for a single short symbol or variable in running prose (e.g. $x$, $E$, $\\pi$, $a_n$). One symbol, no operators.
 * - ANY equation, formula, derivation step, calculation, substitution, or numerical result that is more than a single symbol MUST live in its own $$...$$ block, on its own line, with a BLANK LINE before AND after the block. No exceptions.
 * - NEVER place two equations on the same line. NEVER place an equation inline with prose if it's more than one symbol — break it out.
 * - A multi-step derivation = one $$...$$ block per step, each on its own line, each separated by a blank line. Do NOT cram multiple steps into one block unless they share a single \\begin{aligned} environment.
 * - For grouped multi-line equations, use $$\\begin{aligned} ... \\end{aligned}$$ (use `aligned`, not `align`). Each row separated by \\\\.
 * - Use ONLY KaTeX-supported syntax. NEVER emit \\[...\\], \\(...\\), \\label, \\ref, \\eqref, \\notag, \\nonumber, \\mathds, \\bm, \\mbox, \\hbox, eqnarray, multline, equation, gather (use $$...$$ + aligned/gathered instead). Always brace multi-character super/subscripts: write $x^{12}$ not $x^12$, $a_{ij}$ not $a_ij$.
 * - Do not wrap LaTeX in fenced code blocks unless the user explicitly asks for raw LaTeX source.
 *
 * Prose formatting:
 * - Keep paragraphs short (1-3 sentences).
 * - Add a blank line between paragraphs and around every $$...$$ block — generous vertical spacing aids legibility on the small overlay.
 * - Label key steps clearly (e.g. \"Step 1:\", \"Result:\") on their own line above the equation block.";
 */

#[derive(Debug, Clone, Deserialize)]
pub struct UiMessage {
    /// "user" or "assistant"
    pub role: String,
    pub content: String,
}

#[derive(Debug, thiserror::Error)]
pub enum AiError {
    #[error("invalid provider: {0}")]
    InvalidProvider(String),
    #[error("request too large: {0}")]
    RequestTooLarge(String),
    #[error("keyring: {0}")]
    Keyring(String),
    #[error("api key missing")]
    NoKey,
    #[error("http: {0}")]
    Http(String),
    #[error("api: {status} — {body}")]
    Api { status: u16, body: String },
    #[error("decode: {0}")]
    Decode(String),
    #[error("{provider} returned no text chunks")]
    EmptyResponse { provider: String },
}

impl Serialize for AiError {
    fn serialize<S: Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_str(&self.to_string())
    }
}

impl From<reqwest::Error> for AiError {
    fn from(e: reqwest::Error) -> Self {
        AiError::Http(e.to_string())
    }
}

#[derive(Debug, Clone)]
pub struct AskRequest {
    pub api_key: String,
    pub model: String,
    pub response_profile: String,
    /// Conversation turns. The image is attached by each provider to the
    /// first user message; subsequent messages are text-only.
    pub messages: Vec<UiMessage>,
    /// PNG bytes encoded as base64 (no `data:` prefix).
    pub image_b64: String,
}

pub(crate) fn cloud_client() -> Result<reqwest::Client, AiError> {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(10))
        .read_timeout(Duration::from_secs(90))
        .timeout(Duration::from_secs(300))
        .build()
        .map_err(AiError::from)
}

pub(crate) fn local_client() -> Result<reqwest::Client, AiError> {
    reqwest::Client::builder()
        .connect_timeout(Duration::from_secs(3))
        .read_timeout(Duration::from_secs(180))
        .timeout(Duration::from_secs(600))
        .build()
        .map_err(AiError::from)
}

/// Drain one complete Server-Sent Event from `buf`.
///
/// Providers do not agree on line endings: some streams use `\n\n`, while
/// others use `\r\n\r\n`. Treat both forms (plus lone-CR framing) as valid so
/// a successful request cannot silently finish without parsed chunks.
pub(crate) fn drain_sse_event(buf: &mut Vec<u8>) -> Result<Option<String>, AiError> {
    let Some((end, delimiter_len)) = find_sse_boundary(buf) else {
        return Ok(None);
    };

    let event_bytes = buf.drain(..end + delimiter_len).collect::<Vec<u8>>();
    let event_bytes = &event_bytes[..end];
    let event =
        std::str::from_utf8(event_bytes).map_err(|e| AiError::Decode(e.to_string()))?;
    Ok(Some(event.to_string()))
}

/// Return the data payload for one SSE event, applying the SSE rule that
/// multiple `data:` lines in the same event are joined by newlines.
pub(crate) fn sse_data(event: &str) -> Option<String> {
    let normalized = event.replace("\r\n", "\n").replace('\r', "\n");
    let mut lines = Vec::new();

    for line in normalized.lines() {
        if let Some(rest) = line.strip_prefix("data:") {
            lines.push(rest.strip_prefix(' ').unwrap_or(rest).to_string());
        }
    }

    if lines.is_empty() {
        None
    } else {
        Some(lines.join("\n"))
    }
}

fn find_sse_boundary(buf: &[u8]) -> Option<(usize, usize)> {
    // Spec-compliant SSE event delimiters only: `\n\n` and `\r\n\r\n`. The
    // earlier inclusion of `\r\r` was overly permissive — a JSON-encoded `\r`
    // inside a `data:` payload could otherwise fragment the event.
    [b"\r\n\r\n".as_slice(), b"\n\n".as_slice()]
        .iter()
        .filter_map(|delimiter| find_subseq(buf, delimiter).map(|pos| (pos, delimiter.len())))
        .min_by_key(|(pos, _)| *pos)
}

fn find_subseq(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).position(|w| w == needle)
}

/// Reduce a provider's raw (non-200 or mid-stream) error body to a short,
/// human-readable string suitable for the user-facing toast.
///
/// Why we sanitize on the Rust side:
/// - Some providers echo the request's `Authorization` header or `x-api-key`
///   into their error JSON for debugging. Forwarding that verbatim into the
///   toast would print `sk-...` on screen.
/// - Provider error envelopes are noisy (`request_id`, `type`, nested
///   `details[]`, full HTML 5xx pages). The user only wants the message.
/// - Defense in depth: even though the renderer also parses JSON, this
///   guarantees no secret-shaped substring escapes via the tooltip path or
///   any future consumer that prints `AiError` directly.
pub(crate) fn sanitize_provider_error(body: &str, provider: &str) -> String {
    let extracted = extract_provider_message(body, provider).unwrap_or_else(|| {
        body.lines()
            .map(|l| strip_html_tags(l).trim().to_string())
            .find(|l| !l.is_empty())
            .unwrap_or_default()
    });
    let scrubbed = scrub_api_keys(&extracted);
    let collapsed = collapse_whitespace(&scrubbed);
    truncate_chars(&collapsed, 200)
}

fn extract_provider_message(body: &str, provider: &str) -> Option<String> {
    if let Ok(v) = serde_json::from_str::<serde_json::Value>(body) {
        if let Some(s) = v
            .get("error")
            .and_then(|e| e.get("message"))
            .and_then(|m| m.as_str())
        {
            return Some(s.to_string());
        }
        if let Some(s) = v.get("message").and_then(|m| m.as_str()) {
            return Some(s.to_string());
        }
        // Gemini sometimes wraps errors in a top-level array.
        if let Some(arr) = v.as_array() {
            if let Some(s) = arr
                .first()
                .and_then(|e| e.get("error"))
                .and_then(|e| e.get("message"))
                .and_then(|m| m.as_str())
            {
                return Some(s.to_string());
            }
        }
        if let Some(s) = v.get("error").and_then(|e| e.as_str()) {
            return Some(s.to_string());
        }
    }
    let _ = provider;
    None
}

fn scrub_api_keys(s: &str) -> String {
    redact_prefix(&redact_prefix(s, "sk-", 20), "AIza", 30)
}

fn redact_prefix(s: &str, prefix: &str, min_tail_len: usize) -> String {
    let bytes = s.as_bytes();
    let prefix_bytes = prefix.as_bytes();
    let mut out = String::with_capacity(s.len());
    let mut i = 0;
    while i < bytes.len() {
        if i + prefix_bytes.len() <= bytes.len() && &bytes[i..i + prefix_bytes.len()] == prefix_bytes {
            let mut j = i + prefix_bytes.len();
            while j < bytes.len() {
                let c = bytes[j];
                let is_keychar = c.is_ascii_alphanumeric() || c == b'_' || c == b'-';
                if !is_keychar {
                    break;
                }
                j += 1;
            }
            if j - (i + prefix_bytes.len()) >= min_tail_len {
                out.push_str(prefix);
                out.push_str("***");
                i = j;
                continue;
            }
        }
        out.push(char::from(bytes[i]));
        i += 1;
    }
    // The byte-by-byte fallback corrupts non-ASCII. Return original if no
    // redaction happened to preserve UTF-8.
    if !out.contains("***") {
        return s.to_string();
    }
    out
}

fn strip_html_tags(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut in_tag = false;
    for c in s.chars() {
        match c {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(c),
            _ => {}
        }
    }
    out
}

fn collapse_whitespace(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut prev_space = false;
    for c in s.chars() {
        if c.is_whitespace() {
            if !prev_space {
                out.push(' ');
            }
            prev_space = true;
        } else {
            out.push(c);
            prev_space = false;
        }
    }
    out.trim().to_string()
}

fn truncate_chars(s: &str, max: usize) -> String {
    if s.chars().count() <= max {
        return s.to_string();
    }
    let truncated: String = s.chars().take(max.saturating_sub(1)).collect();
    format!("{truncated}…")
}
