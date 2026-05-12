use super::{AiError, AskEvent, AskRequest, CancelFlag, UiMessage};
use serde_json::{json, Value};
use std::sync::atomic::Ordering;

const API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta/models";

pub async fn stream<F>(req: AskRequest, cancel: CancelFlag, mut on_event: F) -> Result<(), AiError>
where
    F: FnMut(AskEvent),
{
    if req.api_key.is_empty() {
        return Err(AiError::NoKey);
    }

    // Same image-token cost rationale as the other cloud providers — see
    // anthropic.rs.
    let image_b64 = crate::capture::downscale_for_cloud(&req.image_b64, 1024)
        .await
        .unwrap_or_else(|_| req.image_b64.clone());
    let contents = build_contents(&req.messages, &image_b64);
    let system = super::response_format_instructions(&req.response_profile);

    // Gemini 2.5 models are "thinking" models — by default they spend output
    // tokens on internal reasoning before emitting visible text. With a 2048
    // cap, the entire budget can be consumed by thinking, leaving zero `text`
    // parts in the stream (a 200 OK with no chunks). `thinkingBudget: 0`
    // disables thinking; raise the ceiling too in case the response itself
    // is long. Older 1.x / 2.0 models silently ignore `thinkingConfig`.
    let body = json!({
        "systemInstruction": {
            "parts": [
                { "text": system }
            ]
        },
        "contents": contents,
        "generationConfig": {
            "maxOutputTokens": 8192,
            "thinkingConfig": {
                "thinkingBudget": 0
            }
        }
    });

    // Defense-in-depth: `req.model` is interpolated into the request URL.
    // Gemini SKU ids are always lowercase letters, digits, hyphens, and dots
    // (e.g. `gemini-2.5-flash`). Reject anything else before it reaches
    // `format!` so a hostile/malformed model string can't smuggle in `/`,
    // `?`, `#`, or `:` to redirect the POST or inject query params against
    // `generativelanguage.googleapis.com`.
    if req.model.is_empty()
        || !req
            .model
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '.')
    {
        return Err(AiError::InvalidProvider(format!(
            "invalid gemini model id: {}",
            req.model
        )));
    }

    // `alt=sse` opts into Server-Sent Events framing instead of Google's
    // default JSON-array streaming, which lets us reuse the same `\n\n`
    // event-boundary parser shape as the Anthropic / OpenAI clients.
    let url = format!("{API_BASE}/{}:streamGenerateContent?alt=sse", req.model);

    let client = super::cloud_client()?;
    let resp = client
        .post(&url)
        .header("x-goog-api-key", &req.api_key)
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await?;

    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(AiError::Api {
            status: status.as_u16(),
            body: super::sanitize_provider_error(&text, "gemini"),
        });
    }

    // Buffer raw bytes — see anthropic.rs for the multi-byte-codepoint pitfall.
    let mut stream = resp.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    let mut input_tokens: u64 = 0;
    let mut output_tokens: u64 = 0;
    // P-E-R8: see openai.rs — cancel-aware chunk wait drops idle streams
    // on overlay close within ~50 ms instead of after the read timeout.
    while let Some(chunk) = super::cancel_aware_next(&mut stream, &cancel).await {
        if cancel.load(Ordering::Relaxed) {
            return Ok(());
        }
        buf.extend_from_slice(&chunk?);

        while let Some(event) = super::drain_sse_event(&mut buf)? {
            if cancel.load(Ordering::Relaxed) {
                return Ok(());
            }
            if let Some(data) = super::sse_data(&event) {
                if let Ok(v) = serde_json::from_str::<Value>(&data) {
                    // Gemini's `alt=sse` stream may yield
                    // `data: {"error": {"code": ..., "message": ...}}` partway
                    // through a response on quota / safety failures. Without
                    // this branch the stream silently dies after the partial
                    // text already sent.
                    if let Some(err) = v.get("error") {
                        let msg = err
                            .get("message")
                            .and_then(|m| m.as_str())
                            .unwrap_or(&data);
                        return Err(AiError::Api {
                            status: 200,
                            body: super::sanitize_provider_error(msg, "gemini"),
                        });
                    }
                    if let Some(meta) = v.get("usageMetadata") {
                        if let Some(t) = meta.get("promptTokenCount").and_then(|n| n.as_u64()) {
                            input_tokens = t;
                        }
                        if let Some(t) = meta.get("candidatesTokenCount").and_then(|n| n.as_u64())
                        {
                            output_tokens = t;
                        }
                    }
                }
                if let Some(text) = extract_text(&data) {
                    if !text.is_empty() {
                        on_event(AskEvent::Chunk { text });
                    }
                }
            }
        }
    }
    if input_tokens > 0 || output_tokens > 0 {
        on_event(AskEvent::Usage {
            input_tokens,
            output_tokens,
        });
    }

    Ok(())
}

fn build_contents(history: &[UiMessage], image_b64: &str) -> Vec<Value> {
    // Image attached to the LAST user turn (matches the other providers'
    // behaviour: re-cropping mid-conversation works correctly).
    let last_user_idx = history
        .iter()
        .rposition(|m| m.role != "assistant")
        .unwrap_or(usize::MAX);
    let mut out = Vec::with_capacity(history.len().max(1));
    for (i, m) in history.iter().enumerate() {
        // Gemini uses "model" for the assistant role (not "assistant").
        let role = if m.role == "assistant" { "model" } else { "user" };
        if i == last_user_idx {
            let text = if m.content.trim().is_empty() {
                "Describe what's shown in this image clearly and concisely."
            } else {
                m.content.as_str()
            };
            out.push(json!({
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image_b64,
                        }
                    },
                    { "text": text }
                ]
            }));
        } else {
            out.push(json!({
                "role": role,
                "parts": [{ "text": m.content }]
            }));
        }
    }
    if last_user_idx == usize::MAX {
        out.push(json!({
            "role": "user",
            "parts": [
                {
                    "inline_data": {
                        "mime_type": "image/png",
                        "data": image_b64,
                    }
                },
                { "text": "Describe what's shown in this image clearly and concisely." }
            ]
        }));
    }
    out
}

fn extract_text(data: &str) -> Option<String> {
    let v: Value = serde_json::from_str(data).ok()?;
    let candidates = v.get("candidates")?.as_array()?;
    let first = candidates.first()?;
    let parts = first.get("content")?.get("parts")?.as_array()?;
    let mut combined = String::new();
    for part in parts {
        if let Some(t) = part.get("text").and_then(|x| x.as_str()) {
            combined.push_str(t);
        }
    }
    if combined.is_empty() {
        None
    } else {
        Some(combined)
    }
}
