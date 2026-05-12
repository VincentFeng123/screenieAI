use super::{AiError, AskEvent, AskRequest, CancelFlag, UiMessage};
use futures_util::StreamExt;
use serde_json::{json, Value};
use std::sync::atomic::Ordering;

const API_URL: &str = "https://api.anthropic.com/v1/messages";
const API_VERSION: &str = "2023-06-01";

pub async fn stream<F>(req: AskRequest, cancel: CancelFlag, mut on_event: F) -> Result<(), AiError>
where
    F: FnMut(AskEvent),
{
    if req.api_key.is_empty() {
        return Err(AiError::NoKey);
    }

    // Cloud providers bill per image token, and a Retina screenshot is
    // typically 1500–2500 tokens just for the picture. Downscaling to a
    // 1024px long edge keeps quality fine for screenshot-style content
    // and cuts cost roughly 3–4×.
    let image_b64 = crate::capture::downscale_for_cloud(&req.image_b64, 1024)
        .await
        .unwrap_or_else(|_| req.image_b64.clone());
    let messages = build_messages(&req.messages, &image_b64);
    let system = super::response_format_instructions(&req.response_profile);
    // Scale the output cap with the response-style preference. The earlier
    // hard-coded 2048 was prone to truncating "detailed" KaTeX-heavy answers
    // mid-equation.
    let max_tokens = match req.response_profile.as_str() {
        "detailed" => 4096,
        "balanced" => 3072,
        _ => 2048,
    };

    let body = json!({
        "model": req.model,
        "max_tokens": max_tokens,
        "stream": true,
        "system": system,
        "messages": messages,
    });

    let client = super::cloud_client()?;
    let resp = client
        .post(API_URL)
        .header("x-api-key", &req.api_key)
        .header("anthropic-version", API_VERSION)
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await?;

    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(AiError::Api {
            status: status.as_u16(),
            body: super::sanitize_provider_error(&text, "anthropic"),
        });
    }

    // Buffer raw bytes — never decode UTF-8 mid-chunk. Multi-byte codepoints
    // (emoji, CJK text in OCR / translation responses) routinely straddle
    // chunk boundaries, and `str::from_utf8` on a partial codepoint would
    // abort the whole stream. SSE event delimiters are ASCII, so once we have
    // a complete event we can decode it safely.
    let mut stream = resp.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    let mut input_tokens: u64 = 0;
    let mut output_tokens: u64 = 0;
    while let Some(chunk) = stream.next().await {
        if cancel.load(Ordering::Relaxed) {
            return Ok(());
        }
        buf.extend_from_slice(&chunk?);

        while let Some(event) = super::drain_sse_event(&mut buf)? {
            if cancel.load(Ordering::Relaxed) {
                return Ok(());
            }
            // Anthropic streams emit `event: error\ndata: {...}` when something
            // breaks mid-stream (overload, rate-limit, content moderation).
            // Without this branch the stream just stops with no explanation.
            // Surface it as a 200-status `Api` error so the existing toast
            // path renders it.
            if sse_event_field(&event).as_deref() == Some("error") {
                if let Some(data) = super::sse_data(&event) {
                    return Err(AiError::Api {
                        status: 200,
                        body: super::sanitize_provider_error(&data, "anthropic"),
                    });
                }
                return Err(AiError::Api {
                    status: 200,
                    body: "anthropic stream error".to_string(),
                });
            }
            if let Some(data) = super::sse_data(&event) {
                if let Some(text) = extract_text_delta(&data) {
                    on_event(AskEvent::Chunk { text });
                    continue;
                }
                // `message_start` includes prompt tokens (incl. image
                // tokens), `message_delta` includes the running output.
                if let Ok(v) = serde_json::from_str::<Value>(&data) {
                    let kind = v.get("type").and_then(|t| t.as_str()).unwrap_or("");
                    if kind == "message_start" {
                        if let Some(usage) = v.get("message").and_then(|m| m.get("usage")) {
                            if let Some(t) = usage.get("input_tokens").and_then(|n| n.as_u64()) {
                                input_tokens = t;
                            }
                            if let Some(t) = usage.get("output_tokens").and_then(|n| n.as_u64()) {
                                output_tokens = t;
                            }
                        }
                    } else if kind == "message_delta" {
                        if let Some(t) = v
                            .get("usage")
                            .and_then(|u| u.get("output_tokens"))
                            .and_then(|n| n.as_u64())
                        {
                            output_tokens = t;
                        }
                    } else if kind == "error" {
                        // Some Anthropic responses put the error inside a
                        // `data:` block typed as `error` rather than via the
                        // SSE `event:` field. Handle both shapes.
                        let msg = v
                            .get("error")
                            .and_then(|e| e.get("message"))
                            .and_then(|m| m.as_str())
                            .unwrap_or(&data);
                        return Err(AiError::Api {
                            status: 200,
                            body: super::sanitize_provider_error(msg, "anthropic"),
                        });
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

fn build_messages(history: &[UiMessage], image_b64: &str) -> Vec<Value> {
    // Attach the image to the LAST user message — that way, when the user
    // resizes the captured region between turns, the model sees the new image
    // alongside its newest question rather than answering about a stale crop.
    let last_user_idx = history
        .iter()
        .rposition(|m| m.role != "assistant")
        .unwrap_or(usize::MAX);
    let mut out = Vec::with_capacity(history.len().max(1));
    for (i, m) in history.iter().enumerate() {
        let role = if m.role == "assistant" { "assistant" } else { "user" };
        if i == last_user_idx {
            let text = if m.content.trim().is_empty() {
                "Describe what's shown in this image clearly and concisely."
            } else {
                m.content.as_str()
            };
            out.push(json!({
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_b64,
                        }
                    },
                    {"type": "text", "text": text}
                ]
            }));
        } else {
            out.push(json!({
                "role": role,
                "content": m.content,
            }));
        }
    }
    if last_user_idx == usize::MAX {
        out.push(json!({
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": "image/png",
                        "data": image_b64,
                    }
                },
                {"type": "text", "text": "Describe what's shown in this image clearly and concisely."}
            ]
        }));
    }
    out
}

/// Return the value of the `event:` field of an SSE block, if present.
/// Anthropic uses this to flag error events; OpenAI / Gemini do not.
fn sse_event_field(event: &str) -> Option<String> {
    let normalized = event.replace("\r\n", "\n").replace('\r', "\n");
    for line in normalized.lines() {
        if let Some(rest) = line.strip_prefix("event:") {
            return Some(rest.trim().to_string());
        }
    }
    None
}

fn extract_text_delta(data: &str) -> Option<String> {
    let v: Value = serde_json::from_str(data).ok()?;
    let kind = v.get("type")?.as_str()?;
    if kind != "content_block_delta" {
        return None;
    }
    let delta = v.get("delta")?;
    if delta.get("type")?.as_str()? != "text_delta" {
        return None;
    }
    delta.get("text")?.as_str().map(|s| s.to_string())
}
