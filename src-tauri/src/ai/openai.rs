use super::{AiError, AskEvent, AskRequest, CancelFlag, UiMessage};
use futures_util::StreamExt;
use serde_json::{json, Map, Value};
use std::sync::atomic::Ordering;

const API_URL: &str = "https://api.openai.com/v1/chat/completions";

/// OpenAI's reasoning-style models (o-series and GPT-5 family) reject
/// `max_tokens` and require `max_completion_tokens` instead. Some also don't
/// accept the `system` role, so the system prompt is folded into the first user
/// message. Match by model-id prefix without false-positives against `gpt-4o`.
fn is_reasoning_model(model: &str) -> bool {
    let id = model.to_ascii_lowercase();
    if id.starts_with("gpt-5") {
        return true;
    }
    matches!(id.chars().next(), Some('o'))
        && id
            .chars()
            .nth(1)
            .map(|c| c.is_ascii_digit())
            .unwrap_or(false)
}

pub async fn stream<F>(req: AskRequest, cancel: CancelFlag, mut on_event: F) -> Result<(), AiError>
where
    F: FnMut(AskEvent),
{
    if req.api_key.is_empty() {
        return Err(AiError::NoKey);
    }

    // Cap the long edge at 1024px before sending — see anthropic.rs for the
    // token-cost rationale.
    let image_b64 = crate::capture::downscale_for_cloud(&req.image_b64, 1024)
        .unwrap_or_else(|_| req.image_b64.clone());
    let reasoning = is_reasoning_model(&req.model);
    let system = super::response_format_instructions(&req.response_profile);
    let mut messages = build_messages(&req.messages, &image_b64, reasoning.then_some(&system));
    if !reasoning {
        messages.insert(
            0,
            json!({
                "role": "system",
                "content": system,
            }),
        );
    }
    let max_tokens = match req.response_profile.as_str() {
        "detailed" => 4096,
        "balanced" => 3072,
        _ => 2048,
    };

    let mut body = Map::new();
    body.insert("model".into(), json!(req.model));
    body.insert("stream".into(), json!(true));
    // Ask for usage in the final stream chunk so we can surface token
    // counts + cost in the chat panel.
    body.insert(
        "stream_options".into(),
        json!({ "include_usage": true }),
    );
    body.insert("messages".into(), json!(messages));
    if reasoning {
        // o-series rejects `max_tokens`; the field is `max_completion_tokens`.
        // Reasoning eats output tokens silently, so give them more headroom.
        body.insert(
            "max_completion_tokens".into(),
            json!(max_tokens.max(4096) * 2),
        );
    } else {
        body.insert("max_tokens".into(), json!(max_tokens));
    }
    let body = Value::Object(body);

    let client = super::cloud_client()?;
    let resp = client
        .post(API_URL)
        .bearer_auth(&req.api_key)
        .header("content-type", "application/json")
        .json(&body)
        .send()
        .await?;

    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(AiError::Api {
            status: status.as_u16(),
            body: text,
        });
    }

    // See anthropic.rs for the rationale: buffer raw bytes so a multi-byte
    // codepoint split across chunks doesn't break the stream.
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
            if let Some(data) = super::sse_data(&event) {
                if data == "[DONE]" {
                    if input_tokens > 0 || output_tokens > 0 {
                        on_event(AskEvent::Usage {
                            input_tokens,
                            output_tokens,
                        });
                    }
                    return Ok(());
                }
                if let Ok(v) = serde_json::from_str::<Value>(&data) {
                    if let Some(usage) = v.get("usage") {
                        if let Some(t) = usage.get("prompt_tokens").and_then(|n| n.as_u64()) {
                            input_tokens = t;
                        }
                        if let Some(t) = usage.get("completion_tokens").and_then(|n| n.as_u64())
                        {
                            output_tokens = t;
                        }
                    }
                }
                if let Some(text) = extract_content_delta(&data) {
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

fn build_messages(history: &[UiMessage], image_b64: &str, prefix_system: Option<&str>) -> Vec<Value> {
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
            // For o-series we fold the system prompt into the (single) user
            // turn — the API rejects role:"system".
            let text = match prefix_system {
                Some(sys) => format!("{}\n\n---\n\n{}", sys, text),
                None => text.to_string(),
            };
            out.push(json!({
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": format!("data:image/png;base64,{}", image_b64)
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
        let fallback = "Describe what's shown in this image clearly and concisely.";
        let text = match prefix_system {
            Some(sys) => format!("{}\n\n---\n\n{}", sys, fallback),
            None => fallback.to_string(),
        };
        out.push(json!({
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {
                        "url": format!("data:image/png;base64,{}", image_b64)
                    }
                },
                {"type": "text", "text": text}
            ]
        }));
    }
    out
}

fn extract_content_delta(data: &str) -> Option<String> {
    let v: Value = serde_json::from_str(data).ok()?;
    let content = v
        .get("choices")?
        .as_array()?
        .first()?
        .get("delta")?
        .get("content")?
        .as_str()?;
    Some(content.to_string())
}
