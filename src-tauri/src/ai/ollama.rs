use super::{AiError, AskEvent, AskRequest, CancelFlag, UiMessage};
use futures_util::StreamExt;
use serde::Serialize;
use serde_json::{json, Value};
use std::sync::atomic::Ordering;
use std::sync::OnceLock;

const API_URL: &str = "http://localhost:11434/api/chat";
const TAGS_URL: &str = "http://localhost:11434/api/tags";

/// Cached "we have confirmed localhost:11434 is genuinely Ollama" verdict for
/// the process lifetime. We only ever cache the positive case so a transient
/// failure (Ollama starting up, port temporarily owned by another process)
/// can be retried; once verified, subsequent requests skip the banner check
/// to avoid doubling the request count on the chat hot path.
static OLLAMA_BANNER_VERIFIED: OnceLock<bool> = OnceLock::new();

/// Probe `/api/tags` and confirm the response shape matches what the real
/// Ollama daemon returns: a JSON object with a `models` field that is an
/// array. A different process bound to 11434 (port collision with another
/// local AI tool) would fail this check, preventing us from sending prompts
/// + image data to the wrong endpoint.
async fn verify_ollama_banner(client: &reqwest::Client) -> Result<(), AiError> {
    if OLLAMA_BANNER_VERIFIED.get().copied().unwrap_or(false) {
        return Ok(());
    }
    let resp = client.get(TAGS_URL).send().await.map_err(|e| {
        AiError::Http(format!(
            "ollama not reachable at localhost:11434 ({})",
            e
        ))
    })?;
    if !resp.status().is_success() {
        return Err(AiError::Http(format!(
            "localhost:11434 is not Ollama (unexpected banner: HTTP {})",
            resp.status().as_u16()
        )));
    }
    let v: Value = resp.json().await.map_err(|_| {
        AiError::Http("localhost:11434 is not Ollama (unexpected banner)".into())
    })?;
    let looks_like_ollama = v.get("models").and_then(|m| m.as_array()).is_some();
    if !looks_like_ollama {
        return Err(AiError::Http(
            "localhost:11434 is not Ollama (unexpected banner)".into(),
        ));
    }
    let _ = OLLAMA_BANNER_VERIFIED.set(true);
    Ok(())
}

pub async fn stream<F>(req: AskRequest, cancel: CancelFlag, mut on_event: F) -> Result<(), AiError>
where
    F: FnMut(AskEvent),
{
    let messages = build_messages(&req.messages, &req.image_b64, &req.response_profile);

    let body = json!({
        "model": req.model,
        "stream": true,
        "messages": messages,
    });

    let client = super::local_client()?;
    verify_ollama_banner(&client).await?;
    let resp = match client.post(API_URL).json(&body).send().await {
        Ok(r) => r,
        Err(e) => {
            return Err(AiError::Http(format!(
                "ollama not reachable at localhost:11434 ({})",
                e
            )));
        }
    };

    let status = resp.status();
    if !status.is_success() {
        let text = resp.text().await.unwrap_or_default();
        return Err(AiError::Api {
            status: status.as_u16(),
            body: super::sanitize_provider_error(&text, "ollama"),
        });
    }

    // Buffer raw bytes — local Ollama responses can absolutely contain CJK /
    // emoji that splits codepoints across chunks. See anthropic.rs.
    let mut stream = resp.bytes_stream();
    let mut buf: Vec<u8> = Vec::new();
    while let Some(chunk) = stream.next().await {
        if cancel.load(Ordering::Relaxed) {
            return Ok(());
        }
        buf.extend_from_slice(&chunk?);

        while let Some(end) = buf.iter().position(|&b| b == b'\n') {
            let line_bytes = buf.drain(..end + 1).collect::<Vec<u8>>();
            let line_bytes = &line_bytes[..end];
            let line = std::str::from_utf8(line_bytes)
                .map_err(|e| AiError::Decode(e.to_string()))?
                .trim();
            if line.is_empty() {
                continue;
            }
            if let Ok(v) = serde_json::from_str::<Value>(line) {
                // Ollama emits `{"error": "..."}` lines mid-stream on partial
                // failures (model unloaded, OOM during decode). Without this
                // the stream just stops with no toast.
                if let Some(err) = v.get("error").and_then(|e| e.as_str()) {
                    return Err(AiError::Api {
                        status: 200,
                        body: super::sanitize_provider_error(err, "ollama"),
                    });
                }
                if let Some(text) = v
                    .get("message")
                    .and_then(|m| m.get("content"))
                    .and_then(|c| c.as_str())
                {
                    if !text.is_empty() {
                        on_event(AskEvent::Chunk {
                            text: text.to_string(),
                        });
                    }
                }
                if v.get("done").and_then(|d| d.as_bool()).unwrap_or(false) {
                    let input_tokens = v
                        .get("prompt_eval_count")
                        .and_then(|n| n.as_u64())
                        .unwrap_or(0);
                    let output_tokens = v
                        .get("eval_count")
                        .and_then(|n| n.as_u64())
                        .unwrap_or(0);
                    if input_tokens > 0 || output_tokens > 0 {
                        on_event(AskEvent::Usage {
                            input_tokens,
                            output_tokens,
                        });
                    }
                    return Ok(());
                }
            }
        }
    }

    Ok(())
}

fn build_messages(history: &[UiMessage], image_b64: &str, response_profile: &str) -> Vec<Value> {
    let last_user_idx = history
        .iter()
        .rposition(|m| m.role != "assistant")
        .unwrap_or(usize::MAX);
    let mut out = Vec::with_capacity(history.len().max(1) + 1);
    let system = super::response_format_instructions(response_profile);
    out.push(json!({
        "role": "system",
        "content": system,
    }));
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
                "content": text,
                "images": [image_b64],
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
            "content": "Describe what's shown in this image clearly and concisely.",
            "images": [image_b64],
        }));
    }
    out
}

#[derive(Debug, Serialize)]
pub struct OllamaStatus {
    pub running: bool,
    pub models: Vec<String>,
}

impl OllamaStatus {
    pub fn down() -> Self {
        Self {
            running: false,
            models: Vec::new(),
        }
    }
}

pub async fn check_status() -> OllamaStatus {
    let client = match reqwest::Client::builder()
        .timeout(std::time::Duration::from_millis(500))
        .build()
    {
        Ok(c) => c,
        Err(_) => return OllamaStatus::down(),
    };
    let resp = match client.get(TAGS_URL).send().await {
        Ok(r) => r,
        Err(_) => return OllamaStatus::down(),
    };
    if !resp.status().is_success() {
        return OllamaStatus::down();
    }
    let v: Value = match resp.json().await {
        Ok(v) => v,
        Err(_) => return OllamaStatus::down(),
    };
    let models = v
        .get("models")
        .and_then(|m| m.as_array())
        .map(|arr| {
            arr.iter()
                .filter_map(|item| item.get("name").and_then(|n| n.as_str()).map(String::from))
                .collect()
        })
        .unwrap_or_default();
    OllamaStatus {
        running: true,
        models,
    }
}
