use super::AiError;
use serde_json::{json, Map, Value};
use std::time::Duration;

const ANTHROPIC_API_URL: &str = "https://api.anthropic.com/v1/messages";
const ANTHROPIC_API_VERSION: &str = "2023-06-01";
const OPENAI_API_URL: &str = "https://api.openai.com/v1/chat/completions";
const GEMINI_API_BASE: &str = "https://generativelanguage.googleapis.com/v1beta/models";
const OLLAMA_API_URL: &str = "http://localhost:11434/api/chat";
const MAX_DECISION_TOKENS: u32 = 512;
const GEMINI_MIN_THINKING_BUDGET: i32 = 128;
const DECISION_REQUEST_MAX_ATTEMPTS: u8 = 3;
const DECISION_RETRY_BASE_DELAY_MS: u64 = 350;
/// Web-lookup research calls: server-side searches per call, answer budget,
/// and how many `pause_turn` continuations to follow before giving up.
const MAX_WEB_LOOKUP_SEARCHES: u8 = 2;
const MAX_WEB_LOOKUP_TOKENS: u32 = 1024;
const MAX_WEB_LOOKUP_CONTINUATIONS: u8 = 2;

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct DecisionPrompt {
    pub system_prompt: String,
    pub user_prompt: String,
    pub schema: Value,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct DecisionClientConfig {
    pub provider: String,
    pub model: String,
    pub api_key: String,
}

#[derive(Clone, Debug)]
pub(crate) struct DecisionClient {
    config: DecisionClientConfig,
}

impl DecisionClient {
    pub(crate) fn new(config: DecisionClientConfig) -> Self {
        Self { config }
    }

    pub(crate) async fn complete(&self, prompt: DecisionPrompt) -> Result<String, AiError> {
        match self.config.provider.as_str() {
            "anthropic" => {
                if self.config.api_key.is_empty() {
                    return Err(AiError::NoKey);
                }
                complete_anthropic(&self.config, &prompt).await
            }
            "openai" => {
                if self.config.api_key.is_empty() {
                    return Err(AiError::NoKey);
                }
                complete_openai(&self.config, &prompt).await
            }
            "gemini" => {
                if self.config.api_key.is_empty() {
                    return Err(AiError::NoKey);
                }
                complete_gemini(&self.config, &prompt).await
            }
            "ollama" => complete_ollama(&self.config, &prompt).await,
            other => Err(AiError::InvalidProvider(other.to_string())),
        }
    }

    /// Research call with the Anthropic server-side web search tool — used
    /// by the agent's webLookup action to learn where a feature lives in an
    /// app's UI. Anthropic-only; other providers report unsupported.
    pub(crate) async fn web_lookup(
        &self,
        system_prompt: &str,
        user_prompt: &str,
    ) -> Result<String, AiError> {
        if self.config.provider.as_str() != "anthropic" {
            return Err(AiError::InvalidProvider(format!(
                "web lookup requires the anthropic provider, got {}",
                self.config.provider
            )));
        }
        if self.config.api_key.is_empty() {
            return Err(AiError::NoKey);
        }

        let client = super::cloud_client()?;
        let mut messages = vec![json!({ "role": "user", "content": user_prompt })];
        // The server runs its own search loop and may pause mid-turn
        // (stop_reason "pause_turn"); re-sending the assistant content
        // resumes it. Bounded so a wedged loop can't run away.
        for _ in 0..=MAX_WEB_LOOKUP_CONTINUATIONS {
            let body = anthropic_web_lookup_body(&self.config.model, system_prompt, &messages);
            let request = client
                .post(ANTHROPIC_API_URL)
                .header("x-api-key", &self.config.api_key)
                .header("anthropic-version", ANTHROPIC_API_VERSION)
                .header("content-type", "application/json")
                .json(&body);
            let value = post_json(request, "anthropic").await?;

            if value.get("stop_reason").and_then(Value::as_str) == Some("pause_turn") {
                if let Some(content) = value.get("content") {
                    messages.push(json!({ "role": "assistant", "content": content }));
                    continue;
                }
            }
            return extract_anthropic_text(&value).ok_or_else(|| AiError::EmptyResponse {
                provider: "anthropic".into(),
            });
        }
        Err(AiError::EmptyResponse {
            provider: "anthropic".into(),
        })
    }

    pub(crate) async fn complete_vision(
        &self,
        prompt: DecisionPrompt,
        image_png_b64: &str,
    ) -> Result<String, AiError> {
        match self.config.provider.as_str() {
            "anthropic" => {
                if self.config.api_key.is_empty() {
                    return Err(AiError::NoKey);
                }
                complete_anthropic_vision(&self.config, &prompt, image_png_b64).await
            }
            "openai" => {
                if self.config.api_key.is_empty() {
                    return Err(AiError::NoKey);
                }
                complete_openai_vision(&self.config, &prompt, image_png_b64).await
            }
            "gemini" => {
                if self.config.api_key.is_empty() {
                    return Err(AiError::NoKey);
                }
                complete_gemini_vision(&self.config, &prompt, image_png_b64).await
            }
            "ollama" => complete_ollama_vision(&self.config, &prompt, image_png_b64).await,
            other => Err(AiError::InvalidProvider(other.to_string())),
        }
    }
}

async fn complete_anthropic(
    config: &DecisionClientConfig,
    prompt: &DecisionPrompt,
) -> Result<String, AiError> {
    let body = anthropic_decision_body(prompt, &config.model);
    let client = super::cloud_client()?;
    let request = client
        .post(ANTHROPIC_API_URL)
        .header("x-api-key", &config.api_key)
        .header("anthropic-version", ANTHROPIC_API_VERSION)
        .header("content-type", "application/json")
        .json(&body);
    let value = post_json(request, "anthropic").await?;

    extract_anthropic_text(&value).ok_or_else(|| AiError::EmptyResponse {
        provider: "anthropic".into(),
    })
}

async fn complete_openai(
    config: &DecisionClientConfig,
    prompt: &DecisionPrompt,
) -> Result<String, AiError> {
    let body = openai_decision_body(prompt, &config.model);
    let client = super::cloud_client()?;
    let request = client
        .post(OPENAI_API_URL)
        .bearer_auth(&config.api_key)
        .header("content-type", "application/json")
        .json(&body);
    let value = post_json(request, "openai").await?;

    extract_openai_text(&value).ok_or_else(|| AiError::EmptyResponse {
        provider: "openai".into(),
    })
}

async fn complete_gemini(
    config: &DecisionClientConfig,
    prompt: &DecisionPrompt,
) -> Result<String, AiError> {
    validate_gemini_model_id(&config.model)?;
    let body = gemini_decision_body(prompt, &config.model);
    let url = format!("{GEMINI_API_BASE}/{}:generateContent", config.model);
    let client = super::cloud_client()?;
    let request = client
        .post(url)
        .header("x-goog-api-key", &config.api_key)
        .header("content-type", "application/json")
        .json(&body);
    let value = post_json(request, "gemini").await?;

    extract_gemini_text(&value).ok_or_else(|| AiError::EmptyResponse {
        provider: "gemini".into(),
    })
}

async fn complete_ollama(
    config: &DecisionClientConfig,
    prompt: &DecisionPrompt,
) -> Result<String, AiError> {
    let body = ollama_decision_body(prompt, &config.model);
    let client = super::local_client()?;
    let request = client
        .post(OLLAMA_API_URL)
        .header("content-type", "application/json")
        .json(&body);
    let value = post_json(request, "ollama").await?;

    extract_ollama_text(&value).ok_or_else(|| AiError::EmptyResponse {
        provider: "ollama".into(),
    })
}

async fn complete_anthropic_vision(
    config: &DecisionClientConfig,
    prompt: &DecisionPrompt,
    image_png_b64: &str,
) -> Result<String, AiError> {
    let body = anthropic_vision_decision_body(prompt, &config.model, image_png_b64);
    let client = super::cloud_client()?;
    let request = client
        .post(ANTHROPIC_API_URL)
        .header("x-api-key", &config.api_key)
        .header("anthropic-version", ANTHROPIC_API_VERSION)
        .header("content-type", "application/json")
        .json(&body);
    let value = post_json(request, "anthropic").await?;

    extract_anthropic_text(&value).ok_or_else(|| AiError::EmptyResponse {
        provider: "anthropic".into(),
    })
}

async fn complete_openai_vision(
    config: &DecisionClientConfig,
    prompt: &DecisionPrompt,
    image_png_b64: &str,
) -> Result<String, AiError> {
    let body = openai_vision_decision_body(prompt, &config.model, image_png_b64);
    let client = super::cloud_client()?;
    let request = client
        .post(OPENAI_API_URL)
        .bearer_auth(&config.api_key)
        .header("content-type", "application/json")
        .json(&body);
    let value = post_json(request, "openai").await?;

    extract_openai_text(&value).ok_or_else(|| AiError::EmptyResponse {
        provider: "openai".into(),
    })
}

async fn complete_gemini_vision(
    config: &DecisionClientConfig,
    prompt: &DecisionPrompt,
    image_png_b64: &str,
) -> Result<String, AiError> {
    validate_gemini_model_id(&config.model)?;
    let body = gemini_vision_decision_body(prompt, &config.model, image_png_b64);
    let url = format!("{GEMINI_API_BASE}/{}:generateContent", config.model);
    let client = super::cloud_client()?;
    let request = client
        .post(url)
        .header("x-goog-api-key", &config.api_key)
        .header("content-type", "application/json")
        .json(&body);
    let value = post_json(request, "gemini").await?;

    extract_gemini_text(&value).ok_or_else(|| AiError::EmptyResponse {
        provider: "gemini".into(),
    })
}

async fn complete_ollama_vision(
    config: &DecisionClientConfig,
    prompt: &DecisionPrompt,
    image_png_b64: &str,
) -> Result<String, AiError> {
    let body = ollama_vision_decision_body(prompt, &config.model, image_png_b64);
    let client = super::local_client()?;
    let request = client
        .post(OLLAMA_API_URL)
        .header("content-type", "application/json")
        .json(&body);
    let value = post_json(request, "ollama").await?;

    extract_ollama_text(&value).ok_or_else(|| AiError::EmptyResponse {
        provider: "ollama".into(),
    })
}

async fn post_json(
    request: reqwest::RequestBuilder,
    provider: &'static str,
) -> Result<Value, AiError> {
    let mut request = request;
    for attempt in 1..=DECISION_REQUEST_MAX_ATTEMPTS {
        let retry_request = request.try_clone();
        let result = post_json_once(request, provider).await;
        let should_retry = retry_request.is_some()
            && attempt < DECISION_REQUEST_MAX_ATTEMPTS
            && decision_error_is_retryable(result.as_ref().err());

        if should_retry {
            eprintln!(
                "[screenie] {provider} planner request failed on attempt {attempt}; retrying"
            );
            tokio::time::sleep(Duration::from_millis(
                DECISION_RETRY_BASE_DELAY_MS * u64::from(attempt),
            ))
            .await;
            request = retry_request.expect("retry request checked above");
            continue;
        }

        return result;
    }

    unreachable!("decision request retry loop always returns")
}

async fn post_json_once(
    request: reqwest::RequestBuilder,
    provider: &'static str,
) -> Result<Value, AiError> {
    let resp = request.send().await?;
    let status = resp.status();
    let text = resp.text().await.unwrap_or_default();
    if !status.is_success() {
        return Err(AiError::Api {
            status: status.as_u16(),
            body: super::sanitize_provider_error(&text, provider),
        });
    }

    serde_json::from_str::<Value>(&text).map_err(|err| AiError::Decode(err.to_string()))
}

fn decision_error_is_retryable(err: Option<&AiError>) -> bool {
    match err {
        Some(AiError::Api { status, .. }) => *status == 429 || *status >= 500,
        Some(AiError::Http(_)) => true,
        _ => false,
    }
}

/// The system prompt is identical on every step of a run, so it is marked as
/// a cache breakpoint. Below Anthropic's minimum cacheable prefix the field
/// is silently ignored, so this is safe for short prompts too.
fn anthropic_cached_system(prompt: &DecisionPrompt) -> Value {
    json!([
        {
            "type": "text",
            "text": prompt.system_prompt,
            "cache_control": { "type": "ephemeral" }
        }
    ])
}

/// Body for the webLookup research call. Plain text out — the web_search
/// server tool is incompatible with json_schema output, and the caller
/// post-filters the answer anyway. No sampling params (removed on newer
/// models) and no cache_control (the prompt is tiny and rarely repeated).
/// `web_search_20250305` is the broadly compatible tool version; the newer
/// `web_search_20260209` requires 4.6-family models and adds nothing needed
/// for a ≤3-line navigation answer.
pub(crate) fn anthropic_web_lookup_body(
    model: &str,
    system_prompt: &str,
    messages: &[Value],
) -> Value {
    json!({
        "model": model,
        "max_tokens": MAX_WEB_LOOKUP_TOKENS,
        "stream": false,
        "system": system_prompt,
        "messages": messages,
        "tools": [
            {
                "type": "web_search_20250305",
                "name": "web_search",
                "max_uses": MAX_WEB_LOOKUP_SEARCHES
            }
        ]
    })
}

pub(crate) fn anthropic_decision_body(prompt: &DecisionPrompt, model: &str) -> Value {
    json!({
        "model": model,
        "max_tokens": MAX_DECISION_TOKENS,
        "stream": false,
        "temperature": 0,
        "system": anthropic_cached_system(prompt),
        "messages": [
            {
                "role": "user",
                "content": prompt.user_prompt
            }
        ],
        "output_config": {
            "type": "json_schema",
            "schema": prompt.schema
        }
    })
}

pub(crate) fn anthropic_vision_decision_body(
    prompt: &DecisionPrompt,
    model: &str,
    image_png_b64: &str,
) -> Value {
    json!({
        "model": model,
        "max_tokens": MAX_DECISION_TOKENS,
        "stream": false,
        "temperature": 0,
        "system": anthropic_cached_system(prompt),
        "messages": [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_png_b64
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt.user_prompt
                    }
                ]
            }
        ],
        "output_config": {
            "type": "json_schema",
            "schema": prompt.schema
        }
    })
}

pub(crate) fn openai_decision_body(prompt: &DecisionPrompt, model: &str) -> Value {
    let reasoning = is_reasoning_model(model);
    let messages = if reasoning {
        json!([
            {
                "role": "user",
                "content": format!("{}\n\n{}", prompt.system_prompt, prompt.user_prompt)
            }
        ])
    } else {
        json!([
            {
                "role": "system",
                "content": prompt.system_prompt
            },
            {
                "role": "user",
                "content": prompt.user_prompt
            }
        ])
    };

    let mut body = Map::new();
    body.insert("model".into(), json!(model));
    body.insert("stream".into(), json!(false));
    body.insert("messages".into(), messages);
    body.insert(
        "response_format".into(),
        json!({
            "type": "json_schema",
            "json_schema": {
                "name": "computer_action",
                "strict": true,
                "schema": openai_strictify(&prompt.schema)
            }
        }),
    );

    if reasoning {
        body.insert("max_completion_tokens".into(), json!(MAX_DECISION_TOKENS));
    } else {
        body.insert("temperature".into(), json!(0));
        body.insert("max_tokens".into(), json!(MAX_DECISION_TOKENS));
    }

    Value::Object(body)
}

pub(crate) fn openai_vision_decision_body(
    prompt: &DecisionPrompt,
    model: &str,
    image_png_b64: &str,
) -> Value {
    let reasoning = is_reasoning_model(model);
    let text = if reasoning {
        format!("{}\n\n{}", prompt.system_prompt, prompt.user_prompt)
    } else {
        prompt.user_prompt.clone()
    };
    let user_content = json!([
        {
            "type": "image_url",
            "image_url": {
                "url": format!("data:image/png;base64,{}", image_png_b64)
            }
        },
        {
            "type": "text",
            "text": text
        }
    ]);
    let messages = if reasoning {
        json!([
            {
                "role": "user",
                "content": user_content
            }
        ])
    } else {
        json!([
            {
                "role": "system",
                "content": prompt.system_prompt
            },
            {
                "role": "user",
                "content": user_content
            }
        ])
    };

    let mut body = Map::new();
    body.insert("model".into(), json!(model));
    body.insert("stream".into(), json!(false));
    body.insert("messages".into(), messages);
    body.insert(
        "response_format".into(),
        json!({
            "type": "json_schema",
            "json_schema": {
                "name": "computer_action",
                "strict": true,
                "schema": openai_strictify(&prompt.schema)
            }
        }),
    );

    if reasoning {
        body.insert("max_completion_tokens".into(), json!(MAX_DECISION_TOKENS));
    } else {
        body.insert("temperature".into(), json!(0));
        body.insert("max_tokens".into(), json!(MAX_DECISION_TOKENS));
    }

    Value::Object(body)
}

/// Transform a planner-authored JSON schema into OpenAI strict mode's
/// dialect: every object node requires every property (originally-optional
/// properties become nullable), `additionalProperties` is pinned false, and
/// keywords outside the supported set are dropped (length caps are enforced
/// client-side by the parser's truncation). Applied mechanically to whatever
/// schema the prompt carries, so the OpenAI request can never drift from the
/// planner contract again.
pub(crate) fn openai_strictify(schema: &Value) -> Value {
    match schema {
        Value::Object(map) => {
            let originally_required: Vec<&str> = map
                .get("required")
                .and_then(Value::as_array)
                .map(|entries| entries.iter().filter_map(Value::as_str).collect())
                .unwrap_or_default();
            let mut out = Map::new();
            for (key, entry) in map {
                match key.as_str() {
                    "properties" => {
                        let Some(properties) = entry.as_object() else {
                            continue;
                        };
                        let strict_properties: Map<String, Value> = properties
                            .iter()
                            .map(|(name, property)| {
                                let mut strict = openai_strictify(property);
                                if !originally_required.contains(&name.as_str()) {
                                    null_union_type(&mut strict);
                                }
                                (name.clone(), strict)
                            })
                            .collect();
                        out.insert("properties".into(), Value::Object(strict_properties));
                    }
                    "items" => {
                        out.insert("items".into(), openai_strictify(entry));
                    }
                    "type" | "enum" | "minimum" | "maximum" | "minItems" | "maxItems" => {
                        out.insert(key.clone(), entry.clone());
                    }
                    // required/additionalProperties are regenerated below.
                    // Everything else (maxLength, ...) is dropped: length
                    // caps are enforced client-side by the parser's
                    // truncation, so the strict schema doesn't need them.
                    _ => {}
                }
            }
            if let Some(properties) = out.get("properties").and_then(Value::as_object) {
                let names: Vec<String> = properties.keys().cloned().collect();
                out.insert("required".into(), json!(names));
                out.insert("additionalProperties".into(), json!(false));
            }
            Value::Object(out)
        }
        other => other.clone(),
    }
}

/// Strict mode's optional-field emulation: union the property's type with
/// "null". An enum value list is a closed set, so when the property carries
/// one, null is appended there too — the model omits an optional field by
/// emitting literal null, which must itself be a legal enum member.
fn null_union_type(property: &mut Value) {
    let Some(map) = property.as_object_mut() else {
        return;
    };
    match map.get("type") {
        Some(Value::String(single)) => {
            let single = single.clone();
            map.insert("type".into(), json!([single, "null"]));
        }
        Some(Value::Array(_)) => {
            if let Some(Value::Array(types)) = map.get_mut("type") {
                if !types.iter().any(|entry| entry == "null") {
                    types.push(json!("null"));
                }
            }
        }
        _ => return,
    }
    if let Some(Value::Array(values)) = map.get_mut("enum") {
        if !values.iter().any(Value::is_null) {
            values.push(Value::Null);
        }
    }
}

pub(crate) fn gemini_decision_body(prompt: &DecisionPrompt, model: &str) -> Value {
    let response_schema = gemini_schema_for_prompt(prompt);
    let generation_config = gemini_generation_config(response_schema);
    let generation_config = with_gemini_thinking_config(generation_config, model);

    json!({
        "systemInstruction": {
            "parts": [
                { "text": prompt.system_prompt }
            ]
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    { "text": prompt.user_prompt }
                ]
            }
        ],
        "generationConfig": generation_config
    })
}

pub(crate) fn gemini_vision_decision_body(
    prompt: &DecisionPrompt,
    model: &str,
    image_png_b64: &str,
) -> Value {
    let response_schema = gemini_schema_for_prompt(prompt);
    let generation_config = gemini_generation_config(response_schema);
    let generation_config = with_gemini_thinking_config(generation_config, model);

    json!({
        "systemInstruction": {
            "parts": [
                { "text": prompt.system_prompt }
            ]
        },
        "contents": [
            {
                "role": "user",
                "parts": [
                    {
                        "inline_data": {
                            "mime_type": "image/png",
                            "data": image_png_b64
                        }
                    },
                    { "text": prompt.user_prompt }
                ]
            }
        ],
        "generationConfig": generation_config
    })
}

fn gemini_generation_config(response_schema: Value) -> Map<String, Value> {
    let mut generation_config = Map::new();
    generation_config.insert("maxOutputTokens".into(), json!(MAX_DECISION_TOKENS));
    generation_config.insert("responseMimeType".into(), json!("application/json"));
    generation_config.insert("responseSchema".into(), response_schema);
    generation_config.insert("temperature".into(), json!(0));
    generation_config
}

fn with_gemini_thinking_config(
    mut generation_config: Map<String, Value>,
    model: &str,
) -> Map<String, Value> {
    if let Some(thinking_config) = gemini_thinking_config(model) {
        generation_config.insert("thinkingConfig".into(), thinking_config);
    }
    generation_config
}

fn gemini_thinking_config(model: &str) -> Option<Value> {
    let model = model.to_ascii_lowercase();
    if model.starts_with("gemini-3") {
        return Some(json!({ "thinkingLevel": "low" }));
    }

    if model.contains("gemini-2.5-pro") {
        return Some(json!({ "thinkingBudget": GEMINI_MIN_THINKING_BUDGET }));
    }

    if gemini_supports_thinking_budget_zero(&model) {
        return Some(json!({ "thinkingBudget": 0 }));
    }

    None
}

fn gemini_supports_thinking_budget_zero(model: &str) -> bool {
    model.contains("gemini-2.5-flash")
        || model.contains("robotics-er")
        || model.contains("native-audio")
}

fn gemini_schema_for_prompt(prompt: &DecisionPrompt) -> Value {
    strip_json_schema_keyword(&prompt.schema, "additionalProperties")
}

fn strip_json_schema_keyword(value: &Value, keyword: &str) -> Value {
    match value {
        Value::Object(map) => Value::Object(
            map.iter()
                .filter_map(|(key, value)| {
                    if key == keyword {
                        None
                    } else {
                        Some((key.clone(), strip_json_schema_keyword(value, keyword)))
                    }
                })
                .collect(),
        ),
        Value::Array(items) => Value::Array(
            items
                .iter()
                .map(|item| strip_json_schema_keyword(item, keyword))
                .collect(),
        ),
        _ => value.clone(),
    }
}

pub(crate) fn ollama_decision_body(prompt: &DecisionPrompt, model: &str) -> Value {
    json!({
        "model": model,
        "stream": false,
        "messages": [
            {
                "role": "system",
                "content": prompt.system_prompt
            },
            {
                "role": "user",
                "content": prompt.user_prompt
            }
        ],
        "format": prompt.schema,
        "options": {
            "temperature": 0
        }
    })
}

pub(crate) fn ollama_vision_decision_body(
    prompt: &DecisionPrompt,
    model: &str,
    image_png_b64: &str,
) -> Value {
    json!({
        "model": model,
        "stream": false,
        "messages": [
            {
                "role": "system",
                "content": prompt.system_prompt
            },
            {
                "role": "user",
                "content": prompt.user_prompt,
                "images": [image_png_b64]
            }
        ],
        "format": prompt.schema,
        "options": {
            "temperature": 0
        }
    })
}

fn extract_anthropic_text(value: &Value) -> Option<String> {
    value
        .get("content")?
        .as_array()?
        .iter()
        .filter_map(|part| part.get("text").and_then(|text| text.as_str()))
        .collect::<Vec<_>>()
        .join("")
        .trim()
        .to_string()
        .into_non_empty()
}

fn extract_openai_text(value: &Value) -> Option<String> {
    value
        .get("choices")?
        .as_array()?
        .first()?
        .get("message")?
        .get("content")?
        .as_str()
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(ToString::to_string)
}

fn extract_gemini_text(value: &Value) -> Option<String> {
    value
        .get("candidates")?
        .as_array()?
        .first()?
        .get("content")?
        .get("parts")?
        .as_array()?
        .iter()
        .filter_map(|part| part.get("text").and_then(|text| text.as_str()))
        .collect::<Vec<_>>()
        .join("")
        .trim()
        .to_string()
        .into_non_empty()
}

fn extract_ollama_text(value: &Value) -> Option<String> {
    value
        .get("message")?
        .get("content")?
        .as_str()
        .map(str::trim)
        .filter(|text| !text.is_empty())
        .map(ToString::to_string)
}

fn validate_gemini_model_id(model: &str) -> Result<(), AiError> {
    if model.is_empty()
        || !model
            .chars()
            .all(|c| c.is_ascii_lowercase() || c.is_ascii_digit() || c == '-' || c == '.')
    {
        return Err(AiError::InvalidProvider(format!(
            "invalid gemini model id: {model}"
        )));
    }
    Ok(())
}

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

trait NonEmptyString {
    fn into_non_empty(self) -> Option<String>;
}

impl NonEmptyString for String {
    fn into_non_empty(self) -> Option<String> {
        if self.is_empty() {
            None
        } else {
            Some(self)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn openai_decision_body_uses_strict_json_schema() {
        let prompt = prompt();
        let body = openai_decision_body(&prompt, "gpt-4o");

        assert_eq!(body["stream"], false);
        assert_eq!(body["response_format"]["type"], "json_schema");
        assert_eq!(body["response_format"]["json_schema"]["strict"], true);
        // The strict schema is derived from the prompt's schema, never a
        // hand-maintained copy — that copy drifted in production once.
        assert_eq!(
            body["response_format"]["json_schema"]["schema"],
            openai_strictify(&prompt.schema)
        );
        assert_eq!(body["messages"][0]["role"], "system");
        assert_eq!(body["max_tokens"], MAX_DECISION_TOKENS);
    }

    #[test]
    fn openai_strictify_requires_all_fields_and_nullifies_optionals() {
        let strict = openai_strictify(&json!({
            "type": "object",
            "additionalProperties": false,
            "required": ["reason", "action"],
            "properties": {
                "reason": { "type": "string", "maxLength": 200 },
                "action": { "type": "string", "enum": ["click", "done"] },
                "scope": { "type": "string", "enum": ["screen", "window"] },
                "id": { "type": "integer", "minimum": 0 },
                "next": {
                    "type": "array",
                    "maxItems": 3,
                    "items": {
                        "type": "object",
                        "required": ["action"],
                        "properties": {
                            "action": { "type": "string" },
                            "ms": { "type": "integer", "minimum": 0 }
                        }
                    }
                }
            }
        }));

        // Every property required, additionalProperties pinned, at every level.
        assert_eq!(
            strict["required"],
            json!(["action", "id", "next", "reason", "scope"])
        );
        assert_eq!(strict["additionalProperties"], json!(false));
        assert_eq!(
            strict["properties"]["next"]["items"]["required"],
            json!(["action", "ms"])
        );
        assert_eq!(
            strict["properties"]["next"]["items"]["additionalProperties"],
            json!(false)
        );

        // Originally-required properties keep their plain type; optional ones
        // become nullable. An optional property's enum gains null — the enum
        // is a closed set and literal null is how the model omits the field.
        assert_eq!(strict["properties"]["reason"]["type"], json!("string"));
        assert_eq!(strict["properties"]["action"]["type"], json!("string"));
        assert_eq!(
            strict["properties"]["action"]["enum"],
            json!(["click", "done"])
        );
        assert_eq!(
            strict["properties"]["scope"]["type"],
            json!(["string", "null"])
        );
        assert_eq!(
            strict["properties"]["scope"]["enum"],
            json!(["screen", "window", null])
        );
        assert_eq!(
            strict["properties"]["id"]["type"],
            json!(["integer", "null"])
        );
        assert_eq!(
            strict["properties"]["next"]["items"]["properties"]["ms"]["type"],
            json!(["integer", "null"])
        );

        // Unsupported keywords are dropped; supported bounds survive.
        assert!(strict["properties"]["reason"].get("maxLength").is_none());
        assert_eq!(strict["properties"]["id"]["minimum"], json!(0));
        assert_eq!(strict["properties"]["next"]["maxItems"], json!(3));
    }

    #[test]
    fn openai_decision_body_folds_system_for_reasoning_models() {
        let body = openai_decision_body(&prompt(), "o3-mini");

        assert_eq!(body["messages"].as_array().unwrap().len(), 1);
        assert_eq!(body["messages"][0]["role"], "user");
        assert!(body.get("max_tokens").is_none());
        assert_eq!(body["max_completion_tokens"], MAX_DECISION_TOKENS);
    }

    #[test]
    fn anthropic_decision_body_requests_json_schema_output() {
        let prompt = prompt();
        let body = anthropic_decision_body(&prompt, "claude-sonnet-4-6");

        assert_eq!(body["stream"], false);
        assert_eq!(body["output_config"]["type"], "json_schema");
        assert_eq!(body["output_config"]["schema"], prompt.schema);
        // The static system prompt is a cache breakpoint on every step.
        assert_eq!(body["system"][0]["text"], "system");
        assert_eq!(body["system"][0]["cache_control"]["type"], "ephemeral");
    }

    #[test]
    fn gemini_decision_body_requests_json_response_schema() {
        let prompt = prompt();
        let body = gemini_decision_body(&prompt, "gemini-2.5-flash");

        assert_eq!(
            body["generationConfig"]["responseMimeType"],
            "application/json"
        );
        assert_eq!(body["generationConfig"]["responseSchema"], prompt.schema);
        assert_eq!(
            body["generationConfig"]["thinkingConfig"]["thinkingBudget"],
            0
        );
    }

    #[test]
    fn gemini_decision_body_uses_model_compatible_thinking_config() {
        let prompt = prompt();

        let pro = gemini_decision_body(&prompt, "gemini-2.5-pro");
        assert_eq!(
            pro["generationConfig"]["thinkingConfig"]["thinkingBudget"],
            GEMINI_MIN_THINKING_BUDGET
        );

        let gemini3 = gemini_decision_body(&prompt, "gemini-3.5-flash");
        assert_eq!(
            gemini3["generationConfig"]["thinkingConfig"]["thinkingLevel"],
            "low"
        );

        let legacy = gemini_decision_body(&prompt, "gemini-1.5-flash");
        assert!(legacy["generationConfig"]
            .as_object()
            .unwrap()
            .get("thinkingConfig")
            .is_none());
    }

    #[test]
    fn gemini_decision_body_strips_unsupported_additional_properties() {
        let prompt = DecisionPrompt {
            schema: json!({
                "type": "object",
                "additionalProperties": false,
                "properties": {
                    "action": {
                        "type": "object",
                        "additionalProperties": false,
                        "properties": {
                            "kind": { "type": "string" }
                        }
                    }
                },
                "required": ["action"]
            }),
            ..prompt()
        };
        let body = gemini_decision_body(&prompt, "gemini-2.5-flash");
        let schema = &body["generationConfig"]["responseSchema"];

        assert!(schema.get("additionalProperties").is_none());
        assert!(schema["properties"]["action"]
            .get("additionalProperties")
            .is_none());
        assert_eq!(
            schema["properties"]["action"]["properties"]["kind"]["type"],
            "string"
        );
    }

    #[test]
    fn ollama_decision_body_uses_schema_format() {
        let prompt = prompt();
        let body = ollama_decision_body(&prompt, "llama3.2-vision");

        assert_eq!(body["stream"], false);
        assert_eq!(body["format"], prompt.schema);
        assert_eq!(body["messages"][0]["role"], "system");
    }

    #[test]
    fn vision_decision_bodies_attach_png_and_schema_controls() {
        let prompt = prompt();

        let anthropic = anthropic_vision_decision_body(&prompt, "claude-sonnet-4-6", "png123");
        assert_eq!(
            anthropic["messages"][0]["content"][0]["source"]["media_type"],
            "image/png"
        );
        assert_eq!(
            anthropic["messages"][0]["content"][0]["source"]["data"],
            "png123"
        );
        assert_eq!(anthropic["output_config"]["schema"], prompt.schema);

        let openai = openai_vision_decision_body(&prompt, "gpt-4o", "png123");
        assert_eq!(
            openai["messages"][1]["content"][0]["image_url"]["url"],
            "data:image/png;base64,png123"
        );
        assert_eq!(openai["response_format"]["json_schema"]["strict"], true);
        assert_eq!(
            openai["response_format"]["json_schema"]["schema"],
            openai_strictify(&prompt.schema)
        );

        let gemini = gemini_vision_decision_body(&prompt, "gemini-2.5-flash", "png123");
        assert_eq!(
            gemini["contents"][0]["parts"][0]["inline_data"]["mime_type"],
            "image/png"
        );
        assert_eq!(gemini["generationConfig"]["responseSchema"], prompt.schema);

        let ollama = ollama_vision_decision_body(&prompt, "llama3.2-vision", "png123");
        assert_eq!(ollama["messages"][1]["images"][0], "png123");
        assert_eq!(ollama["format"], prompt.schema);
    }

    #[test]
    fn openai_vision_control_prompt_strictifies_nested_action_schema() {
        // The grounding contract's nested action object goes through the same
        // mechanical transform — no sniffing, no parallel hand-built schema.
        let prompt = DecisionPrompt {
            schema: json!({
                "type": "object",
                "properties": {
                    "observation": { "type": "string" },
                    "target": { "type": "string" },
                    "reasoning": { "type": "string" },
                    "action": {
                        "type": "object",
                        "required": ["type"],
                        "properties": {
                            "type": { "type": "string", "enum": ["click", "done"] },
                            "x": { "type": "integer", "minimum": 0 }
                        }
                    }
                },
                "required": ["observation", "target", "reasoning", "action"]
            }),
            ..prompt()
        };
        let body = openai_vision_decision_body(&prompt, "gpt-4o", "png123");
        let schema = &body["response_format"]["json_schema"]["schema"];

        assert_eq!(
            schema["required"],
            json!(["action", "observation", "reasoning", "target"])
        );
        assert_eq!(schema["properties"]["action"]["required"], json!(["type", "x"]));
        assert_eq!(
            schema["properties"]["action"]["additionalProperties"],
            json!(false)
        );
        assert_eq!(
            schema["properties"]["action"]["properties"]["type"]["type"],
            json!("string")
        );
        assert_eq!(
            schema["properties"]["action"]["properties"]["x"]["type"],
            json!(["integer", "null"])
        );
    }

    #[test]
    fn extract_text_from_provider_shapes() {
        assert_eq!(
            extract_openai_text(&json!({
                "choices": [{"message": {"content": "{\"action\":\"done\"}"}}]
            })),
            Some("{\"action\":\"done\"}".into())
        );
        assert_eq!(
            extract_anthropic_text(&json!({
                "content": [{"type": "text", "text": "{\"action\":\"done\"}"}]
            })),
            Some("{\"action\":\"done\"}".into())
        );
        assert_eq!(
            extract_gemini_text(&json!({
                "candidates": [{"content": {"parts": [{"text": "{\"action\":\"done\"}"}]}}]
            })),
            Some("{\"action\":\"done\"}".into())
        );
        assert_eq!(
            extract_ollama_text(&json!({
                "message": {"content": "{\"action\":\"done\"}"}
            })),
            Some("{\"action\":\"done\"}".into())
        );
    }

    fn prompt() -> DecisionPrompt {
        DecisionPrompt {
            system_prompt: "system".into(),
            user_prompt: "user".into(),
            schema: json!({
                "type": "object",
                "properties": {
                    "action": { "type": "string" }
                },
                "required": ["action"]
            }),
        }
    }

    #[test]
    fn anthropic_web_lookup_body_declares_search_tool_and_no_schema_or_sampling() {
        let messages = vec![json!({
            "role": "user",
            "content": "App: Safari (com.apple.Safari). Feature: enable develop menu."
        })];
        let body = anthropic_web_lookup_body("claude-sonnet-4-6", "lookup system", &messages);

        assert_eq!(body["tools"][0]["type"], "web_search_20250305");
        assert_eq!(body["tools"][0]["name"], "web_search");
        assert_eq!(
            body["tools"][0]["max_uses"],
            u64::from(MAX_WEB_LOOKUP_SEARCHES)
        );
        assert_eq!(body["system"], "lookup system");
        assert_eq!(body["messages"], json!(messages));
        // web_search is incompatible with json_schema output, and sampling
        // params are removed on newer models.
        assert!(body.get("output_config").is_none());
        assert!(body.get("temperature").is_none());
    }
}
