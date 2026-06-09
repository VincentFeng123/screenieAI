use super::executor::ClickPoint;
use base64::{engine::general_purpose::STANDARD, Engine as _};
use image::{imageops, DynamicImage, ImageFormat, RgbaImage};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::collections::HashSet;
use std::io::Cursor;
use std::sync::{Arc, Mutex};
use std::time::Instant;

pub(crate) const DEFAULT_GROUNDER_ENDPOINT: &str = "http://127.0.0.1:8080/v1/chat/completions";
pub(crate) const DEFAULT_GROUNDER_HEALTH_URL: &str = "http://127.0.0.1:8080/models";
pub(crate) const DEFAULT_GROUNDER_MODEL: &str = "mlx-community/LocateAnything-3B-4bit";
pub(crate) const DEFAULT_GROUNDER_CONFIDENCE_THRESHOLD: f64 = 0.50;

const COARSE_MAX_DIM: u32 = 768;
const CROP_ZOOM_MAX_DIM: u32 = 768;
const MIN_CROP_SIDE: u32 = 160;

#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct GrounderConfig {
    pub enabled: bool,
    pub endpoint: String,
    pub health_url: String,
    pub model: String,
}

impl GrounderConfig {
    pub(crate) fn key(&self) -> String {
        format!("{}|{}|{}", self.endpoint, self.health_url, self.model)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub enum GroundingMode {
    OnePass,
    CoarseToFine,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GroundingPixel {
    pub x: u32,
    pub y: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize)]
#[serde(rename_all = "camelCase")]
pub struct GroundingReport {
    pub target_text: String,
    pub model: String,
    pub endpoint: String,
    pub mode: GroundingMode,
    pub latency_ms: u64,
    pub confidence: Option<f64>,
    pub confidence_threshold: f64,
    pub raw_output_convention: Option<String>,
    pub screenshot_pixel: Option<GroundingPixel>,
    pub final_screen_click_point: Option<ClickPoint>,
    pub failure_reason: Option<String>,
}

impl GroundingReport {
    pub(crate) fn is_low_confidence(&self) -> bool {
        self.confidence
            .is_some_and(|confidence| confidence < self.confidence_threshold)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GroundingRequest {
    pub image_png_b64: String,
    pub image_width: u32,
    pub image_height: u32,
    pub target_text: String,
    pub mode: GroundingMode,
    pub confidence_threshold: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GroundingResponse {
    pub report: GroundingReport,
    pub point: Option<GroundingPixel>,
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GroundedPoint {
    pub x: u32,
    pub y: u32,
    pub confidence: Option<f64>,
    pub raw_output_convention: String,
}

#[async_trait::async_trait]
pub(crate) trait Grounder: Send + Sync {
    async fn ground(&self, request: GroundingRequest) -> GroundingResponse;
}

#[derive(Clone, Copy, Debug, Default)]
#[cfg(test)]
pub(crate) struct NoopGrounder;

#[cfg(test)]
#[async_trait::async_trait]
impl Grounder for NoopGrounder {
    async fn ground(&self, request: GroundingRequest) -> GroundingResponse {
        GroundingResponse {
            report: GroundingReport {
                target_text: request.target_text,
                model: "none".into(),
                endpoint: "none".into(),
                mode: request.mode,
                latency_ms: 0,
                confidence: None,
                confidence_threshold: request.confidence_threshold,
                raw_output_convention: None,
                screenshot_pixel: None,
                final_screen_click_point: None,
                failure_reason: Some("grounding is not configured".into()),
            },
            point: None,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct LocalHttpGrounder {
    client: Client,
    config: GrounderConfig,
}

impl LocalHttpGrounder {
    fn with_client(client: Client, config: GrounderConfig) -> Self {
        Self { client, config }
    }

    async fn ground_one_pass(
        &self,
        image_png_b64: &str,
        image_width: u32,
        image_height: u32,
        target_text: &str,
    ) -> Result<GroundedPoint, String> {
        let raw = self
            .complete_grounding(image_png_b64, image_width, image_height, target_text)
            .await?;
        parse_grounding_output(&raw, image_width, image_height)
    }

    async fn ground_coarse_to_fine(
        &self,
        request: &GroundingRequest,
    ) -> Result<GroundedPoint, String> {
        let original = decode_png_b64(&request.image_png_b64)?;
        let original_width = original.width();
        let original_height = original.height();
        let coarse = resize_max_dim(&original, COARSE_MAX_DIM)?;
        let coarse_b64 = encode_png_b64(&coarse)?;
        let first = self
            .ground_one_pass(
                &coarse_b64,
                coarse.width(),
                coarse.height(),
                &request.target_text,
            )
            .await?;
        let approx_x = scale_coord(first.x, coarse.width(), original_width);
        let approx_y = scale_coord(first.y, coarse.height(), original_height);
        let crop = crop_around_point(&original, approx_x, approx_y);
        let zoom = resize_max_dim(&crop.image, CROP_ZOOM_MAX_DIM)?;
        let zoom_b64 = encode_png_b64(&zoom)?;
        let second = self
            .ground_one_pass(&zoom_b64, zoom.width(), zoom.height(), &request.target_text)
            .await?;
        let crop_x = scale_coord(second.x, zoom.width(), crop.width);
        let crop_y = scale_coord(second.y, zoom.height(), crop.height);
        Ok(GroundedPoint {
            x: crop
                .x
                .saturating_add(crop_x)
                .min(original_width.saturating_sub(1)),
            y: crop
                .y
                .saturating_add(crop_y)
                .min(original_height.saturating_sub(1)),
            confidence: second.confidence.or(first.confidence),
            raw_output_convention: format!(
                "coarse-to-fine:{}->{}",
                first.raw_output_convention, second.raw_output_convention
            ),
        })
    }

    async fn complete_grounding(
        &self,
        image_png_b64: &str,
        image_width: u32,
        image_height: u32,
        target_text: &str,
    ) -> Result<String, String> {
        let system_prompt = [
            "You locate UI targets in a screenshot.",
            "Return exactly one compact JSON object, no markdown.",
            r#"Use {"point":[x,y],"confidence":0.0-1.0} or {"box":[x1,y1,x2,y2],"confidence":0.0-1.0}."#,
            "Coordinates may be image pixels, normalized 0..1 values, or 0..1000 coordinate tokens.",
        ]
        .join("\n");
        let user_text = format!(
            "Screen screenshot size: {image_width}x{image_height}.\nLocate the center of: {target_text}"
        );
        let body = json!({
            "model": self.config.model,
            "temperature": 0,
            "max_tokens": 256,
            "messages": [
                {"role": "system", "content": system_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": user_text},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": format!("data:image/png;base64,{image_png_b64}")
                            }
                        }
                    ]
                }
            ]
        });
        let response = self
            .client
            .post(&self.config.endpoint)
            .json(&body)
            .send()
            .await
            .map_err(|err| format!("grounder request failed: {err}"))?;
        let status = response.status();
        let text = response
            .text()
            .await
            .map_err(|err| format!("grounder response read failed: {err}"))?;
        if !status.is_success() {
            return Err(format!(
                "grounder returned HTTP {status}: {}",
                compact(&text)
            ));
        }
        chat_completion_content(&text)
    }
}

#[async_trait::async_trait]
impl Grounder for LocalHttpGrounder {
    async fn ground(&self, request: GroundingRequest) -> GroundingResponse {
        let started = Instant::now();
        if !self.config.enabled {
            return self.response(started, request, None, Some("grounding is disabled".into()));
        }

        let result = match request.mode {
            GroundingMode::OnePass => {
                self.ground_one_pass(
                    &request.image_png_b64,
                    request.image_width,
                    request.image_height,
                    &request.target_text,
                )
                .await
            }
            GroundingMode::CoarseToFine => self.ground_coarse_to_fine(&request).await,
        };

        match result {
            Ok(point) => self.response(started, request, Some(point), None),
            Err(err) => self.response(started, request, None, Some(err)),
        }
    }
}

impl LocalHttpGrounder {
    fn response(
        &self,
        started: Instant,
        request: GroundingRequest,
        point: Option<GroundedPoint>,
        failure_reason: Option<String>,
    ) -> GroundingResponse {
        let latency_ms = started.elapsed().as_millis().min(u64::MAX as u128) as u64;
        let screenshot_pixel = point.as_ref().map(|point| GroundingPixel {
            x: point.x,
            y: point.y,
        });
        let confidence = point.as_ref().and_then(|point| point.confidence);
        let raw_output_convention = point
            .as_ref()
            .map(|point| point.raw_output_convention.clone());
        eprintln!(
            "[screenie] grounder endpoint={} model={} mode={:?} latency_ms={} success={} confidence={:?}",
            self.config.endpoint,
            self.config.model,
            request.mode,
            latency_ms,
            failure_reason.is_none(),
            confidence
        );
        GroundingResponse {
            report: GroundingReport {
                target_text: request.target_text,
                model: self.config.model.clone(),
                endpoint: self.config.endpoint.clone(),
                mode: request.mode,
                latency_ms,
                confidence,
                confidence_threshold: request.confidence_threshold,
                raw_output_convention,
                screenshot_pixel,
                final_screen_click_point: None,
                failure_reason,
            },
            point: screenshot_pixel,
        }
    }
}

#[derive(Clone, Debug)]
pub(crate) struct GrounderManager {
    client: Client,
    warmed: Arc<Mutex<HashSet<String>>>,
}

impl Default for GrounderManager {
    fn default() -> Self {
        Self {
            client: Client::new(),
            warmed: Arc::new(Mutex::new(HashSet::new())),
        }
    }
}

impl GrounderManager {
    pub(crate) fn local_http_grounder(&self, config: GrounderConfig) -> LocalHttpGrounder {
        self.warm_once(config.clone());
        LocalHttpGrounder::with_client(self.client.clone(), config)
    }

    fn warm_once(&self, config: GrounderConfig) {
        if !config.enabled || config.health_url.trim().is_empty() {
            return;
        }
        let key = config.key();
        {
            let mut warmed = self.warmed.lock().unwrap_or_else(|err| err.into_inner());
            if !warmed.insert(key) {
                return;
            }
        }
        let client = self.client.clone();
        tokio::spawn(async move {
            let started = Instant::now();
            let result = client.get(&config.health_url).send().await;
            let latency_ms = started.elapsed().as_millis().min(u64::MAX as u128) as u64;
            match result {
                Ok(response) => eprintln!(
                    "[screenie] grounder warm endpoint={} model={} health={} status={} latency_ms={}",
                    config.endpoint,
                    config.model,
                    config.health_url,
                    response.status(),
                    latency_ms
                ),
                Err(err) => eprintln!(
                    "[screenie] grounder warm endpoint={} model={} health={} failed={} latency_ms={}",
                    config.endpoint,
                    config.model,
                    config.health_url,
                    err,
                    latency_ms
                ),
            }
        });
    }
}

pub(crate) fn parse_grounding_output(
    raw_output: &str,
    image_width: u32,
    image_height: u32,
) -> Result<GroundedPoint, String> {
    if image_width == 0 || image_height == 0 {
        return Err("image dimensions must be non-zero".into());
    }

    if let Ok(value) = serde_json::from_str::<Value>(raw_output.trim()) {
        if let Some(candidate) = candidate_from_value(&value) {
            return normalize_candidate(candidate, image_width, image_height);
        }
    }

    let numbers = scan_numbers(raw_output);
    if numbers.len() >= 4 && raw_output.to_ascii_lowercase().contains("box") {
        normalize_candidate(
            Candidate::Box {
                values: [numbers[0], numbers[1], numbers[2], numbers[3]],
                confidence: None,
                convention_prefix: "text_box".into(),
            },
            image_width,
            image_height,
        )
    } else if numbers.len() >= 2 {
        normalize_candidate(
            Candidate::Point {
                x: numbers[0],
                y: numbers[1],
                confidence: None,
                convention_prefix: "text_point".into(),
            },
            image_width,
            image_height,
        )
    } else {
        Err("grounder output did not contain a point or box".into())
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Candidate {
    Point {
        x: f64,
        y: f64,
        confidence: Option<f64>,
        convention_prefix: String,
    },
    Box {
        values: [f64; 4],
        confidence: Option<f64>,
        convention_prefix: String,
    },
}

fn candidate_from_value(value: &Value) -> Option<Candidate> {
    match value {
        Value::Object(map) => {
            let confidence = number_field(map, &["confidence", "score"]);
            if let (Some(x), Some(y)) = (
                map.get("x").and_then(number_value),
                map.get("y").and_then(number_value),
            ) {
                return Some(Candidate::Point {
                    x,
                    y,
                    confidence,
                    convention_prefix: "json_xy".into(),
                });
            }
            for field in ["point", "coordinate", "coordinates", "center"] {
                if let Some(candidate) = map
                    .get(field)
                    .and_then(|value| point_from_value(value, confidence, field))
                {
                    return Some(candidate);
                }
            }
            for field in ["box", "bbox", "bounds", "bounding_box", "boundingBox"] {
                if let Some(candidate) = map
                    .get(field)
                    .and_then(|value| box_from_value(value, confidence, field))
                {
                    return Some(candidate);
                }
            }
            for field in ["result", "location", "target"] {
                if let Some(candidate) = map.get(field).and_then(candidate_from_value) {
                    return Some(candidate);
                }
            }
            None
        }
        Value::Array(items) => {
            if let Some(candidate) = point_from_value(value, None, "json_array") {
                return Some(candidate);
            }
            if items.len() >= 4 {
                box_from_value(value, None, "json_array")
            } else {
                None
            }
        }
        _ => None,
    }
}

fn point_from_value(value: &Value, confidence: Option<f64>, prefix: &str) -> Option<Candidate> {
    match value {
        Value::Array(items) if items.len() >= 2 => Some(Candidate::Point {
            x: number_value(&items[0])?,
            y: number_value(&items[1])?,
            confidence,
            convention_prefix: prefix.into(),
        }),
        Value::Object(map) => Some(Candidate::Point {
            x: number_value(map.get("x")?)?,
            y: number_value(map.get("y")?)?,
            confidence: confidence.or_else(|| number_field(map, &["confidence", "score"])),
            convention_prefix: prefix.into(),
        }),
        _ => None,
    }
}

fn box_from_value(value: &Value, confidence: Option<f64>, prefix: &str) -> Option<Candidate> {
    match value {
        Value::Array(items) if items.len() >= 4 => Some(Candidate::Box {
            values: [
                number_value(&items[0])?,
                number_value(&items[1])?,
                number_value(&items[2])?,
                number_value(&items[3])?,
            ],
            confidence,
            convention_prefix: prefix.into(),
        }),
        Value::Object(map) => {
            if let (Some(x), Some(y), Some(width), Some(height)) = (
                map.get("x").and_then(number_value),
                map.get("y").and_then(number_value),
                map.get("width").and_then(number_value),
                map.get("height").and_then(number_value),
            ) {
                return Some(Candidate::Box {
                    values: [x, y, x + width, y + height],
                    confidence: confidence.or_else(|| number_field(map, &["confidence", "score"])),
                    convention_prefix: prefix.into(),
                });
            }
            Some(Candidate::Box {
                values: [
                    number_value(map.get("x1").or_else(|| map.get("left"))?)?,
                    number_value(map.get("y1").or_else(|| map.get("top"))?)?,
                    number_value(map.get("x2").or_else(|| map.get("right"))?)?,
                    number_value(map.get("y2").or_else(|| map.get("bottom"))?)?,
                ],
                confidence: confidence.or_else(|| number_field(map, &["confidence", "score"])),
                convention_prefix: prefix.into(),
            })
        }
        _ => None,
    }
}

fn number_field(map: &serde_json::Map<String, Value>, fields: &[&str]) -> Option<f64> {
    fields
        .iter()
        .find_map(|field| map.get(*field).and_then(number_value))
}

fn number_value(value: &Value) -> Option<f64> {
    match value {
        Value::Number(number) => number.as_f64(),
        Value::String(text) => text.trim().parse::<f64>().ok(),
        _ => None,
    }
    .filter(|value| value.is_finite())
}

fn normalize_candidate(
    candidate: Candidate,
    image_width: u32,
    image_height: u32,
) -> Result<GroundedPoint, String> {
    match candidate {
        Candidate::Point {
            x,
            y,
            confidence,
            convention_prefix,
        } => {
            let (x, y, convention) = normalize_pair(x, y, image_width, image_height)?;
            Ok(GroundedPoint {
                x,
                y,
                confidence,
                raw_output_convention: format!("{convention_prefix}:{convention}"),
            })
        }
        Candidate::Box {
            values,
            confidence,
            convention_prefix,
        } => {
            let (x1, y1, x2, y2, convention) = normalize_box(values, image_width, image_height)?;
            Ok(GroundedPoint {
                x: (((x1 + x2) / 2.0).round() as u32).min(image_width.saturating_sub(1)),
                y: (((y1 + y2) / 2.0).round() as u32).min(image_height.saturating_sub(1)),
                confidence,
                raw_output_convention: format!("{convention_prefix}:{convention}:box_center"),
            })
        }
    }
}

fn normalize_pair(
    x: f64,
    y: f64,
    image_width: u32,
    image_height: u32,
) -> Result<(u32, u32, &'static str), String> {
    if !x.is_finite() || !y.is_finite() {
        return Err("grounder point contains non-finite values".into());
    }
    if (0.0..=1.0).contains(&x) && (0.0..=1.0).contains(&y) {
        return Ok((
            scaled_unit(x, image_width),
            scaled_unit(y, image_height),
            "unit",
        ));
    }
    if x >= 0.0 && y >= 0.0 && x < image_width as f64 && y < image_height as f64 {
        return Ok((x.round() as u32, y.round() as u32, "absolute_pixels"));
    }
    if (0.0..=1000.0).contains(&x) && (0.0..=1000.0).contains(&y) {
        return Ok((
            scaled_token(x, image_width),
            scaled_token(y, image_height),
            "coordinate_tokens_1000",
        ));
    }
    Err(format!(
        "grounder point ({x}, {y}) is outside {image_width}x{image_height}"
    ))
}

fn normalize_box(
    values: [f64; 4],
    image_width: u32,
    image_height: u32,
) -> Result<(f64, f64, f64, f64, &'static str), String> {
    if values.iter().any(|value| !value.is_finite()) {
        return Err("grounder box contains non-finite values".into());
    }
    let all_unit = values.iter().all(|value| (0.0..=1.0).contains(value));
    let all_tokens = values.iter().all(|value| (0.0..=1000.0).contains(value));
    let absolute = values[0] >= 0.0
        && values[1] >= 0.0
        && values[2] >= 0.0
        && values[3] >= 0.0
        && values[0] < image_width as f64
        && values[2] <= image_width as f64
        && values[1] < image_height as f64
        && values[3] <= image_height as f64;

    let (x1, y1, x2, y2, convention) = if all_unit {
        (
            values[0] * image_width as f64,
            values[1] * image_height as f64,
            values[2] * image_width as f64,
            values[3] * image_height as f64,
            "unit",
        )
    } else if absolute {
        (
            values[0],
            values[1],
            values[2],
            values[3],
            "absolute_pixels",
        )
    } else if all_tokens {
        (
            values[0] / 1000.0 * image_width as f64,
            values[1] / 1000.0 * image_height as f64,
            values[2] / 1000.0 * image_width as f64,
            values[3] / 1000.0 * image_height as f64,
            "coordinate_tokens_1000",
        )
    } else {
        return Err(format!(
            "grounder box {:?} is outside {image_width}x{image_height}",
            values
        ));
    };

    Ok((
        x1.min(x2).clamp(0.0, image_width.saturating_sub(1) as f64),
        y1.min(y2).clamp(0.0, image_height.saturating_sub(1) as f64),
        x1.max(x2).clamp(0.0, image_width.saturating_sub(1) as f64),
        y1.max(y2).clamp(0.0, image_height.saturating_sub(1) as f64),
        convention,
    ))
}

fn scaled_unit(value: f64, extent: u32) -> u32 {
    (value * extent.saturating_sub(1) as f64)
        .round()
        .clamp(0.0, extent.saturating_sub(1) as f64) as u32
}

fn scaled_token(value: f64, extent: u32) -> u32 {
    ((value / 1000.0) * extent.saturating_sub(1) as f64)
        .round()
        .clamp(0.0, extent.saturating_sub(1) as f64) as u32
}

fn scale_coord(value: u32, from_extent: u32, to_extent: u32) -> u32 {
    if from_extent <= 1 || to_extent <= 1 {
        return 0;
    }
    ((value as f64 / from_extent.saturating_sub(1) as f64) * to_extent.saturating_sub(1) as f64)
        .round()
        .clamp(0.0, to_extent.saturating_sub(1) as f64) as u32
}

fn scan_numbers(text: &str) -> Vec<f64> {
    let mut values = Vec::new();
    let mut token = String::new();
    for ch in text.chars() {
        if ch.is_ascii_digit() || matches!(ch, '.' | '-' | '+' | 'e' | 'E') {
            token.push(ch);
        } else if !token.is_empty() {
            if let Ok(value) = token.parse::<f64>() {
                if value.is_finite() {
                    values.push(value);
                }
            }
            token.clear();
        }
    }
    if !token.is_empty() {
        if let Ok(value) = token.parse::<f64>() {
            if value.is_finite() {
                values.push(value);
            }
        }
    }
    values
}

fn chat_completion_content(raw_json: &str) -> Result<String, String> {
    #[derive(Deserialize)]
    struct Response {
        choices: Vec<Choice>,
    }
    #[derive(Deserialize)]
    struct Choice {
        message: Message,
    }
    #[derive(Deserialize)]
    struct Message {
        content: Value,
    }

    let response: Response =
        serde_json::from_str(raw_json).map_err(|err| format!("invalid grounder JSON: {err}"))?;
    let content = response
        .choices
        .into_iter()
        .next()
        .ok_or_else(|| "grounder response had no choices".to_string())?
        .message
        .content;
    match content {
        Value::String(text) => Ok(text),
        other => Ok(other.to_string()),
    }
}

fn decode_png_b64(image_png_b64: &str) -> Result<RgbaImage, String> {
    let bytes = STANDARD
        .decode(image_png_b64)
        .map_err(|err| format!("decode screenshot base64: {err}"))?;
    image::load_from_memory(&bytes)
        .map_err(|err| format!("decode screenshot png: {err}"))
        .map(|image| image.to_rgba8())
}

fn encode_png_b64(image: &RgbaImage) -> Result<String, String> {
    let mut bytes = Vec::new();
    DynamicImage::ImageRgba8(image.clone())
        .write_to(&mut Cursor::new(&mut bytes), ImageFormat::Png)
        .map_err(|err| format!("encode screenshot png: {err}"))?;
    Ok(STANDARD.encode(bytes))
}

fn resize_max_dim(image: &RgbaImage, max_dim: u32) -> Result<RgbaImage, String> {
    let width = image.width();
    let height = image.height();
    if width == 0 || height == 0 {
        return Err("cannot resize empty image".into());
    }
    let current_max = width.max(height);
    if current_max <= max_dim {
        return Ok(image.clone());
    }
    let scale = max_dim as f64 / current_max as f64;
    let new_width = ((width as f64 * scale).round() as u32).max(1);
    let new_height = ((height as f64 * scale).round() as u32).max(1);
    Ok(imageops::resize(
        image,
        new_width,
        new_height,
        imageops::FilterType::Triangle,
    ))
}

struct Crop {
    image: RgbaImage,
    x: u32,
    y: u32,
    width: u32,
    height: u32,
}

fn crop_around_point(image: &RgbaImage, point_x: u32, point_y: u32) -> Crop {
    let crop_width = image.width().min((image.width() / 3).max(MIN_CROP_SIDE));
    let crop_height = image.height().min((image.height() / 3).max(MIN_CROP_SIDE));
    let half_w = crop_width / 2;
    let half_h = crop_height / 2;
    let max_x = image.width().saturating_sub(crop_width);
    let max_y = image.height().saturating_sub(crop_height);
    let x = point_x.saturating_sub(half_w).min(max_x);
    let y = point_y.saturating_sub(half_h).min(max_y);
    Crop {
        image: imageops::crop_imm(image, x, y, crop_width, crop_height).to_image(),
        x,
        y,
        width: crop_width,
        height: crop_height,
    }
}

fn compact(value: &str) -> String {
    let normalized = value.split_whitespace().collect::<Vec<_>>().join(" ");
    if normalized.chars().count() <= 240 {
        normalized
    } else {
        let mut truncated = normalized.chars().take(237).collect::<String>();
        truncated.push_str("...");
        truncated
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parser_normalizes_unit_point() {
        let point =
            parse_grounding_output(r#"{"point":[0.5,0.25],"confidence":0.72}"#, 800, 400).unwrap();
        assert_eq!((point.x, point.y), (400, 100));
        assert_eq!(point.confidence, Some(0.72));
        assert!(point.raw_output_convention.contains("unit"));
    }

    #[test]
    fn parser_accepts_absolute_pixels() {
        let point = parse_grounding_output(r#"{"x":120,"y":80}"#, 800, 400).unwrap();
        assert_eq!((point.x, point.y), (120, 80));
        assert!(point.raw_output_convention.contains("absolute_pixels"));
    }

    #[test]
    fn parser_accepts_coordinate_tokens() {
        let point = parse_grounding_output(r#"{"point":[250,500]}"#, 200, 100).unwrap();
        assert_eq!((point.x, point.y), (50, 50));
        assert!(point
            .raw_output_convention
            .contains("coordinate_tokens_1000"));
    }

    #[test]
    fn parser_converts_box_to_center() {
        let point =
            parse_grounding_output(r#"{"box":[100,50,300,150],"confidence":0.6}"#, 800, 400)
                .unwrap();
        assert_eq!((point.x, point.y), (200, 100));
        assert_eq!(point.confidence, Some(0.6));
        assert!(point.raw_output_convention.contains("box_center"));
    }

    #[test]
    fn parser_rejects_out_of_bounds() {
        assert!(parse_grounding_output(r#"{"point":[1200,80]}"#, 800, 400).is_err());
    }

    #[test]
    fn crop_offset_mapping_scales_back_to_original_space() {
        assert_eq!(scale_coord(384, 768, 160), 80);
        let image = RgbaImage::new(900, 600);
        let crop = crop_around_point(&image, 600, 300);
        assert_eq!(crop.x, 450);
        assert_eq!(crop.y, 200);
        assert_eq!(crop.width, 300);
        assert_eq!(crop.height, 200);
    }
}
