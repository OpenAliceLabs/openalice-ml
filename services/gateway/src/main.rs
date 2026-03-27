//! OpenAlice ML Gateway — API routing, key validation, rate limiting, usage tracking.
//!
//! Routes requests to backend services:
//!   /v1/embed  → openalice-ml (8103)
//!   /v1/rerank → openalice-ml (8103)
//!   /v1/ner    → openalice-ner (8104)
//!
//! Features:
//!   - API key validation (Bearer token)
//!   - Per-key rate limiting (requests/day, tokens/month)
//!   - Usage tracking (persisted to disk)
//!   - Health aggregation from backend services

use std::collections::HashMap;
use std::sync::Arc;
use std::time::Instant;

use axum::{
    Json, Router,
    body::Body,
    extract::State,
    http::{HeaderMap, Request, StatusCode},
    middleware::{self, Next},
    response::{IntoResponse, Response},
    routing::{get, post},
};
use chrono::{Datelike, Utc};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{info, warn};

const DEFAULT_PORT: u16 = 8105;

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

#[derive(Clone)]
struct GatewayConfig {
    embed_url: String,   // http://openalice-ml:8103 or localhost
    ner_url: String,     // http://openalice-ner:8104 or localhost
    keys_file: String,   // persistence path for API keys
    usage_file: String,  // persistence path for usage data
}

impl Default for GatewayConfig {
    fn default() -> Self {
        Self {
            embed_url: std::env::var("EMBED_SERVICE_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:8103".into()),
            ner_url: std::env::var("NER_SERVICE_URL")
                .unwrap_or_else(|_| "http://127.0.0.1:8104".into()),
            keys_file: std::env::var("KEYS_FILE")
                .unwrap_or_else(|_| "data/keys.json".into()),
            usage_file: std::env::var("USAGE_FILE")
                .unwrap_or_else(|_| "data/usage.json".into()),
        }
    }
}

// ---------------------------------------------------------------------------
// API Key Management
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, Deserialize)]
struct ApiKey {
    /// The key hash (sha256 of the actual key).
    key_hash: String,
    /// Display name.
    name: String,
    /// Owner/org.
    owner: String,
    /// Created at.
    created_at: chrono::DateTime<Utc>,
    /// Rate limit: max requests per day (0 = unlimited).
    rate_limit_day: u64,
    /// Rate limit: max tokens per month (0 = unlimited).
    token_limit_month: u64,
    /// Whether the key is active.
    active: bool,
}

#[derive(Debug, Clone, Default, Serialize, Deserialize)]
struct UsageRecord {
    /// Requests today.
    requests_today: u64,
    /// Tokens this month.
    tokens_month: u64,
    /// Total requests all time.
    total_requests: u64,
    /// Total tokens all time.
    total_tokens: u64,
    /// Last request timestamp.
    last_request: Option<chrono::DateTime<Utc>>,
    /// Day of last reset (for daily counter).
    last_day_reset: u32,
    /// Month of last reset (for monthly counter).
    last_month_reset: u32,
}

// ---------------------------------------------------------------------------
// App State
// ---------------------------------------------------------------------------

struct AppState {
    config: GatewayConfig,
    client: reqwest::Client,
    /// API keys: key_hash → ApiKey
    keys: HashMap<String, ApiKey>,
    /// Usage per key_hash.
    usage: HashMap<String, UsageRecord>,
    started_at: Instant,
}

type SharedState = Arc<RwLock<AppState>>;

impl AppState {
    fn validate_key(&self, bearer: &str) -> Result<String, String> {
        let hash = sha256_hex(bearer);
        match self.keys.get(&hash) {
            Some(key) if key.active => Ok(hash),
            Some(_) => Err("API key is deactivated".into()),
            None => Err("Invalid API key".into()),
        }
    }

    fn check_rate_limit(&self, key_hash: &str) -> Result<(), String> {
        let key = self.keys.get(key_hash).ok_or("key not found")?;
        let usage = self.usage.get(key_hash);

        if let Some(u) = usage {
            if key.rate_limit_day > 0 && u.requests_today >= key.rate_limit_day {
                return Err(format!(
                    "Daily rate limit exceeded ({}/{})",
                    u.requests_today, key.rate_limit_day
                ));
            }
            if key.token_limit_month > 0 && u.tokens_month >= key.token_limit_month {
                return Err(format!(
                    "Monthly token limit exceeded ({}/{})",
                    u.tokens_month, key.token_limit_month
                ));
            }
        }
        Ok(())
    }

    fn record_usage(&mut self, key_hash: &str, tokens: u64) {
        let now = Utc::now();
        let usage = self.usage.entry(key_hash.to_string()).or_default();

        // Reset daily counter if new day.
        let today = now.day();
        if usage.last_day_reset != today {
            usage.requests_today = 0;
            usage.last_day_reset = today;
        }

        // Reset monthly counter if new month.
        let this_month = now.month();
        if usage.last_month_reset != this_month {
            usage.tokens_month = 0;
            usage.last_month_reset = this_month;
        }

        usage.requests_today += 1;
        usage.tokens_month += tokens;
        usage.total_requests += 1;
        usage.total_tokens += tokens;
        usage.last_request = Some(now);
    }

    fn save(&self) {
        // Save keys.
        if let Ok(data) = serde_json::to_string_pretty(&self.keys) {
            let _ = std::fs::write(&self.config.keys_file, data);
        }
        // Save usage.
        if let Ok(data) = serde_json::to_string_pretty(&self.usage) {
            let _ = std::fs::write(&self.config.usage_file, data);
        }
    }

    fn load(config: &GatewayConfig) -> (HashMap<String, ApiKey>, HashMap<String, UsageRecord>) {
        let keys: HashMap<String, ApiKey> = std::fs::read_to_string(&config.keys_file)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        let usage: HashMap<String, UsageRecord> = std::fs::read_to_string(&config.usage_file)
            .ok()
            .and_then(|s| serde_json::from_str(&s).ok())
            .unwrap_or_default();
        (keys, usage)
    }
}

fn sha256_hex(input: &str) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(input.as_bytes());
    format!("{:x}", hasher.finalize())
}

// ---------------------------------------------------------------------------
// Request/Response types
// ---------------------------------------------------------------------------

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    services: Vec<ServiceStatus>,
    uptime_seconds: u64,
}

#[derive(Serialize)]
struct ServiceStatus {
    name: String,
    url: String,
    status: String,
}

#[derive(Serialize)]
struct UsageResponse {
    requests_today: u64,
    tokens_month: u64,
    total_requests: u64,
    total_tokens: u64,
    rate_limit_day: u64,
    token_limit_month: u64,
}

#[derive(Deserialize)]
struct CreateKeyRequest {
    name: String,
    owner: String,
    #[serde(default = "default_rate_day")]
    rate_limit_day: u64,
    #[serde(default = "default_token_month")]
    token_limit_month: u64,
}

fn default_rate_day() -> u64 { 1000 }
fn default_token_month() -> u64 { 1_000_000 }

#[derive(Serialize)]
struct CreateKeyResponse {
    key: String,
    name: String,
    rate_limit_day: u64,
    token_limit_month: u64,
}

#[derive(Serialize)]
struct ErrorResponse {
    error: String,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn health(State(state): State<SharedState>) -> Json<HealthResponse> {
    let s = state.read().await;
    let client = &s.client;

    let mut services = Vec::new();

    // Check embed/rerank service.
    let embed_status = match client
        .get(format!("{}/health", s.config.embed_url))
        .send()
        .await
    {
        Ok(r) if r.status().is_success() => "healthy".into(),
        Ok(r) => format!("unhealthy ({})", r.status()),
        Err(e) => format!("unreachable: {}", e),
    };
    services.push(ServiceStatus {
        name: "embed-rerank".into(),
        url: s.config.embed_url.clone(),
        status: embed_status,
    });

    // Check NER service.
    let ner_status = match client
        .get(format!("{}/health", s.config.ner_url))
        .send()
        .await
    {
        Ok(r) if r.status().is_success() => "healthy".into(),
        Ok(r) => format!("unhealthy ({})", r.status()),
        Err(e) => format!("unreachable: {}", e),
    };
    services.push(ServiceStatus {
        name: "ner".into(),
        url: s.config.ner_url.clone(),
        status: ner_status,
    });

    Json(HealthResponse {
        status: "ok".into(),
        services,
        uptime_seconds: s.started_at.elapsed().as_secs(),
    })
}

async fn get_usage(
    State(state): State<SharedState>,
    headers: HeaderMap,
) -> Result<Json<UsageResponse>, (StatusCode, Json<ErrorResponse>)> {
    let bearer = extract_bearer(&headers)?;
    let s = state.read().await;
    let key_hash = s.validate_key(&bearer).map_err(|e| err(StatusCode::UNAUTHORIZED, e))?;

    let key = s.keys.get(&key_hash).unwrap();
    let usage = s.usage.get(&key_hash).cloned().unwrap_or_default();

    Ok(Json(UsageResponse {
        requests_today: usage.requests_today,
        tokens_month: usage.tokens_month,
        total_requests: usage.total_requests,
        total_tokens: usage.total_tokens,
        rate_limit_day: key.rate_limit_day,
        token_limit_month: key.token_limit_month,
    }))
}

async fn create_key(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Json(req): Json<CreateKeyRequest>,
) -> Result<Json<CreateKeyResponse>, (StatusCode, Json<ErrorResponse>)> {
    // Admin auth: check ADMIN_KEY env var.
    let bearer = extract_bearer(&headers)?;
    let admin_key = std::env::var("ADMIN_KEY").unwrap_or_default();
    if admin_key.is_empty() || bearer != admin_key {
        return Err(err(StatusCode::FORBIDDEN, "Admin key required".into()));
    }

    let raw_key = format!("oaml-{}", uuid::Uuid::new_v4().to_string().replace('-', ""));
    let key_hash = sha256_hex(&raw_key);

    let api_key = ApiKey {
        key_hash: key_hash.clone(),
        name: req.name.clone(),
        owner: req.owner,
        created_at: Utc::now(),
        rate_limit_day: req.rate_limit_day,
        token_limit_month: req.token_limit_month,
        active: true,
    };

    let mut s = state.write().await;
    s.keys.insert(key_hash, api_key);
    s.save();

    info!(name = %req.name, "API key created");

    Ok(Json(CreateKeyResponse {
        key: raw_key,
        name: req.name,
        rate_limit_day: req.rate_limit_day,
        token_limit_month: req.token_limit_month,
    }))
}

/// Proxy a request to a backend service.
async fn proxy(
    state: &AppState,
    key_hash: &str,
    backend_url: &str,
    path: &str,
    body: serde_json::Value,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let url = format!("{}{}", backend_url, path);

    let resp = state
        .client
        .post(&url)
        .json(&body)
        .send()
        .await
        .map_err(|e| err(StatusCode::BAD_GATEWAY, format!("Backend error: {}", e)))?;

    let status = resp.status();
    let resp_body: serde_json::Value = resp
        .json()
        .await
        .map_err(|e| err(StatusCode::BAD_GATEWAY, format!("Backend decode error: {}", e)))?;

    // Estimate tokens from the request (rough: 4 chars = 1 token).
    let tokens = body.to_string().len() as u64 / 4;

    // We'll record usage after returning (need mutable state).
    // Store key_hash + tokens in response extension for middleware.
    Ok((StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::OK), Json(resp_body)).into_response())
}

async fn proxy_embed(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let bearer = extract_bearer(&headers)?;
    let mut s = state.write().await;
    let key_hash = s.validate_key(&bearer).map_err(|e| err(StatusCode::UNAUTHORIZED, e))?;
    s.check_rate_limit(&key_hash).map_err(|e| err(StatusCode::TOO_MANY_REQUESTS, e))?;

    let tokens = body.to_string().len() as u64 / 4;
    let url = format!("{}/v1/embed", s.config.embed_url);

    let resp = s.client.post(&url).json(&body).send().await
        .map_err(|e| err(StatusCode::BAD_GATEWAY, format!("embed service: {}", e)))?;
    let status = resp.status();
    let resp_body: serde_json::Value = resp.json().await
        .map_err(|e| err(StatusCode::BAD_GATEWAY, format!("embed decode: {}", e)))?;

    s.record_usage(&key_hash, tokens);
    s.save();

    Ok((StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::OK), Json(resp_body)).into_response())
}

async fn proxy_rerank(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let bearer = extract_bearer(&headers)?;
    let mut s = state.write().await;
    let key_hash = s.validate_key(&bearer).map_err(|e| err(StatusCode::UNAUTHORIZED, e))?;
    s.check_rate_limit(&key_hash).map_err(|e| err(StatusCode::TOO_MANY_REQUESTS, e))?;

    let tokens = body.to_string().len() as u64 / 4;
    let url = format!("{}/v1/rerank", s.config.embed_url);

    let resp = s.client.post(&url).json(&body).send().await
        .map_err(|e| err(StatusCode::BAD_GATEWAY, format!("rerank service: {}", e)))?;
    let status = resp.status();
    let resp_body: serde_json::Value = resp.json().await
        .map_err(|e| err(StatusCode::BAD_GATEWAY, format!("rerank decode: {}", e)))?;

    s.record_usage(&key_hash, tokens);
    s.save();

    Ok((StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::OK), Json(resp_body)).into_response())
}

async fn proxy_ner(
    State(state): State<SharedState>,
    headers: HeaderMap,
    Json(body): Json<serde_json::Value>,
) -> Result<Response, (StatusCode, Json<ErrorResponse>)> {
    let bearer = extract_bearer(&headers)?;
    let mut s = state.write().await;
    let key_hash = s.validate_key(&bearer).map_err(|e| err(StatusCode::UNAUTHORIZED, e))?;
    s.check_rate_limit(&key_hash).map_err(|e| err(StatusCode::TOO_MANY_REQUESTS, e))?;

    let tokens = body.to_string().len() as u64 / 4;
    let url = format!("{}/v1/ner", s.config.ner_url);

    let resp = s.client.post(&url).json(&body).send().await
        .map_err(|e| err(StatusCode::BAD_GATEWAY, format!("ner service: {}", e)))?;
    let status = resp.status();
    let resp_body: serde_json::Value = resp.json().await
        .map_err(|e| err(StatusCode::BAD_GATEWAY, format!("ner decode: {}", e)))?;

    s.record_usage(&key_hash, tokens);
    s.save();

    Ok((StatusCode::from_u16(status.as_u16()).unwrap_or(StatusCode::OK), Json(resp_body)).into_response())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn extract_bearer(headers: &HeaderMap) -> Result<String, (StatusCode, Json<ErrorResponse>)> {
    headers
        .get("Authorization")
        .and_then(|v| v.to_str().ok())
        .and_then(|v| v.strip_prefix("Bearer "))
        .map(|s| s.to_string())
        .ok_or_else(|| err(StatusCode::UNAUTHORIZED, "Missing Authorization: Bearer <key>".into()))
}

fn err(status: StatusCode, message: String) -> (StatusCode, Json<ErrorResponse>) {
    (status, Json(ErrorResponse { error: message }))
}

// ---------------------------------------------------------------------------
// Main
// ---------------------------------------------------------------------------

#[tokio::main]
async fn main() {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info".into()),
        )
        .json()
        .init();

    info!("OpenAlice ML Gateway starting...");

    let config = GatewayConfig::default();

    // Ensure data directory exists.
    let data_dir = std::path::Path::new(&config.keys_file).parent().unwrap_or(std::path::Path::new("."));
    let _ = std::fs::create_dir_all(data_dir);

    let (keys, usage) = AppState::load(&config);
    info!(keys = keys.len(), "Loaded API keys");

    let state = Arc::new(RwLock::new(AppState {
        config,
        client: reqwest::Client::new(),
        keys,
        usage,
        started_at: Instant::now(),
    }));

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/embed", post(proxy_embed))
        .route("/v1/rerank", post(proxy_rerank))
        .route("/v1/ner", post(proxy_ner))
        .route("/v1/usage", get(get_usage))
        .route("/admin/keys", post(create_key))
        .with_state(state);

    let port = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(DEFAULT_PORT);

    let addr = format!("0.0.0.0:{}", port);
    info!(addr = %addr, "OpenAlice ML Gateway ready");

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
