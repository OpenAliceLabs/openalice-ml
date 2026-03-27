//! OpenAlice NER Service — zero-shot Named Entity Recognition via GLiNER.
//!
//! Uses gline-rs (Rust GLiNER) with multilingual ONNX models.
//! Zero-shot: pass ANY entity labels at runtime, no training needed.
//!
//! API:
//!   POST /v1/ner      — extract named entities
//!   GET  /v1/models   — list loaded models
//!   GET  /health      — health check

use std::sync::Arc;
use std::time::Instant;

use axum::{
    Json, Router,
    extract::State,
    http::StatusCode,
    routing::{get, post},
};
use gliner::model::{GLiNER, input::text::TextInput, params::Parameters};
use gliner::model::pipeline::span::SpanMode;
use orp::params::RuntimeParameters;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{info, warn};

const DEFAULT_PORT: u16 = 8104;

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

struct AppState {
    model: Option<GLiNER<SpanMode>>,
    model_name: String,
    started_at: Instant,
}

type SharedState = Arc<RwLock<AppState>>;

// ---------------------------------------------------------------------------
// Request/Response types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct NerRequest {
    text: String,
    /// Zero-shot entity labels. Default: ["person", "location", "organization", "date"]
    #[serde(default)]
    labels: Vec<String>,
}

#[derive(Serialize)]
struct NerResponse {
    entities: Vec<NerEntity>,
    model: String,
}

#[derive(Serialize)]
struct NerEntity {
    text: String,
    label: String,
    score: f32,
}

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    model: String,
    model_status: String,
    uptime_seconds: u64,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn health(State(state): State<SharedState>) -> Json<HealthResponse> {
    let s = state.read().await;
    Json(HealthResponse {
        status: "ok".into(),
        model: s.model_name.clone(),
        model_status: if s.model.is_some() { "loaded" } else { "not_loaded" }.into(),
        uptime_seconds: s.started_at.elapsed().as_secs(),
    })
}

async fn ner(
    State(state): State<SharedState>,
    Json(req): Json<NerRequest>,
) -> Result<Json<NerResponse>, (StatusCode, String)> {
    let s = state.read().await;
    let model = s.model.as_ref().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "GLiNER model not loaded. Download model files first.".into(),
    ))?;

    let default_labels = vec!["person", "location", "organization", "date", "event", "concept"];
    let labels: Vec<&str> = if req.labels.is_empty() {
        default_labels
    } else {
        req.labels.iter().map(|s| s.as_str()).collect()
    };

    let start = Instant::now();

    let input = TextInput::from_str(&[&req.text], &labels)
        .map_err(|e| (StatusCode::BAD_REQUEST, format!("input error: {e}")))?;

    let output = model
        .inference(input)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("inference error: {e}")))?;

    let elapsed = start.elapsed();

    let mut entities = Vec::new();
    if let Some(spans) = output.spans.first() {
        for span in spans {
            entities.push(NerEntity {
                text: span.text().to_string(),
                label: span.class().to_string(),
                score: span.probability() as f32,
            });
        }
    }

    info!(
        entities = entities.len(),
        labels = labels.len(),
        ms = elapsed.as_millis(),
        "ner: extracted"
    );

    Ok(Json(NerResponse {
        entities,
        model: s.model_name.clone(),
    }))
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

    info!("OpenAlice NER Service starting...");

    let models_dir = std::env::var("MODELS_DIR").unwrap_or_else(|_| "models".into());
    let tokenizer_path = format!("{}/tokenizer.json", models_dir);
    let model_path = format!("{}/model.onnx", models_dir);

    let model = if std::path::Path::new(&tokenizer_path).exists()
        && std::path::Path::new(&model_path).exists()
    {
        info!(tokenizer = %tokenizer_path, model = %model_path, "Loading GLiNER model...");
        match GLiNER::<SpanMode>::new(
            Parameters::default(),
            RuntimeParameters::default(),
            &tokenizer_path,
            &model_path,
        ) {
            Ok(m) => {
                info!("GLiNER model loaded successfully");
                Some(m)
            }
            Err(e) => {
                warn!(error = %e, "Failed to load GLiNER model");
                None
            }
        }
    } else {
        warn!(
            tokenizer = %tokenizer_path,
            model = %model_path,
            "Model files not found! Download from HuggingFace:\n\
             - tokenizer.json from gliner_multi-v2.1 or gliner_small-v2.1\n\
             - model.onnx (ONNX export)"
        );
        None
    };

    let model_name = if model.is_some() {
        "GLiNER-multi".to_string()
    } else {
        "none".to_string()
    };

    let state = Arc::new(RwLock::new(AppState {
        model,
        model_name,
        started_at: Instant::now(),
    }));

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/ner", post(ner))
        .with_state(state);

    let port = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(DEFAULT_PORT);

    let addr = format!("0.0.0.0:{}", port);
    info!(addr = %addr, "OpenAlice NER Service ready");

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
