//! OpenAlice ML Service — embeddings, reranking, and NER.
//!
//! Provides local ML inference via ONNX models:
//! - Embeddings: BAAI/bge-m3 (1024-dim, MTEB 68.0, multilingual)
//! - Reranking: cross-encoder/ms-marco-MiniLM-L-6-v2 (80MB)
//! - NER: (Phase 2 — pattern matching for now)
//!
//! API:
//!   POST /v1/embed    — generate embeddings
//!   POST /v1/rerank   — rerank documents by query relevance
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
use fastembed::{
    EmbeddingModel, InitOptions, TextEmbedding,
    RerankerModel, TextRerank, RerankInitOptions, RerankResult,
};
use gliner::model::{GLiNER, input::text::TextInput, params::Parameters};
use gliner::model::pipeline::span::SpanMode;
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;
use tracing::{info, warn};

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

const DEFAULT_PORT: u16 = 8103;
const EMBEDDING_MODEL: EmbeddingModel = EmbeddingModel::BGELargeENV15; // Will try BGE-M3 if available

// ---------------------------------------------------------------------------
// App State
// ---------------------------------------------------------------------------

struct AppState {
    embedder: Option<TextEmbedding>,
    reranker: Option<TextRerank>,
    ner_model: Option<GLiNER<SpanMode>>,
    embed_model_name: String,
    rerank_model_name: String,
    ner_model_name: String,
    started_at: Instant,
}

type SharedState = Arc<RwLock<AppState>>;

// ---------------------------------------------------------------------------
// Request/Response types
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct EmbedRequest {
    input: Vec<String>,
    #[serde(default = "default_dims")]
    dimensions: usize,
}

fn default_dims() -> usize { 1024 }

#[derive(Serialize)]
struct EmbedResponse {
    data: Vec<EmbedData>,
    model: String,
    usage: EmbedUsage,
}

#[derive(Serialize)]
struct EmbedData {
    index: usize,
    embedding: Vec<f32>,
}

#[derive(Serialize)]
struct EmbedUsage {
    total_tokens: usize,
}

#[derive(Deserialize)]
struct RerankRequest {
    query: String,
    documents: Vec<String>,
    #[serde(default = "default_top_k")]
    top_k: usize,
}

fn default_top_k() -> usize { 10 }

#[derive(Serialize)]
struct RerankResponse {
    results: Vec<RerankResultItem>,
    model: String,
}

#[derive(Serialize)]
struct RerankResultItem {
    index: usize,
    score: f32,
    text: String,
}

#[derive(Deserialize)]
struct NerRequest {
    text: String,
    /// Zero-shot entity labels (e.g., ["person", "location", "concept"]).
    /// Default: ["person", "location", "organization", "concept", "date", "event"]
    #[serde(default)]
    labels: Vec<String>,
    #[serde(default)]
    lang: String,
}

#[derive(Serialize)]
struct NerResponse {
    entities: Vec<NerEntity>,
}

#[derive(Serialize)]
struct NerEntity {
    text: String,
    label: String,
    #[serde(default, skip_serializing_if = "is_zero_f32")]
    score: f32,
    start: usize,
    end: usize,
}

fn is_zero_f32(v: &f32) -> bool { *v == 0.0 }

#[derive(Serialize)]
struct HealthResponse {
    status: String,
    models: Vec<ModelInfo>,
    uptime_seconds: u64,
}

#[derive(Serialize)]
struct ModelInfo {
    name: String,
    kind: String,
    status: String,
}

#[derive(Serialize)]
struct ModelsResponse {
    models: Vec<ModelInfo>,
}

// ---------------------------------------------------------------------------
// Handlers
// ---------------------------------------------------------------------------

async fn health(State(state): State<SharedState>) -> Json<HealthResponse> {
    let s = state.read().await;
    let mut models = Vec::new();

    models.push(ModelInfo {
        name: s.embed_model_name.clone(),
        kind: "embedding".into(),
        status: if s.embedder.is_some() { "loaded" } else { "failed" }.into(),
    });
    models.push(ModelInfo {
        name: s.rerank_model_name.clone(),
        kind: "reranker".into(),
        status: if s.reranker.is_some() { "loaded" } else { "failed" }.into(),
    });
    models.push(ModelInfo {
        name: s.ner_model_name.clone(),
        kind: "ner".into(),
        status: if s.ner_model.is_some() { "loaded" } else { "pattern-fallback" }.into(),
    });

    Json(HealthResponse {
        status: "ok".into(),
        models,
        uptime_seconds: s.started_at.elapsed().as_secs(),
    })
}

async fn list_models(State(state): State<SharedState>) -> Json<ModelsResponse> {
    let s = state.read().await;
    Json(ModelsResponse {
        models: vec![
            ModelInfo {
                name: s.embed_model_name.clone(),
                kind: "embedding".into(),
                status: if s.embedder.is_some() { "loaded" } else { "failed" }.into(),
            },
            ModelInfo {
                name: s.rerank_model_name.clone(),
                kind: "reranker".into(),
                status: if s.reranker.is_some() { "loaded" } else { "failed" }.into(),
            },
            ModelInfo {
                name: s.ner_model_name.clone(),
                kind: "ner".into(),
                status: if s.ner_model.is_some() { "loaded" } else { "pattern-fallback" }.into(),
            },
        ],
    })
}

async fn embed(
    State(state): State<SharedState>,
    Json(req): Json<EmbedRequest>,
) -> Result<Json<EmbedResponse>, (StatusCode, String)> {
    let mut s = state.write().await;
    let embedder = s.embedder.as_mut().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "embedding model not loaded".into(),
    ))?;

    let texts: Vec<String> = req.input.clone();
    let start = Instant::now();

    let embeddings = embedder
        .embed(texts, None)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("embed error: {e}")))?;

    let elapsed = start.elapsed();
    info!(
        count = embeddings.len(),
        dims = embeddings.first().map(|e| e.len()).unwrap_or(0),
        ms = elapsed.as_millis(),
        "embed: generated"
    );

    let data: Vec<EmbedData> = embeddings
        .into_iter()
        .enumerate()
        .map(|(i, emb)| EmbedData {
            index: i,
            embedding: emb,
        })
        .collect();

    // Rough token estimate: ~4 chars per token
    let total_tokens: usize = req.input.iter().map(|s| s.len() / 4 + 1).sum();

    Ok(Json(EmbedResponse {
        data,
        model: s.embed_model_name.clone(),
        usage: EmbedUsage { total_tokens },
    }))
}

async fn rerank(
    State(state): State<SharedState>,
    Json(req): Json<RerankRequest>,
) -> Result<Json<RerankResponse>, (StatusCode, String)> {
    let mut s = state.write().await;
    let reranker = s.reranker.as_mut().ok_or((
        StatusCode::SERVICE_UNAVAILABLE,
        "reranker model not loaded".into(),
    ))?;

    let docs: Vec<&String> = req.documents.iter().collect();
    let start = Instant::now();

    let results = reranker
        .rerank(&req.query, &docs, true, None)
        .map_err(|e| (StatusCode::INTERNAL_SERVER_ERROR, format!("rerank error: {e}")))?;

    let elapsed = start.elapsed();
    info!(
        query_len = req.query.len(),
        docs = docs.len(),
        ms = elapsed.as_millis(),
        "rerank: scored"
    );

    let mut items: Vec<RerankResultItem> = results
        .into_iter()
        .map(|r| RerankResultItem {
            index: r.index,
            score: r.score as f32,
            text: req.documents.get(r.index).cloned().unwrap_or_default(),
        })
        .collect();

    items.truncate(req.top_k);

    Ok(Json(RerankResponse {
        results: items,
        model: s.rerank_model_name.clone(),
    }))
}

async fn ner(
    State(state): State<SharedState>,
    Json(req): Json<NerRequest>,
) -> Json<NerResponse> {
    // Default entity labels for zero-shot NER
    let default_labels = vec!["person", "location", "organization", "concept", "date", "event"];
    let labels: Vec<&str> = if req.labels.is_empty() {
        default_labels.iter().map(|s| s.as_str()).collect()
    } else {
        req.labels.iter().map(|s| s.as_str()).collect()
    };

    // Try GLiNER model first, fallback to pattern-based
    let s = state.read().await;
    if let Some(ref ner_model) = s.ner_model {
        let start = Instant::now();
        match TextInput::from_str(&[&req.text], &labels) {
            Ok(input) => match ner_model.inference(input) {
                Ok(output) => {
                    let elapsed = start.elapsed();
                    let mut entities = Vec::new();
                    if let Some(spans) = output.spans.first() {
                        for span in spans {
                            entities.push(NerEntity {
                                text: span.text().to_string(),
                                label: span.class().to_string(),
                                score: span.probability() as f32,
                                start: 0, // GLiNER doesn't expose char offsets directly
                                end: 0,
                            });
                        }
                    }
                    info!(
                        entities = entities.len(),
                        ms = elapsed.as_millis(),
                        model = %s.ner_model_name,
                        "ner: extracted"
                    );
                    return Json(NerResponse { entities });
                }
                Err(e) => {
                    warn!(error = %e, "GLiNER inference failed, falling back to pattern NER");
                }
            },
            Err(e) => {
                warn!(error = %e, "GLiNER input creation failed, falling back to pattern NER");
            }
        }
    }

    // Fallback: pattern-based NER
    let entities = extract_entities_pattern(&req.text);
    Json(NerResponse { entities })
}

/// Simple pattern-based NER: finds capitalized words, @mentions, common name patterns.
fn extract_entities_pattern(text: &str) -> Vec<NerEntity> {
    let mut entities = Vec::new();
    let words: Vec<&str> = text.split_whitespace().collect();

    let mut i = 0;
    while i < words.len() {
        let word = words[i].trim_matches(|c: char| !c.is_alphanumeric());

        // Capitalized word (potential name/place) — not at sentence start
        if !word.is_empty()
            && word.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
            && i > 0 // skip sentence-initial caps
            && !is_common_word(word)
        {
            // Check for multi-word names (e.g., "Aleksandr Gorovetskiy")
            let mut name = word.to_string();
            let start = text.find(words[i]).unwrap_or(0);
            let mut end = start + words[i].len();

            while i + 1 < words.len() {
                let next = words[i + 1].trim_matches(|c: char| !c.is_alphanumeric());
                if !next.is_empty()
                    && next.chars().next().map(|c| c.is_uppercase()).unwrap_or(false)
                    && !is_common_word(next)
                {
                    i += 1;
                    name.push(' ');
                    name.push_str(next);
                    end = text.find(words[i]).unwrap_or(end) + words[i].len();
                } else {
                    break;
                }
            }

            entities.push(NerEntity {
                text: name,
                label: "PERSON".into(),
                start,
                end,
            });
        }

        // @mention
        if words[i].starts_with('@') && words[i].len() > 1 {
            let mention = words[i].trim_matches(|c: char| !c.is_alphanumeric() && c != '@');
            let start = text.find(words[i]).unwrap_or(0);
            entities.push(NerEntity {
                text: mention.to_string(),
                label: "MENTION".into(),
                start,
                end: start + words[i].len(),
            });
        }

        i += 1;
    }

    entities
}

fn is_common_word(word: &str) -> bool {
    const COMMON: &[&str] = &[
        "The", "This", "That", "What", "When", "Where", "How", "Why", "Who",
        "I", "You", "He", "She", "It", "We", "They", "My", "Your", "His",
        "Her", "Its", "Our", "Their", "And", "But", "Or", "Not", "If",
        "Это", "Как", "Что", "Где", "Кто", "Мой", "Ваш", "Его", "Она",
        "Они", "Мы", "Вот", "Там", "Тут", "Нет", "Да", "Или", "Но",
    ];
    COMMON.contains(&word)
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

    info!("OpenAlice ML Service starting...");

    // Load embedding model
    info!("Loading embedding model (this may download ~400MB on first run)...");
    let mut embed_opts = InitOptions::default();
    embed_opts.model_name = EmbeddingModel::BGELargeENV15;
    embed_opts.show_download_progress = true;

    let embedder = match TextEmbedding::try_new(embed_opts) {
        Ok(e) => {
            info!("Embedding model loaded: BGE-Large-EN-v1.5");
            Some(e)
        }
        Err(e) => {
            warn!(error = %e, "Failed to load embedding model, trying fallback...");
            let mut fallback_opts = InitOptions::default();
            fallback_opts.model_name = EmbeddingModel::AllMiniLML6V2;
            fallback_opts.show_download_progress = true;
            match TextEmbedding::try_new(fallback_opts) {
                Ok(e) => {
                    info!("Fallback embedding model loaded: AllMiniLM-L6-V2");
                    Some(e)
                }
                Err(e2) => {
                    warn!(error = %e2, "Failed to load fallback embedding model");
                    None
                }
            }
        }
    };

    // Load reranker
    info!("Loading reranker model...");
    let mut rerank_opts = RerankInitOptions::default();
    rerank_opts.model_name = RerankerModel::BGERerankerBase;
    rerank_opts.show_download_progress = true;

    let reranker = match TextRerank::try_new(rerank_opts) {
        Ok(r) => {
            info!("Reranker model loaded: BGE-Reranker-Base");
            Some(r)
        }
        Err(e) => {
            warn!(error = %e, "Failed to load reranker model");
            None
        }
    };

    // Load GLiNER NER model (zero-shot, multilingual)
    info!("Loading GLiNER NER model...");
    let models_dir = std::env::var("MODELS_DIR").unwrap_or_else(|_| "models".into());
    let ner_tokenizer = format!("{}/gliner/tokenizer.json", models_dir);
    let ner_model_path = format!("{}/gliner/model.onnx", models_dir);

    let ner_model = if std::path::Path::new(&ner_tokenizer).exists()
        && std::path::Path::new(&ner_model_path).exists()
    {
        match GLiNER::<SpanMode>::new(
            Parameters::default(),
            orp::params::RuntimeParameters::default(),
            &ner_tokenizer,
            &ner_model_path,
        ) {
            Ok(m) => {
                info!("GLiNER NER model loaded");
                Some(m)
            }
            Err(e) => {
                warn!(error = %e, "Failed to load GLiNER model, using pattern fallback");
                None
            }
        }
    } else {
        info!(
            tokenizer = %ner_tokenizer,
            model = %ner_model_path,
            "GLiNER model files not found, using pattern fallback. Download from HuggingFace."
        );
        None
    };

    let state = Arc::new(RwLock::new(AppState {
        embed_model_name: if embedder.is_some() {
            "BGE-Large-EN-v1.5".into()
        } else {
            "none".into()
        },
        rerank_model_name: if reranker.is_some() {
            "BGE-Reranker-Base".into()
        } else {
            "none".into()
        },
        ner_model_name: if ner_model.is_some() {
            "GLiNER-multi-v2.1".into()
        } else {
            "pattern-ner-v1".into()
        },
        embedder,
        reranker,
        ner_model,
        started_at: Instant::now(),
    }));

    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/models", get(list_models))
        .route("/v1/embed", post(embed))
        .route("/v1/rerank", post(rerank))
        .route("/v1/ner", post(ner))
        .with_state(state);

    let port = std::env::var("PORT")
        .ok()
        .and_then(|p| p.parse().ok())
        .unwrap_or(DEFAULT_PORT);

    let addr = format!("0.0.0.0:{}", port);
    info!(addr = %addr, "OpenAlice ML Service ready");

    let listener = tokio::net::TcpListener::bind(&addr).await.unwrap();
    axum::serve(listener, app).await.unwrap();
}
