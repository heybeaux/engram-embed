//! engram-embed: Local embedding server
//!
//! A drop-in replacement for OpenAI's embeddings API, running locally with Candle.
//! Supports multiple models for ensemble retrieval.

use anyhow::Result;
use axum::{
    extract::State,
    http::StatusCode,
    routing::{get, post},
    Json, Router,
};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::sync::Arc;
use tracing::{info, Level};
use tracing_subscriber::FmtSubscriber;

mod embedder;

use embedder::{EmbedResult, ModelId, ModelRegistry};

// ============================================================================
// Types (OpenAI-compatible + Extensions)
// ============================================================================

#[derive(Debug, Deserialize)]
struct EmbeddingRequest {
    /// Text to embed (string or array of strings)
    input: StringOrVec,
    /// Model name: "bge-base", "minilm", or "*" for all models
    #[serde(default = "default_model")]
    model: String,
}

#[derive(Debug, Deserialize)]
#[serde(untagged)]
enum StringOrVec {
    Single(String),
    Multiple(Vec<String>),
}

fn default_model() -> String {
    "bge-base".to_string()
}

/// OpenAI-compatible response
#[derive(Debug, Serialize)]
struct EmbeddingResponse {
    object: String,
    data: Vec<EmbeddingData>,
    model: String,
    usage: Usage,
}

#[derive(Debug, Serialize)]
struct EmbeddingData {
    object: String,
    embedding: Vec<f32>,
    index: usize,
}

#[derive(Debug, Serialize)]
struct Usage {
    prompt_tokens: usize,
    total_tokens: usize,
}

/// Extended response for multi-model embeddings
#[derive(Debug, Serialize)]
struct MultiModelResponse {
    object: String,
    embeddings: Vec<ModelEmbeddings>,
    timing: Timing,
}

#[derive(Debug, Serialize)]
struct ModelEmbeddings {
    model: String,
    dimensions: usize,
    data: Vec<EmbeddingData>,
}

#[derive(Debug, Serialize)]
struct Timing {
    total_ms: u64,
    per_model: HashMap<String, u64>,
}

#[derive(Debug, Serialize)]
struct HealthResponse {
    status: String,
    models: Vec<ModelStatus>,
    version: String,
}

#[derive(Debug, Serialize)]
struct ModelStatus {
    id: String,
    dimensions: usize,
    default: bool,
}

// ============================================================================
// App State
// ============================================================================

struct AppState {
    registry: ModelRegistry,
}

// ============================================================================
// Handlers
// ============================================================================

async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let models = state.registry.loaded_models();
    let default_model = models.first().copied().unwrap_or(ModelId::BgeBase);

    Json(HealthResponse {
        status: "ok".to_string(),
        models: models
            .into_iter()
            .map(|m| ModelStatus {
                id: m.display_name().to_string(),
                dimensions: m.dimensions(),
                default: m == default_model,
            })
            .collect(),
        version: env!("CARGO_PKG_VERSION").to_string(),
    })
}

/// Standard embedding endpoint (OpenAI-compatible)
/// Use model="*" for multi-model response
async fn embed(
    State(state): State<Arc<AppState>>,
    Json(request): Json<EmbeddingRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    let texts: Vec<String> = match request.input {
        StringOrVec::Single(s) => vec![s],
        StringOrVec::Multiple(v) => v,
    };

    // Count tokens (rough approximation)
    let token_count: usize = texts.iter().map(|t| t.split_whitespace().count()).sum();

    // Check if requesting all models
    if request.model == "*" || request.model == "all" {
        let start = std::time::Instant::now();
        let results = state.registry.embed_all(&texts);
        let total_ms = start.elapsed().as_millis() as u64;

        let response = MultiModelResponse {
            object: "list".to_string(),
            embeddings: results
                .iter()
                .map(|r| ModelEmbeddings {
                    model: r.model.display_name().to_string(),
                    dimensions: r.dimensions,
                    data: r
                        .vectors
                        .iter()
                        .enumerate()
                        .map(|(i, v)| EmbeddingData {
                            object: "embedding".to_string(),
                            embedding: v.clone(),
                            index: i,
                        })
                        .collect(),
                })
                .collect(),
            timing: Timing {
                total_ms,
                per_model: results
                    .iter()
                    .map(|r| (r.model.display_name().to_string(), r.latency_ms))
                    .collect(),
            },
        };

        return Ok(Json(serde_json::to_value(response).unwrap()));
    }

    // Single model request (OpenAI-compatible)
    let result = state
        .registry
        .embed(&texts, Some(&request.model))
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;

    let data: Vec<EmbeddingData> = result
        .vectors
        .into_iter()
        .enumerate()
        .map(|(i, embedding)| EmbeddingData {
            object: "embedding".to_string(),
            embedding,
            index: i,
        })
        .collect();

    let response = EmbeddingResponse {
        object: "list".to_string(),
        data,
        model: result.model.display_name().to_string(),
        usage: Usage {
            prompt_tokens: token_count,
            total_tokens: token_count,
        },
    };

    Ok(Json(serde_json::to_value(response).unwrap()))
}

/// List available models
async fn list_models(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let models: Vec<serde_json::Value> = state
        .registry
        .loaded_models()
        .into_iter()
        .map(|m| {
            serde_json::json!({
                "id": m.display_name(),
                "object": "model",
                "created": 0,
                "owned_by": "engram-embed",
                "dimensions": m.dimensions(),
            })
        })
        .collect();

    Json(serde_json::json!({
        "object": "list",
        "data": models
    }))
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize tracing
    let subscriber = FmtSubscriber::builder()
        .with_max_level(Level::INFO)
        .finish();
    tracing::subscriber::set_global_default(subscriber)?;

    info!("🚀 engram-embed v{} starting...", env!("CARGO_PKG_VERSION"));

    // Determine which models to load from env
    let models_env = std::env::var("EMBED_MODELS").unwrap_or_else(|_| "bge-base".to_string());
    let model_ids: Vec<ModelId> = models_env
        .split(',')
        .filter_map(|s| ModelId::from_str(s.trim()))
        .collect();

    let model_ids = if model_ids.is_empty() {
        vec![ModelId::BgeBase]
    } else {
        model_ids
    };

    info!(
        "📦 Loading {} model(s): {:?}",
        model_ids.len(),
        model_ids.iter().map(|m| m.display_name()).collect::<Vec<_>>()
    );

    // Load models
    let registry = ModelRegistry::new(&model_ids)?;
    info!("✅ All models loaded successfully");

    // Create app state
    let state = Arc::new(AppState { registry });

    // Build router
    let app = Router::new()
        .route("/health", get(health))
        .route("/v1/embeddings", post(embed))
        .route("/v1/models", get(list_models))
        .with_state(state);

    // Start server
    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    let addr = format!("127.0.0.1:{}", port);
    info!("🌐 Listening on http://{}", addr);
    info!("📡 POST /v1/embeddings - OpenAI-compatible endpoint");
    info!("📡 GET  /v1/models     - List available models");
    info!("");
    info!("Usage:");
    info!("  Single model:  {{ \"input\": \"text\", \"model\": \"bge-base\" }}");
    info!("  All models:    {{ \"input\": \"text\", \"model\": \"*\" }}");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    axum::serve(listener, app).await?;

    Ok(())
}
