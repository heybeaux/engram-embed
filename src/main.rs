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
use std::time::Instant;
use tokio::signal;
use tokio::sync::watch;
use tracing::{info, warn};
use tracing_subscriber::{fmt, layer::SubscriberExt, util::SubscriberInitExt, EnvFilter};

mod embedder;
mod metal_bert;
mod metal_compat;
mod metrics;
mod nomic_bert;

use embedder::{ModelId, ModelRegistry};
use metrics::Metrics;

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
    loaded_count: usize,
    version: String,
    uptime_secs: u64,
    memory_bytes: u64,
    memory_mb: f64,
    last_embedding_time: Option<u64>,
    in_flight_requests: usize,
}

#[derive(Debug, Serialize)]
struct ModelStatus {
    id: String,
    dimensions: usize,
    max_tokens: usize,
    loaded: bool,
    default: bool,
}

// ============================================================================
// App State
// ============================================================================

struct AppState {
    registry: ModelRegistry,
    metrics: Metrics,
    #[allow(dead_code)]
    shutdown_rx: watch::Receiver<bool>,
}

// ============================================================================
// Handlers
// ============================================================================

async fn health(State(state): State<Arc<AppState>>) -> Json<HealthResponse> {
    let enabled = state.registry.enabled_models();
    let loaded = state.registry.loaded_models();
    let default_model = enabled.first().copied().unwrap_or(ModelId::BgeBase);
    
    let memory_bytes = state.metrics.memory_bytes();
    let last_time = state.metrics.last_embedding_time();

    Json(HealthResponse {
        status: "ok".to_string(),
        models: enabled
            .iter()
            .map(|&m| ModelStatus {
                id: m.display_name().to_string(),
                dimensions: m.dimensions(),
                max_tokens: m.max_tokens(),
                loaded: loaded.contains(&m),
                default: m == default_model,
            })
            .collect(),
        loaded_count: loaded.len(),
        version: env!("CARGO_PKG_VERSION").to_string(),
        uptime_secs: state.metrics.uptime_secs(),
        memory_bytes,
        memory_mb: memory_bytes as f64 / 1024.0 / 1024.0,
        last_embedding_time: if last_time > 0 { Some(last_time) } else { None },
        in_flight_requests: state.metrics.in_flight_count(),
    })
}

/// Metrics endpoint - returns in-memory counters
async fn metrics_handler(State(state): State<Arc<AppState>>) -> Json<serde_json::Value> {
    let snapshot = state.metrics.snapshot();
    Json(serde_json::to_value(snapshot).unwrap())
}

/// Standard embedding endpoint (OpenAI-compatible)
/// Use model="*" for multi-model response
async fn embed(
    State(state): State<Arc<AppState>>,
    Json(request): Json<EmbeddingRequest>,
) -> Result<Json<serde_json::Value>, (StatusCode, String)> {
    // Track in-flight request
    let _guard = state.metrics.request_start();

    let texts: Vec<String> = match request.input {
        StringOrVec::Single(s) => vec![s],
        StringOrVec::Multiple(v) => v,
    };

    let text_count = texts.len();
    let total_chars: usize = texts.iter().map(|t| t.len()).sum();

    // Count tokens (rough approximation)
    let token_count: usize = texts.iter().map(|t| t.split_whitespace().count()).sum();

    // Check if requesting all models
    if request.model == "*" || request.model == "all" {
        let start = Instant::now();
        let results = state.registry.embed_all(&texts);
        let total_ms = start.elapsed().as_millis() as u64;

        // Record metrics for each model
        for r in &results {
            let latency_us = r.latency_ms * 1000;
            state.metrics.record_embedding(r.model.display_name(), text_count, latency_us);
            
            info!(
                model = r.model.display_name(),
                text_count = text_count,
                total_chars = total_chars,
                latency_ms = r.latency_ms,
                "Embedding completed (multi-model)"
            );
        }

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
    let start = Instant::now();
    let result = state
        .registry
        .embed(&texts, Some(&request.model))
        .map_err(|e| (StatusCode::BAD_REQUEST, e.to_string()))?;
    
    let latency_us = start.elapsed().as_micros() as u64;
    let latency_ms = latency_us / 1000;

    // Record metrics
    state.metrics.record_embedding(result.model.display_name(), text_count, latency_us);

    // Structured logging
    info!(
        model = result.model.display_name(),
        text_count = text_count,
        total_chars = total_chars,
        latency_ms = latency_ms,
        "Embedding completed"
    );

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
    let loaded = state.registry.loaded_models();
    let models: Vec<serde_json::Value> = state
        .registry
        .enabled_models()
        .into_iter()
        .map(|m| {
            serde_json::json!({
                "id": m.display_name(),
                "object": "model",
                "created": 0,
                "owned_by": "engram-embed",
                "dimensions": m.dimensions(),
                "max_tokens": m.max_tokens(),
                "loaded": loaded.contains(&m),
            })
        })
        .collect();

    Json(serde_json::json!({
        "object": "list",
        "data": models
    }))
}

// ============================================================================
// Graceful Shutdown
// ============================================================================

/// Wait for shutdown signal (SIGTERM or SIGINT)
async fn shutdown_signal() {
    let ctrl_c = async {
        signal::ctrl_c()
            .await
            .expect("Failed to install Ctrl+C handler");
    };

    #[cfg(unix)]
    let terminate = async {
        signal::unix::signal(signal::unix::SignalKind::terminate())
            .expect("Failed to install SIGTERM handler")
            .recv()
            .await;
    };

    #[cfg(not(unix))]
    let terminate = std::future::pending::<()>();

    tokio::select! {
        _ = ctrl_c => {},
        _ = terminate => {},
    }
}

// ============================================================================
// Main
// ============================================================================

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize structured logging
    // Use JSON format if EMBED_LOG_FORMAT=json, otherwise use pretty format
    let log_format = std::env::var("EMBED_LOG_FORMAT").unwrap_or_default();
    let env_filter = EnvFilter::try_from_default_env()
        .unwrap_or_else(|_| EnvFilter::new("info"));

    if log_format == "json" {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt::layer().json())
            .init();
    } else {
        tracing_subscriber::registry()
            .with(env_filter)
            .with(fmt::layer())
            .init();
    }

    info!(version = env!("CARGO_PKG_VERSION"), "engram-embed starting");

    // Determine which models to enable from env
    // Use "*" or "all" for all models, or comma-separated list
    let models_env = std::env::var("EMBED_MODELS").unwrap_or_else(|_| "bge-base".to_string());
    
    let model_ids: Vec<ModelId> = if models_env == "*" || models_env.to_lowercase() == "all" {
        ModelId::all().to_vec()
    } else {
        models_env
            .split(',')
            .filter_map(|s| ModelId::from_str(s.trim()))
            .collect()
    };

    let model_ids = if model_ids.is_empty() {
        vec![ModelId::BgeBase]
    } else {
        model_ids
    };

    // Create registry (lazy loading - models loaded on first request)
    let registry = ModelRegistry::new(&model_ids)?;
    
    // Create metrics
    let metrics = Metrics::new();

    // Create shutdown channel
    let (shutdown_tx, shutdown_rx) = watch::channel(false);

    // Create app state
    let state = Arc::new(AppState { 
        registry, 
        metrics,
        shutdown_rx,
    });

    // Build router
    let app = Router::new()
        .route("/health", get(health))
        .route("/metrics", get(metrics_handler))
        .route("/v1/embeddings", post(embed))
        .route("/v1/models", get(list_models))
        .with_state(state.clone());

    // Start server
    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    let addr = format!("127.0.0.1:{}", port);
    
    info!(address = %addr, "Server listening");
    info!("Endpoints:");
    info!("  POST /v1/embeddings - OpenAI-compatible embedding endpoint");
    info!("  GET  /v1/models     - List available models");
    info!("  GET  /health        - Health check with model status");
    info!("  GET  /metrics       - In-memory metrics");
    info!("");
    info!("Available models:");
    for m in &model_ids {
        info!(
            model = m.display_name(),
            dimensions = m.dimensions(),
            max_tokens = m.max_tokens(),
            "Model enabled"
        );
    }
    info!("");
    info!("Usage:");
    info!("  Single model:  {{ \"input\": \"text\", \"model\": \"bge-base\" }}");
    info!("  All models:    {{ \"input\": \"text\", \"model\": \"*\" }}");
    info!("Models loaded lazily on first request");

    let listener = tokio::net::TcpListener::bind(&addr).await?;
    
    // Use axum's graceful shutdown
    axum::serve(listener, app)
        .with_graceful_shutdown(async move {
            shutdown_signal().await;
            
            // Signal shutdown to any listeners
            let _ = shutdown_tx.send(true);
            
            // Wait for in-flight requests to complete (with timeout)
            let deadline = Instant::now() + std::time::Duration::from_secs(30);
            loop {
                let in_flight = state.metrics.in_flight_count();
                if in_flight == 0 {
                    info!("All in-flight requests completed");
                    break;
                }
                
                if Instant::now() > deadline {
                    warn!(in_flight = in_flight, "Shutdown timeout, forcing exit");
                    break;
                }
                
                info!(in_flight = in_flight, "Waiting for in-flight requests to complete...");
                tokio::time::sleep(std::time::Duration::from_millis(100)).await;
            }
            
            info!("Graceful shutdown complete");
        })
        .await?;

    Ok(())
}
