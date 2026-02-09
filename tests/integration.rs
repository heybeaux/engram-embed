//! Integration tests for engram-embed server
//!
//! These tests require a running server or will start one.
//! Run with: cargo test --test integration -- --ignored
//!
//! For faster iteration, start server manually:
//!   EMBED_MODELS=minilm cargo run
//!
//! Then run tests:
//!   ENGRAM_EMBED_URL=http://127.0.0.1:8080 cargo test --test integration

mod fixtures;

use reqwest::Client;
use serde_json::{json, Value};
use std::process::{Child, Command};
use std::time::Duration;

/// Get the server URL from env or default
fn server_url() -> String {
    std::env::var("ENGRAM_EMBED_URL").unwrap_or_else(|_| "http://127.0.0.1:8080".to_string())
}

/// Helper to start a server process for testing
struct TestServer {
    child: Option<Child>,
    url: String,
}

impl TestServer {
    /// Start a new test server or use existing one from env
    async fn new() -> Self {
        // If URL is set, assume server is already running
        if std::env::var("ENGRAM_EMBED_URL").is_ok() {
            return Self {
                child: None,
                url: server_url(),
            };
        }

        // Start server with minilm (smallest/fastest for tests)
        let child = Command::new("cargo")
            .args(["run", "--release"])
            .env("EMBED_MODELS", "minilm")
            .env("PORT", "18080")
            .spawn()
            .expect("Failed to start test server");

        let url = "http://127.0.0.1:18080".to_string();

        // Wait for server to be ready
        let client = Client::new();
        for _ in 0..60 {
            tokio::time::sleep(Duration::from_secs(1)).await;
            if client.get(&format!("{}/health", url)).send().await.is_ok() {
                break;
            }
        }

        Self {
            child: Some(child),
            url,
        }
    }

    fn url(&self) -> &str {
        &self.url
    }
}

impl Drop for TestServer {
    fn drop(&mut self) {
        if let Some(ref mut child) = self.child {
            let _ = child.kill();
        }
    }
}

// ============================================================================
// Health Check Tests
// ============================================================================

#[tokio::test]
#[ignore = "requires server"]
async fn test_health_check() {
    let client = Client::new();
    let url = server_url();

    let response = client
        .get(&format!("{}/health", url))
        .send()
        .await
        .expect("Failed to send request");

    assert!(response.status().is_success());

    let body: Value = response.json().await.expect("Failed to parse JSON");

    // Verify response structure
    assert_eq!(body["status"], "ok");
    assert!(body["models"].is_array());
    assert!(body["version"].is_string());
    assert!(body["loaded_count"].is_number());
}

#[tokio::test]
#[ignore = "requires server"]
async fn test_health_check_model_info() {
    let client = Client::new();
    let url = server_url();

    let response = client
        .get(&format!("{}/health", url))
        .send()
        .await
        .expect("Failed to send request");

    let body: Value = response.json().await.expect("Failed to parse JSON");
    let models = body["models"].as_array().expect("models should be array");

    // At least one model should be configured
    assert!(!models.is_empty());

    // Each model should have required fields
    for model in models {
        assert!(model["id"].is_string());
        assert!(model["dimensions"].is_number());
        assert!(model["max_tokens"].is_number());
        assert!(model["loaded"].is_boolean());
        assert!(model["default"].is_boolean());
    }
}

// ============================================================================
// Basic Embedding Tests
// ============================================================================

#[tokio::test]
#[ignore = "requires server"]
async fn test_single_embedding() {
    let client = Client::new();
    let url = server_url();

    let response = client
        .post(&format!("{}/v1/embeddings", url))
        .json(&json!({
            "input": fixtures::SHORT_TEXT,
            "model": "minilm"
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(response.status().is_success());

    let body: Value = response.json().await.expect("Failed to parse JSON");

    // OpenAI-compatible response format
    assert_eq!(body["object"], "list");
    assert!(body["data"].is_array());
    assert!(body["model"].is_string());
    assert!(body["usage"].is_object());

    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);

    let embedding = &data[0];
    assert_eq!(embedding["object"], "embedding");
    assert_eq!(embedding["index"], 0);
    assert!(embedding["embedding"].is_array());

    // MiniLM has 384 dimensions
    let vector = embedding["embedding"].as_array().unwrap();
    assert_eq!(vector.len(), 384);
}

#[tokio::test]
#[ignore = "requires server"]
async fn test_batch_embedding() {
    let client = Client::new();
    let url = server_url();

    let texts = vec![
        "First sentence for embedding.",
        "Second sentence for embedding.",
        "Third sentence for embedding.",
    ];

    let response = client
        .post(&format!("{}/v1/embeddings", url))
        .json(&json!({
            "input": texts,
            "model": "minilm"
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(response.status().is_success());

    let body: Value = response.json().await.expect("Failed to parse JSON");
    let data = body["data"].as_array().unwrap();

    // Should have 3 embeddings
    assert_eq!(data.len(), 3);

    // Each embedding should have correct index
    for (i, embedding) in data.iter().enumerate() {
        assert_eq!(embedding["index"], i);
        let vector = embedding["embedding"].as_array().unwrap();
        assert_eq!(vector.len(), 384);
    }
}

// ============================================================================
// Edge Case Tests
// ============================================================================

#[tokio::test]
#[ignore = "requires server"]
async fn test_empty_string_embedding() {
    let client = Client::new();
    let url = server_url();

    let response = client
        .post(&format!("{}/v1/embeddings", url))
        .json(&json!({
            "input": "",
            "model": "minilm"
        }))
        .send()
        .await
        .expect("Failed to send request");

    // Empty string should still produce an embedding
    assert!(response.status().is_success());

    let body: Value = response.json().await.expect("Failed to parse JSON");
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);

    let vector = data[0]["embedding"].as_array().unwrap();
    assert_eq!(vector.len(), 384);
}

#[tokio::test]
#[ignore = "requires server"]
async fn test_unicode_text_embedding() {
    let client = Client::new();
    let url = server_url();

    let response = client
        .post(&format!("{}/v1/embeddings", url))
        .json(&json!({
            "input": fixtures::UNICODE_TEXT,
            "model": "minilm"
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(response.status().is_success());

    let body: Value = response.json().await.expect("Failed to parse JSON");
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);

    let vector = data[0]["embedding"].as_array().unwrap();
    assert_eq!(vector.len(), 384);
}

#[tokio::test]
#[ignore = "requires server"]
async fn test_long_text_truncation() {
    let client = Client::new();
    let url = server_url();

    let response = client
        .post(&format!("{}/v1/embeddings", url))
        .json(&json!({
            "input": fixtures::LONG_TEXT,
            "model": "minilm"
        }))
        .send()
        .await
        .expect("Failed to send request");

    // Long text should be truncated, not error
    assert!(response.status().is_success());

    let body: Value = response.json().await.expect("Failed to parse JSON");
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 1);

    let vector = data[0]["embedding"].as_array().unwrap();
    assert_eq!(vector.len(), 384);
}

#[tokio::test]
#[ignore = "requires server"]
async fn test_code_text_embedding() {
    let client = Client::new();
    let url = server_url();

    let response = client
        .post(&format!("{}/v1/embeddings", url))
        .json(&json!({
            "input": fixtures::CODE_TEXT,
            "model": "minilm"
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(response.status().is_success());

    let body: Value = response.json().await.expect("Failed to parse JSON");
    let vector = body["data"][0]["embedding"].as_array().unwrap();
    assert_eq!(vector.len(), 384);
}

// ============================================================================
// Error Handling Tests
// ============================================================================

#[tokio::test]
#[ignore = "requires server"]
async fn test_invalid_model_error() {
    let client = Client::new();
    let url = server_url();

    let response = client
        .post(&format!("{}/v1/embeddings", url))
        .json(&json!({
            "input": "test text",
            "model": "nonexistent-model-xyz"
        }))
        .send()
        .await
        .expect("Failed to send request");

    // Should return 400 Bad Request
    assert_eq!(response.status().as_u16(), 400);

    let body = response.text().await.expect("Failed to get response text");
    assert!(body.contains("Unknown model") || body.contains("not enabled"));
}

#[tokio::test]
#[ignore = "requires server"]
async fn test_invalid_json_error() {
    let client = Client::new();
    let url = server_url();

    let response = client
        .post(&format!("{}/v1/embeddings", url))
        .header("Content-Type", "application/json")
        .body("{invalid json}")
        .send()
        .await
        .expect("Failed to send request");

    // Should return error status
    assert!(response.status().is_client_error());
}

// ============================================================================
// OpenAI Compatibility Tests
// ============================================================================

#[tokio::test]
#[ignore = "requires server"]
async fn test_openai_compatible_response_format() {
    let client = Client::new();
    let url = server_url();

    let response = client
        .post(&format!("{}/v1/embeddings", url))
        .json(&json!({
            "input": "test text"
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(response.status().is_success());

    let body: Value = response.json().await.expect("Failed to parse JSON");

    // Required OpenAI fields
    assert_eq!(body["object"], "list");
    assert!(body["data"].is_array());
    assert!(body["model"].is_string());
    assert!(body["usage"]["prompt_tokens"].is_number());
    assert!(body["usage"]["total_tokens"].is_number());

    let data = body["data"].as_array().unwrap();
    assert!(!data.is_empty());

    let embedding = &data[0];
    assert_eq!(embedding["object"], "embedding");
    assert!(embedding["index"].is_number());
    assert!(embedding["embedding"].is_array());
}

#[tokio::test]
#[ignore = "requires server"]
async fn test_list_models_endpoint() {
    let client = Client::new();
    let url = server_url();

    let response = client
        .get(&format!("{}/v1/models", url))
        .send()
        .await
        .expect("Failed to send request");

    assert!(response.status().is_success());

    let body: Value = response.json().await.expect("Failed to parse JSON");

    assert_eq!(body["object"], "list");
    assert!(body["data"].is_array());

    let models = body["data"].as_array().unwrap();
    assert!(!models.is_empty());

    for model in models {
        assert!(model["id"].is_string());
        assert_eq!(model["object"], "model");
        assert!(model["dimensions"].is_number());
        assert!(model["max_tokens"].is_number());
    }
}

// ============================================================================
// Embedding Quality Tests
// ============================================================================

#[tokio::test]
#[ignore = "requires server"]
async fn test_embedding_normalization() {
    let client = Client::new();
    let url = server_url();

    let response = client
        .post(&format!("{}/v1/embeddings", url))
        .json(&json!({
            "input": "test text for normalization check"
        }))
        .send()
        .await
        .expect("Failed to send request");

    let body: Value = response.json().await.expect("Failed to parse JSON");
    let vector: Vec<f64> = body["data"][0]["embedding"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_f64().unwrap())
        .collect();

    // Calculate L2 norm
    let norm: f64 = vector.iter().map(|x| x * x).sum::<f64>().sqrt();

    // Should be approximately 1.0 (unit vector)
    assert!(
        (norm - 1.0).abs() < 0.01,
        "Embedding norm should be ~1.0, got {}",
        norm
    );
}

#[tokio::test]
#[ignore = "requires server"]
async fn test_similar_texts_similar_embeddings() {
    let client = Client::new();
    let url = server_url();

    let texts = vec![
        "The cat sat on the mat.",
        "A cat is sitting on a mat.",
        "The weather is nice today.",
    ];

    let response = client
        .post(&format!("{}/v1/embeddings", url))
        .json(&json!({
            "input": texts
        }))
        .send()
        .await
        .expect("Failed to send request");

    let body: Value = response.json().await.expect("Failed to parse JSON");
    let data = body["data"].as_array().unwrap();

    let get_vector = |i: usize| -> Vec<f64> {
        data[i]["embedding"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_f64().unwrap())
            .collect()
    };

    let v0 = get_vector(0);
    let v1 = get_vector(1);
    let v2 = get_vector(2);

    // Cosine similarity helper
    let cosine_sim = |a: &[f64], b: &[f64]| -> f64 {
        let dot: f64 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f64 = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let norm_b: f64 = b.iter().map(|x| x * x).sum::<f64>().sqrt();
        dot / (norm_a * norm_b)
    };

    let sim_01 = cosine_sim(&v0, &v1); // Similar sentences
    let sim_02 = cosine_sim(&v0, &v2); // Different sentences

    // Similar sentences should have higher similarity
    assert!(
        sim_01 > sim_02,
        "Similar texts should have higher similarity: sim(0,1)={} should be > sim(0,2)={}",
        sim_01,
        sim_02
    );
}

// ============================================================================
// Multi-Model Tests (if multiple models are enabled)
// ============================================================================

#[tokio::test]
#[ignore = "requires server with multiple models"]
async fn test_multi_model_embedding() {
    let client = Client::new();
    let url = server_url();

    let response = client
        .post(&format!("{}/v1/embeddings", url))
        .json(&json!({
            "input": "test text",
            "model": "*"
        }))
        .send()
        .await
        .expect("Failed to send request");

    assert!(response.status().is_success());

    let body: Value = response.json().await.expect("Failed to parse JSON");

    // Multi-model response format
    assert_eq!(body["object"], "list");
    assert!(body["embeddings"].is_array());
    assert!(body["timing"].is_object());

    let embeddings = body["embeddings"].as_array().unwrap();
    assert!(!embeddings.is_empty());

    for model_result in embeddings {
        assert!(model_result["model"].is_string());
        assert!(model_result["dimensions"].is_number());
        assert!(model_result["data"].is_array());
    }
}

// ============================================================================
// Performance Tests
// ============================================================================

#[tokio::test]
#[ignore = "requires server"]
async fn test_batch_embedding_performance() {
    let client = Client::new();
    let url = server_url();

    // Create a batch of 10 texts
    let texts: Vec<String> = (0..10)
        .map(|i| format!("This is test sentence number {} for batch embedding.", i))
        .collect();

    let start = std::time::Instant::now();

    let response = client
        .post(&format!("{}/v1/embeddings", url))
        .json(&json!({
            "input": texts
        }))
        .send()
        .await
        .expect("Failed to send request");

    let elapsed = start.elapsed();

    assert!(response.status().is_success());

    let body: Value = response.json().await.expect("Failed to parse JSON");
    let data = body["data"].as_array().unwrap();
    assert_eq!(data.len(), 10);

    // Batch of 10 should complete in reasonable time (< 5 seconds)
    assert!(
        elapsed.as_secs() < 5,
        "Batch embedding took too long: {:?}",
        elapsed
    );
}
