//! Embedding models using Candle (HuggingFace's Rust ML framework)
//!
//! Supports multiple embedding models for ensemble retrieval:
//! - bge-base-en-v1.5 (768-dim) - General purpose anchor model
//! - all-MiniLM-L6-v2 (384-dim) - Fast, good for short text
//!
//! This module handles:
//! - Downloading models from HuggingFace Hub
//! - Loading model weights into Candle
//! - Tokenizing text
//! - Running inference to get embeddings

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::collections::HashMap;
use std::sync::Arc;
use tokenizers::Tokenizer;
use tracing::info;

/// Supported model identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelId {
    BgeBase,
    MiniLM,
}

impl ModelId {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "bge-base" | "bge-base-en-v1.5" | "baai/bge-base-en-v1.5" => Some(Self::BgeBase),
            "minilm" | "all-minilm-l6-v2" | "sentence-transformers/all-minilm-l6-v2" => {
                Some(Self::MiniLM)
            }
            _ => None,
        }
    }

    pub fn to_hf_id(&self) -> &'static str {
        match self {
            Self::BgeBase => "BAAI/bge-base-en-v1.5",
            Self::MiniLM => "sentence-transformers/all-MiniLM-L6-v2",
        }
    }

    pub fn dimensions(&self) -> usize {
        match self {
            Self::BgeBase => 768,
            Self::MiniLM => 384,
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::BgeBase => "bge-base",
            Self::MiniLM => "minilm",
        }
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

/// Single embedding model wrapper
pub struct Embedder {
    model: BertModel,
    tokenizer: Tokenizer,
    device: Device,
    model_id: ModelId,
    normalize: bool,
}

impl Embedder {
    /// Load a sentence-transformers model from HuggingFace
    pub fn new(model_id: ModelId) -> Result<Self> {
        let hf_id = model_id.to_hf_id();

        // Select device (CPU for now - Metal has incomplete BERT op support)
        let device = Self::select_device()?;
        info!("Loading {} using device: {:?}", hf_id, device);

        // Download model files from HuggingFace
        let api = Api::new()?;
        let repo = api.repo(Repo::new(hf_id.to_string(), RepoType::Model));

        info!("Downloading model files for {}...", hf_id);
        let config_path = repo.get("config.json")?;
        let tokenizer_path = repo.get("tokenizer.json")?;
        let weights_path = repo
            .get("model.safetensors")
            .or_else(|_| repo.get("pytorch_model.bin"))?;

        // Load config
        let config_str = std::fs::read_to_string(&config_path)?;
        let config: BertConfig = serde_json::from_str(&config_str)?;

        // Load tokenizer
        let tokenizer = Tokenizer::from_file(&tokenizer_path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // Load model weights
        info!("Loading model weights for {}...", hf_id);
        let vb = if weights_path
            .extension()
            .map_or(false, |e| e == "safetensors")
        {
            unsafe { VarBuilder::from_mmaped_safetensors(&[weights_path], DType::F32, &device)? }
        } else {
            // PyTorch .bin format
            VarBuilder::from_pth(weights_path, DType::F32, &device)?
        };

        let model = BertModel::load(vb, &config)?;

        Ok(Self {
            model,
            tokenizer,
            device,
            model_id,
            normalize: true, // Sentence transformers use normalized embeddings
        })
    }

    /// Get the model identifier
    pub fn model_id(&self) -> ModelId {
        self.model_id
    }

    /// Get the embedding dimensions
    pub fn dimensions(&self) -> usize {
        self.model_id.dimensions()
    }

    /// Select the best available device
    fn select_device() -> Result<Device> {
        // Metal has incomplete op support for BERT models (missing layer-norm)
        // Use CPU for now - still very fast on Apple Silicon
        Ok(Device::Cpu)
    }

    /// Generate embeddings for a batch of texts
    pub fn embed(&self, texts: &[String]) -> Result<Vec<Vec<f32>>> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Tokenize all texts
        let encodings = self
            .tokenizer
            .encode_batch(texts.to_vec(), true)
            .map_err(|e| anyhow::anyhow!("Tokenization failed: {}", e))?;

        // Find max length for padding
        let max_len = encodings
            .iter()
            .map(|e| e.get_ids().len())
            .max()
            .unwrap_or(0);

        // Create input tensors
        let mut all_input_ids = Vec::new();
        let mut all_attention_mask = Vec::new();
        let mut all_token_type_ids = Vec::new();

        for encoding in &encodings {
            let ids = encoding.get_ids();
            let mask = encoding.get_attention_mask();
            let type_ids = encoding.get_type_ids();

            // Pad to max length
            let mut padded_ids = ids.to_vec();
            let mut padded_mask = mask.to_vec();
            let mut padded_types = type_ids.to_vec();

            padded_ids.resize(max_len, 0);
            padded_mask.resize(max_len, 0);
            padded_types.resize(max_len, 0);

            all_input_ids.extend(padded_ids.iter().map(|&x| x as i64));
            all_attention_mask.extend(padded_mask.iter().map(|&x| x as i64));
            all_token_type_ids.extend(padded_types.iter().map(|&x| x as i64));
        }

        let batch_size = texts.len();
        let input_ids = Tensor::from_vec(all_input_ids, (batch_size, max_len), &self.device)?;
        let attention_mask =
            Tensor::from_vec(all_attention_mask, (batch_size, max_len), &self.device)?;
        let token_type_ids =
            Tensor::from_vec(all_token_type_ids, (batch_size, max_len), &self.device)?;

        // Run model forward pass
        let embeddings =
            self.model
                .forward(&input_ids, &token_type_ids, Some(&attention_mask))?;

        // Mean pooling over sequence length (with attention mask)
        let pooled = self.mean_pooling(&embeddings, &attention_mask)?;

        // Normalize if requested
        let final_embeddings = if self.normalize {
            self.normalize_l2(&pooled)?
        } else {
            pooled
        };

        // Convert to Vec<Vec<f32>>
        let result = final_embeddings.to_vec2::<f32>()?;
        Ok(result)
    }

    /// Mean pooling: average token embeddings, weighted by attention mask
    fn mean_pooling(&self, embeddings: &Tensor, attention_mask: &Tensor) -> Result<Tensor> {
        // embeddings: (batch, seq_len, hidden_size)
        // attention_mask: (batch, seq_len)

        // Expand attention mask to match embedding dimensions
        let mask = attention_mask.unsqueeze(2)?.to_dtype(DType::F32)?;

        // Multiply embeddings by mask and sum
        let masked = embeddings.broadcast_mul(&mask)?;
        let summed = masked.sum(1)?;

        // Divide by sum of mask (number of non-padding tokens)
        let mask_sum = mask.sum(1)?.clamp(1e-9, f64::INFINITY)?;
        let pooled = summed.broadcast_div(&mask_sum)?;

        Ok(pooled)
    }

    /// L2 normalize embeddings (unit vectors)
    fn normalize_l2(&self, embeddings: &Tensor) -> Result<Tensor> {
        let norm = embeddings
            .sqr()?
            .sum_keepdim(1)?
            .sqrt()?
            .clamp(1e-9, f64::INFINITY)?;
        let normalized = embeddings.broadcast_div(&norm)?;
        Ok(normalized)
    }
}

/// Registry managing multiple embedding models
pub struct ModelRegistry {
    models: HashMap<ModelId, Arc<Embedder>>,
    default_model: ModelId,
}

impl ModelRegistry {
    /// Create a new registry and load specified models
    pub fn new(model_ids: &[ModelId]) -> Result<Self> {
        let mut models = HashMap::new();
        let default_model = model_ids.first().copied().unwrap_or(ModelId::BgeBase);

        for &model_id in model_ids {
            info!("Loading model: {}", model_id);
            let embedder = Embedder::new(model_id)
                .with_context(|| format!("Failed to load model: {}", model_id))?;
            models.insert(model_id, Arc::new(embedder));
            info!(
                "✅ {} loaded successfully ({} dimensions)",
                model_id,
                model_id.dimensions()
            );
        }

        Ok(Self {
            models,
            default_model,
        })
    }

    /// Get an embedder by model ID
    pub fn get(&self, model_id: ModelId) -> Option<Arc<Embedder>> {
        self.models.get(&model_id).cloned()
    }

    /// Get the default embedder
    pub fn get_default(&self) -> Arc<Embedder> {
        self.models
            .get(&self.default_model)
            .cloned()
            .expect("Default model should always be loaded")
    }

    /// Get model by string name (for API)
    pub fn get_by_name(&self, name: &str) -> Option<Arc<Embedder>> {
        ModelId::from_str(name).and_then(|id| self.get(id))
    }

    /// List all loaded models
    pub fn loaded_models(&self) -> Vec<ModelId> {
        self.models.keys().copied().collect()
    }

    /// Embed with a specific model or default
    pub fn embed(&self, texts: &[String], model_name: Option<&str>) -> Result<EmbedResult> {
        let embedder = match model_name {
            Some(name) => self
                .get_by_name(name)
                .ok_or_else(|| anyhow::anyhow!("Unknown model: {}", name))?,
            None => self.get_default(),
        };

        let start = std::time::Instant::now();
        let vectors = embedder.embed(texts)?;
        let latency_ms = start.elapsed().as_millis() as u64;

        Ok(EmbedResult {
            model: embedder.model_id(),
            dimensions: embedder.dimensions(),
            vectors,
            latency_ms,
        })
    }

    /// Embed with all loaded models in parallel
    pub fn embed_all(&self, texts: &[String]) -> Vec<EmbedResult> {
        // For now, sequential - could parallelize with rayon
        self.models
            .values()
            .filter_map(|embedder| {
                let start = std::time::Instant::now();
                match embedder.embed(texts) {
                    Ok(vectors) => Some(EmbedResult {
                        model: embedder.model_id(),
                        dimensions: embedder.dimensions(),
                        vectors,
                        latency_ms: start.elapsed().as_millis() as u64,
                    }),
                    Err(e) => {
                        tracing::error!("Embedding failed for {}: {}", embedder.model_id(), e);
                        None
                    }
                }
            })
            .collect()
    }
}

/// Result of an embedding operation
#[derive(Debug, Clone)]
pub struct EmbedResult {
    pub model: ModelId,
    pub dimensions: usize,
    pub vectors: Vec<Vec<f32>>,
    pub latency_ms: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_id_parsing() {
        assert_eq!(ModelId::from_str("bge-base"), Some(ModelId::BgeBase));
        assert_eq!(
            ModelId::from_str("BAAI/bge-base-en-v1.5"),
            Some(ModelId::BgeBase)
        );
        assert_eq!(ModelId::from_str("minilm"), Some(ModelId::MiniLM));
        assert_eq!(
            ModelId::from_str("all-MiniLM-L6-v2"),
            Some(ModelId::MiniLM)
        );
        assert_eq!(ModelId::from_str("unknown"), None);
    }

    #[test]
    #[ignore] // Requires model download
    fn test_embedder_bge() -> Result<()> {
        let embedder = Embedder::new(ModelId::BgeBase)?;

        let texts = vec!["Hello, world!".to_string(), "This is a test.".to_string()];

        let embeddings = embedder.embed(&texts)?;

        assert_eq!(embeddings.len(), 2);
        assert_eq!(embeddings[0].len(), 768);

        // Check normalization (should be unit vectors)
        let norm: f32 = embeddings[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);

        Ok(())
    }

    #[test]
    #[ignore] // Requires model download
    fn test_embedder_minilm() -> Result<()> {
        let embedder = Embedder::new(ModelId::MiniLM)?;

        let texts = vec!["Hello, world!".to_string()];

        let embeddings = embedder.embed(&texts)?;

        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), 384);

        Ok(())
    }
}
