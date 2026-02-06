//! Embedding models using Candle (HuggingFace's Rust ML framework)
//!
//! Supports multiple embedding models for ensemble retrieval:
//! - bge-base-en-v1.5 (768-dim) - General purpose anchor model
//! - all-MiniLM-L6-v2 (384-dim) - Fast, good for short text
//! - nomic-embed-text-v1.5 (768-dim) - Long context, good for documents
//!
//! This module handles:
//! - Downloading models from HuggingFace Hub
//! - Loading model weights into Candle
//! - Tokenizing text
//! - Running inference to get embeddings
//! - Lazy loading with LRU eviction

use anyhow::{Context, Result};
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::bert::{BertModel, Config as BertConfig};
use hf_hub::{api::sync::Api, Repo, RepoType};
use std::collections::HashMap;
use std::sync::{Arc, RwLock};
use tokenizers::Tokenizer;
use tracing::info;

use crate::nomic_bert::{NomicBertConfig, NomicBertModel};

/// Supported model identifiers
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelId {
    BgeBase,
    MiniLM,
    GteBase,
    Nomic,
}

impl ModelId {
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "bge-base" | "bge-base-en-v1.5" | "baai/bge-base-en-v1.5" => Some(Self::BgeBase),
            "minilm" | "all-minilm-l6-v2" | "sentence-transformers/all-minilm-l6-v2" => {
                Some(Self::MiniLM)
            }
            "gte-base" | "gte" | "thenlper/gte-base" => Some(Self::GteBase),
            "nomic" | "nomic-embed-text-v1.5" | "nomic-ai/nomic-embed-text-v1.5" => {
                Some(Self::Nomic)
            }
            _ => None,
        }
    }

    pub fn to_hf_id(&self) -> &'static str {
        match self {
            Self::BgeBase => "BAAI/bge-base-en-v1.5",
            Self::MiniLM => "sentence-transformers/all-MiniLM-L6-v2",
            Self::GteBase => "thenlper/gte-base",
            Self::Nomic => "nomic-ai/nomic-embed-text-v1.5",
        }
    }

    pub fn dimensions(&self) -> usize {
        match self {
            Self::BgeBase => 768,
            Self::MiniLM => 384,
            Self::GteBase => 768,
            Self::Nomic => 768,
        }
    }

    pub fn max_tokens(&self) -> usize {
        match self {
            Self::BgeBase => 512,
            Self::MiniLM => 256,
            Self::GteBase => 512,
            Self::Nomic => 8192,
        }
    }

    pub fn display_name(&self) -> &'static str {
        match self {
            Self::BgeBase => "bge-base",
            Self::MiniLM => "minilm",
            Self::GteBase => "gte-base",
            Self::Nomic => "nomic",
        }
    }

    /// Get the prefix required for this model (if any)
    /// Some models like Nomic require task-specific prefixes for optimal performance
    pub fn prefix(&self) -> Option<&'static str> {
        match self {
            Self::Nomic => Some("search_document: "),
            _ => None,
        }
    }

    /// All available models
    pub fn all() -> &'static [ModelId] {
        &[ModelId::BgeBase, ModelId::MiniLM, ModelId::GteBase, ModelId::Nomic]
    }
}

impl std::fmt::Display for ModelId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.display_name())
    }
}

/// Model backend enum - supports different model architectures
enum ModelBackend {
    Bert(BertModel),
    NomicBert(NomicBertModel),
}

/// Single embedding model wrapper
pub struct Embedder {
    model: ModelBackend,
    tokenizer: Tokenizer,
    device: Device,
    model_id: ModelId,
    normalize: bool,
    /// Prefix to add before text (some models like Nomic need this)
    prefix: Option<String>,
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

        // Load model based on architecture
        let config_str = std::fs::read_to_string(&config_path)?;
        let model = match model_id {
            ModelId::Nomic => {
                // Nomic uses custom architecture with rotary embeddings and SwiGLU
                let config: NomicBertConfig = serde_json::from_str(&config_str)?;
                let nomic_model = NomicBertModel::load(vb, &config)?;
                ModelBackend::NomicBert(nomic_model)
            }
            _ => {
                // Standard BERT-based models
                let config: BertConfig = serde_json::from_str(&config_str)?;
                let bert_model = BertModel::load(vb, &config)?;
                ModelBackend::Bert(bert_model)
            }
        };

        // Some models require a prefix for optimal performance (e.g., Nomic)
        let prefix = model_id.prefix().map(|s| s.to_string());

        Ok(Self {
            model,
            tokenizer,
            device,
            model_id,
            normalize: true, // Sentence transformers use normalized embeddings
            prefix,
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

        // Apply prefix if required (e.g., Nomic model)
        let texts: Vec<String> = match &self.prefix {
            Some(prefix) => texts.iter().map(|t| format!("{}{}", prefix, t)).collect(),
            None => texts.to_vec(),
        };

        // Tokenize all texts
        let encodings = self
            .tokenizer
            .encode_batch(texts.clone(), true)
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
        let embeddings = match &self.model {
            ModelBackend::Bert(model) => {
                model.forward(&input_ids, &token_type_ids, Some(&attention_mask))?
            }
            ModelBackend::NomicBert(model) => {
                model.forward(&input_ids, &token_type_ids, Some(&attention_mask))?
            }
        };

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

/// Registry managing multiple embedding models with lazy loading
pub struct ModelRegistry {
    /// Lazily loaded models (loaded on first request)
    models: RwLock<HashMap<ModelId, Arc<Embedder>>>,
    /// Models enabled for this instance
    enabled_models: Vec<ModelId>,
    /// Default model to use
    default_model: ModelId,
    /// Max models to keep loaded (LRU eviction when exceeded)
    max_loaded: usize,
    /// Access order for LRU (most recent last)
    access_order: RwLock<Vec<ModelId>>,
}

impl ModelRegistry {
    /// Create a new registry (lazy loading - models loaded on first use)
    pub fn new(model_ids: &[ModelId]) -> Result<Self> {
        let default_model = model_ids.first().copied().unwrap_or(ModelId::BgeBase);
        let enabled_models = if model_ids.is_empty() {
            vec![ModelId::BgeBase]
        } else {
            model_ids.to_vec()
        };

        info!(
            "📦 Registry initialized with {} enabled model(s): {:?}",
            enabled_models.len(),
            enabled_models.iter().map(|m| m.display_name()).collect::<Vec<_>>()
        );
        info!("💤 Models will be loaded lazily on first request");

        Ok(Self {
            models: RwLock::new(HashMap::new()),
            enabled_models,
            default_model,
            max_loaded: 3, // Keep up to 3 models in memory
            access_order: RwLock::new(Vec::new()),
        })
    }

    /// Create registry and eagerly load specified models (for tests/benchmarks)
    pub fn new_eager(model_ids: &[ModelId]) -> Result<Self> {
        let registry = Self::new(model_ids)?;
        
        // Eagerly load all enabled models
        for &model_id in &registry.enabled_models {
            registry.get_or_load(model_id)?;
        }
        
        Ok(registry)
    }

    /// Get or lazily load a model
    fn get_or_load(&self, model_id: ModelId) -> Result<Arc<Embedder>> {
        // Check if already loaded
        {
            let models = self.models.read().unwrap();
            if let Some(embedder) = models.get(&model_id) {
                // Update access order
                self.update_access_order(model_id);
                return Ok(embedder.clone());
            }
        }

        // Check if model is enabled
        if !self.enabled_models.contains(&model_id) {
            return Err(anyhow::anyhow!(
                "Model '{}' is not enabled. Available models: {:?}",
                model_id.display_name(),
                self.enabled_models.iter().map(|m| m.display_name()).collect::<Vec<_>>()
            ));
        }

        // Load the model (may need to evict first)
        self.maybe_evict();
        
        info!("🔄 Loading model on first request: {}", model_id);
        let embedder = Embedder::new(model_id)
            .with_context(|| format!("Failed to load model: {}", model_id))?;
        let embedder = Arc::new(embedder);
        
        info!(
            "✅ {} loaded successfully ({} dimensions)",
            model_id,
            model_id.dimensions()
        );

        // Store and return
        {
            let mut models = self.models.write().unwrap();
            models.insert(model_id, embedder.clone());
        }
        self.update_access_order(model_id);

        Ok(embedder)
    }

    /// Update LRU access order
    fn update_access_order(&self, model_id: ModelId) {
        let mut order = self.access_order.write().unwrap();
        order.retain(|&id| id != model_id);
        order.push(model_id);
    }

    /// Evict least recently used model if at capacity
    fn maybe_evict(&self) {
        let models = self.models.read().unwrap();
        if models.len() < self.max_loaded {
            return;
        }
        drop(models);

        // Find LRU model (first in access order)
        let to_evict = {
            let order = self.access_order.read().unwrap();
            order.first().copied()
        };

        if let Some(model_id) = to_evict {
            info!("🗑️ Evicting least recently used model: {}", model_id);
            let mut models = self.models.write().unwrap();
            models.remove(&model_id);
            let mut order = self.access_order.write().unwrap();
            order.retain(|&id| id != model_id);
        }
    }

    /// Get an embedder by model ID (loads lazily)
    pub fn get(&self, model_id: ModelId) -> Option<Arc<Embedder>> {
        self.get_or_load(model_id).ok()
    }

    /// Get the default embedder (loads lazily)
    pub fn get_default(&self) -> Arc<Embedder> {
        self.get_or_load(self.default_model)
            .expect("Default model should always be loadable")
    }

    /// Get model by string name (for API, loads lazily)
    pub fn get_by_name(&self, name: &str) -> Option<Arc<Embedder>> {
        ModelId::from_str(name).and_then(|id| self.get(id))
    }

    /// List all enabled models (may not all be loaded)
    pub fn enabled_models(&self) -> Vec<ModelId> {
        self.enabled_models.clone()
    }

    /// List currently loaded models
    pub fn loaded_models(&self) -> Vec<ModelId> {
        let models = self.models.read().unwrap();
        models.keys().copied().collect()
    }

    /// Check if a model is currently loaded
    pub fn is_loaded(&self, model_id: ModelId) -> bool {
        let models = self.models.read().unwrap();
        models.contains_key(&model_id)
    }

    /// Embed with a specific model or default
    pub fn embed(&self, texts: &[String], model_name: Option<&str>) -> Result<EmbedResult> {
        let model_id = match model_name {
            Some(name) => ModelId::from_str(name)
                .ok_or_else(|| anyhow::anyhow!("Unknown model: {}", name))?,
            None => self.default_model,
        };

        let embedder = self.get_or_load(model_id)?;

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

    /// Embed with all enabled models
    pub fn embed_all(&self, texts: &[String]) -> Vec<EmbedResult> {
        self.enabled_models
            .iter()
            .filter_map(|&model_id| {
                match self.get_or_load(model_id) {
                    Ok(embedder) => {
                        let start = std::time::Instant::now();
                        match embedder.embed(texts) {
                            Ok(vectors) => Some(EmbedResult {
                                model: embedder.model_id(),
                                dimensions: embedder.dimensions(),
                                vectors,
                                latency_ms: start.elapsed().as_millis() as u64,
                            }),
                            Err(e) => {
                                tracing::error!("Embedding failed for {}: {}", model_id, e);
                                None
                            }
                        }
                    }
                    Err(e) => {
                        tracing::error!("Failed to load {}: {}", model_id, e);
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
        // BGE Base
        assert_eq!(ModelId::from_str("bge-base"), Some(ModelId::BgeBase));
        assert_eq!(
            ModelId::from_str("BAAI/bge-base-en-v1.5"),
            Some(ModelId::BgeBase)
        );
        
        // MiniLM
        assert_eq!(ModelId::from_str("minilm"), Some(ModelId::MiniLM));
        assert_eq!(
            ModelId::from_str("all-MiniLM-L6-v2"),
            Some(ModelId::MiniLM)
        );
        
        // GTE Base
        assert_eq!(ModelId::from_str("gte-base"), Some(ModelId::GteBase));
        assert_eq!(ModelId::from_str("gte"), Some(ModelId::GteBase));
        assert_eq!(
            ModelId::from_str("thenlper/gte-base"),
            Some(ModelId::GteBase)
        );
        
        // Nomic
        assert_eq!(ModelId::from_str("nomic"), Some(ModelId::Nomic));
        assert_eq!(
            ModelId::from_str("nomic-embed-text-v1.5"),
            Some(ModelId::Nomic)
        );
        assert_eq!(
            ModelId::from_str("nomic-ai/nomic-embed-text-v1.5"),
            Some(ModelId::Nomic)
        );
        
        // Unknown
        assert_eq!(ModelId::from_str("unknown"), None);
    }

    #[test]
    fn test_model_dimensions() {
        assert_eq!(ModelId::BgeBase.dimensions(), 768);
        assert_eq!(ModelId::MiniLM.dimensions(), 384);
        assert_eq!(ModelId::GteBase.dimensions(), 768);
        assert_eq!(ModelId::Nomic.dimensions(), 768);
    }

    #[test]
    fn test_model_max_tokens() {
        assert_eq!(ModelId::BgeBase.max_tokens(), 512);
        assert_eq!(ModelId::MiniLM.max_tokens(), 256);
        assert_eq!(ModelId::GteBase.max_tokens(), 512);
        assert_eq!(ModelId::Nomic.max_tokens(), 8192);
    }

    #[test]
    fn test_model_prefix() {
        assert_eq!(ModelId::BgeBase.prefix(), None);
        assert_eq!(ModelId::MiniLM.prefix(), None);
        assert_eq!(ModelId::GteBase.prefix(), None);
        assert_eq!(ModelId::Nomic.prefix(), Some("search_document: "));
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

    #[test]
    #[ignore] // Requires model download
    fn test_embedder_gte() -> Result<()> {
        let embedder = Embedder::new(ModelId::GteBase)?;

        let texts = vec!["Hello, world!".to_string()];

        let embeddings = embedder.embed(&texts)?;

        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), 768);

        // Check normalization
        let norm: f32 = embeddings[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);

        Ok(())
    }

    #[test]
    #[ignore] // Requires model download
    fn test_embedder_nomic() -> Result<()> {
        let embedder = Embedder::new(ModelId::Nomic)?;

        let texts = vec!["Hello, world!".to_string()];

        let embeddings = embedder.embed(&texts)?;

        assert_eq!(embeddings.len(), 1);
        assert_eq!(embeddings[0].len(), 768);

        // Check normalization (should be unit vector)
        let norm: f32 = embeddings[0].iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01);

        Ok(())
    }

    #[test]
    #[ignore] // Requires model download
    fn test_lazy_loading() -> Result<()> {
        let registry = ModelRegistry::new(&[ModelId::BgeBase, ModelId::MiniLM])?;
        
        // Nothing loaded initially
        assert!(registry.loaded_models().is_empty());
        
        // Request bge-base - should load
        let _result = registry.embed(&vec!["test".to_string()], Some("bge-base"))?;
        assert!(registry.is_loaded(ModelId::BgeBase));
        assert!(!registry.is_loaded(ModelId::MiniLM));
        
        // Request minilm - should load
        let _result = registry.embed(&vec!["test".to_string()], Some("minilm"))?;
        assert!(registry.is_loaded(ModelId::BgeBase));
        assert!(registry.is_loaded(ModelId::MiniLM));
        
        Ok(())
    }
}
