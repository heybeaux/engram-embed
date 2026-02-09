//! Metal-compatible BERT model implementation
//!
//! Standard BERT architecture (compatible with BGE, MiniLM, GTE models)
//! using MetalSafeLayerNorm instead of candle_nn::LayerNorm.
//!
//! Key differences from NomicBert:
//! - Absolute positional embeddings (not rotary)
//! - Post-norm architecture (LayerNorm after residual)
//! - GELU activation (not SwiGLU)
//! - Separate Q, K, V projections (not combined QKV)

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Tensor, D};
use candle_nn::{Embedding, Module, VarBuilder};
use serde::Deserialize;

use crate::metal_compat::{metal_safe_layer_norm, MetalSafeLayerNorm};

/// BERT configuration (compatible with HuggingFace config.json)
#[derive(Debug, Clone, Deserialize)]
pub struct MetalBertConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub intermediate_size: usize,
    pub hidden_act: HiddenAct,
    #[serde(default = "default_hidden_dropout_prob")]
    pub hidden_dropout_prob: f64,
    pub max_position_embeddings: usize,
    pub type_vocab_size: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_eps: f64,
}

fn default_hidden_dropout_prob() -> f64 {
    0.1
}

fn default_layer_norm_eps() -> f64 {
    1e-12
}

#[derive(Debug, Clone, Copy, Deserialize, Default)]
#[serde(rename_all = "lowercase")]
pub enum HiddenAct {
    #[default]
    Gelu,
    Relu,
}

impl MetalBertConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

/// BERT Embeddings: word + position + token_type
struct MetalBertEmbeddings {
    word_embeddings: Embedding,
    position_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: MetalSafeLayerNorm,
}

impl MetalBertEmbeddings {
    fn new(vb: VarBuilder, cfg: &MetalBertConfig) -> Result<Self> {
        let word_embeddings = candle_nn::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb.pp("word_embeddings"),
        )?;
        let position_embeddings = candle_nn::embedding(
            cfg.max_position_embeddings,
            cfg.hidden_size,
            vb.pp("position_embeddings"),
        )?;
        let token_type_embeddings = candle_nn::embedding(
            cfg.type_vocab_size,
            cfg.hidden_size,
            vb.pp("token_type_embeddings"),
        )?;
        let layer_norm = metal_safe_layer_norm(
            cfg.hidden_size,
            cfg.layer_norm_eps,
            vb.pp("LayerNorm"),
        )?;

        Ok(Self {
            word_embeddings,
            position_embeddings,
            token_type_embeddings,
            layer_norm,
        })
    }

    fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        position_ids: Option<&Tensor>,
    ) -> Result<Tensor> {
        let seq_len = input_ids.dim(1)?;
        let device = input_ids.device();

        // Create position IDs if not provided
        let position_ids = match position_ids {
            Some(ids) => ids.clone(),
            None => {
                let positions: Vec<i64> = (0..seq_len as i64).collect();
                let positions = Tensor::from_vec(positions, (1, seq_len), device)?;
                positions.broadcast_as(input_ids.shape())?
            }
        };

        let word_emb = self.word_embeddings.forward(input_ids)?;
        let position_emb = self.position_embeddings.forward(&position_ids)?;
        let token_type_emb = self.token_type_embeddings.forward(token_type_ids)?;

        let embeddings = ((word_emb + position_emb)? + token_type_emb)?;
        self.layer_norm.forward(&embeddings).map_err(Into::into)
    }
}

/// BERT Self-Attention
struct MetalBertSelfAttention {
    query: Tensor,
    query_bias: Tensor,
    key: Tensor,
    key_bias: Tensor,
    value: Tensor,
    value_bias: Tensor,
    num_heads: usize,
    head_dim: usize,
}

impl MetalBertSelfAttention {
    fn new(vb: VarBuilder, cfg: &MetalBertConfig) -> Result<Self> {
        let hidden_size = cfg.hidden_size;

        let query = vb.get((hidden_size, hidden_size), "query.weight")?;
        let query_bias = vb.get(hidden_size, "query.bias")?;
        let key = vb.get((hidden_size, hidden_size), "key.weight")?;
        let key_bias = vb.get(hidden_size, "key.bias")?;
        let value = vb.get((hidden_size, hidden_size), "value.weight")?;
        let value_bias = vb.get(hidden_size, "value.bias")?;

        Ok(Self {
            query,
            query_bias,
            key,
            key_bias,
            value,
            value_bias,
            num_heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim(),
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = hidden_states.dims3()?;

        // Flatten to 2D for matmul
        let hidden_2d = hidden_states.reshape((batch_size * seq_len, hidden_size))?;

        // Q, K, V projections
        let q = hidden_2d
            .matmul(&self.query.t()?)?
            .broadcast_add(&self.query_bias)?;
        let k = hidden_2d
            .matmul(&self.key.t()?)?
            .broadcast_add(&self.key_bias)?;
        let v = hidden_2d
            .matmul(&self.value.t()?)?
            .broadcast_add(&self.value_bias)?;

        // Reshape to (batch, heads, seq, head_dim)
        let q = q
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;
        let k = k
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;
        let v = v
            .reshape((batch_size, seq_len, self.num_heads, self.head_dim))?
            .permute((0, 2, 1, 3))?;

        // Scaled dot-product attention
        let scale = 1.0 / (self.head_dim as f64).sqrt();
        let k_t = k.transpose(D::Minus2, D::Minus1)?.contiguous()?;
        let scores = q.contiguous()?.matmul(&k_t)?;
        let scores = (scores * scale)?;

        // Apply attention mask if provided
        let scores = match attention_mask {
            Some(mask) => {
                // mask: (batch, seq) -> (batch, 1, 1, seq)
                let mask = mask.unsqueeze(1)?.unsqueeze(1)?;
                let mask = mask.to_dtype(scores.dtype())?;
                // Convert 0/1 mask to additive mask (-inf for padding)
                let ones = Tensor::ones_like(&mask)?;
                let inv_mask = ones.sub(&mask)?;
                let neg_inf = Tensor::new(&[-1e9f32], mask.device())?;
                let mask = inv_mask.broadcast_mul(&neg_inf)?;
                scores.broadcast_add(&mask)?
            }
            None => scores,
        };

        let attn_weights = candle_nn::ops::softmax(&scores, D::Minus1)?;
        let attn_output = attn_weights.matmul(&v.contiguous()?)?;

        // Reshape back to (batch, seq, hidden)
        let attn_output = attn_output.permute((0, 2, 1, 3))?;
        attn_output
            .reshape((batch_size, seq_len, hidden_size))
            .map_err(Into::into)
    }
}

/// BERT Self-Attention Output (dense + LayerNorm)
struct MetalBertSelfOutput {
    dense: Tensor,
    dense_bias: Tensor,
    layer_norm: MetalSafeLayerNorm,
}

impl MetalBertSelfOutput {
    fn new(vb: VarBuilder, cfg: &MetalBertConfig) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let dense = vb.get((hidden_size, hidden_size), "dense.weight")?;
        let dense_bias = vb.get(hidden_size, "dense.bias")?;
        let layer_norm =
            metal_safe_layer_norm(hidden_size, cfg.layer_norm_eps, vb.pp("LayerNorm"))?;

        Ok(Self {
            dense,
            dense_bias,
            layer_norm,
        })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = hidden_states.dims3()?;

        // Dense projection
        let hidden_2d = hidden_states.reshape((batch_size * seq_len, hidden_size))?;
        let hidden = hidden_2d
            .matmul(&self.dense.t()?)?
            .broadcast_add(&self.dense_bias)?;
        let hidden = hidden.reshape((batch_size, seq_len, hidden_size))?;

        // Residual + LayerNorm (post-norm)
        let hidden = (hidden + input_tensor)?;
        self.layer_norm.forward(&hidden).map_err(Into::into)
    }
}

/// BERT Attention (self-attention + output)
struct MetalBertAttention {
    self_attention: MetalBertSelfAttention,
    output: MetalBertSelfOutput,
}

impl MetalBertAttention {
    fn new(vb: VarBuilder, cfg: &MetalBertConfig) -> Result<Self> {
        let self_attention = MetalBertSelfAttention::new(vb.pp("self"), cfg)?;
        let output = MetalBertSelfOutput::new(vb.pp("output"), cfg)?;

        Ok(Self {
            self_attention,
            output,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let self_output = self.self_attention.forward(hidden_states, attention_mask)?;
        self.output.forward(&self_output, hidden_states)
    }
}

/// BERT Intermediate (first FFN layer)
struct MetalBertIntermediate {
    dense: Tensor,
    dense_bias: Tensor,
    act: HiddenAct,
}

impl MetalBertIntermediate {
    fn new(vb: VarBuilder, cfg: &MetalBertConfig) -> Result<Self> {
        let dense = vb.get((cfg.intermediate_size, cfg.hidden_size), "dense.weight")?;
        let dense_bias = vb.get(cfg.intermediate_size, "dense.bias")?;

        Ok(Self {
            dense,
            dense_bias,
            act: cfg.hidden_act,
        })
    }

    fn forward(&self, hidden_states: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, hidden_size) = hidden_states.dims3()?;

        let hidden_2d = hidden_states.reshape((batch_size * seq_len, hidden_size))?;
        let hidden = hidden_2d
            .matmul(&self.dense.t()?)?
            .broadcast_add(&self.dense_bias)?;

        let hidden = match self.act {
            HiddenAct::Gelu => hidden.gelu()?,
            HiddenAct::Relu => hidden.relu()?,
        };

        hidden
            .reshape((batch_size, seq_len, ()))
            .map_err(Into::into)
    }
}

/// BERT Output (second FFN layer + LayerNorm)
struct MetalBertOutput {
    dense: Tensor,
    dense_bias: Tensor,
    layer_norm: MetalSafeLayerNorm,
}

impl MetalBertOutput {
    fn new(vb: VarBuilder, cfg: &MetalBertConfig) -> Result<Self> {
        let dense = vb.get((cfg.hidden_size, cfg.intermediate_size), "dense.weight")?;
        let dense_bias = vb.get(cfg.hidden_size, "dense.bias")?;
        let layer_norm =
            metal_safe_layer_norm(cfg.hidden_size, cfg.layer_norm_eps, vb.pp("LayerNorm"))?;

        Ok(Self {
            dense,
            dense_bias,
            layer_norm,
        })
    }

    fn forward(&self, hidden_states: &Tensor, input_tensor: &Tensor) -> Result<Tensor> {
        let (batch_size, seq_len, intermediate_size) = hidden_states.dims3()?;

        let hidden_2d = hidden_states.reshape((batch_size * seq_len, intermediate_size))?;
        let hidden = hidden_2d
            .matmul(&self.dense.t()?)?
            .broadcast_add(&self.dense_bias)?;

        let hidden_size = input_tensor.dim(2)?;
        let hidden = hidden.reshape((batch_size, seq_len, hidden_size))?;

        // Residual + LayerNorm (post-norm)
        let hidden = (hidden + input_tensor)?;
        self.layer_norm.forward(&hidden).map_err(Into::into)
    }
}

/// BERT Encoder Layer
struct MetalBertLayer {
    attention: MetalBertAttention,
    intermediate: MetalBertIntermediate,
    output: MetalBertOutput,
}

impl MetalBertLayer {
    fn new(vb: VarBuilder, cfg: &MetalBertConfig) -> Result<Self> {
        let attention = MetalBertAttention::new(vb.pp("attention"), cfg)?;
        let intermediate = MetalBertIntermediate::new(vb.pp("intermediate"), cfg)?;
        let output = MetalBertOutput::new(vb.pp("output"), cfg)?;

        Ok(Self {
            attention,
            intermediate,
            output,
        })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let attention_output = self.attention.forward(hidden_states, attention_mask)?;
        let intermediate_output = self.intermediate.forward(&attention_output)?;
        self.output.forward(&intermediate_output, &attention_output)
    }
}

/// BERT Encoder (stack of layers)
struct MetalBertEncoder {
    layers: Vec<MetalBertLayer>,
}

impl MetalBertEncoder {
    fn new(vb: VarBuilder, cfg: &MetalBertConfig) -> Result<Self> {
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let layer = MetalBertLayer::new(vb.pp(format!("layer.{}", i)), cfg)?;
            layers.push(layer);
        }

        Ok(Self { layers })
    }

    fn forward(&self, hidden_states: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let mut hidden = hidden_states.clone();
        for layer in &self.layers {
            hidden = layer.forward(&hidden, attention_mask)?;
        }
        Ok(hidden)
    }
}

/// Metal-compatible BERT Model
///
/// Drop-in replacement for candle_transformers::models::bert::BertModel
/// that uses MetalSafeLayerNorm for Metal GPU compatibility.
pub struct MetalBertModel {
    embeddings: MetalBertEmbeddings,
    encoder: MetalBertEncoder,
}

impl MetalBertModel {
    /// Load model from VarBuilder
    pub fn load(vb: VarBuilder, cfg: &MetalBertConfig) -> Result<Self> {
        let embeddings = MetalBertEmbeddings::new(vb.pp("embeddings"), cfg)?;
        let encoder = MetalBertEncoder::new(vb.pp("encoder"), cfg)?;

        Ok(Self {
            embeddings,
            encoder,
        })
    }

    /// Forward pass
    ///
    /// Arguments:
    /// - input_ids: Token IDs (batch_size, seq_len)
    /// - token_type_ids: Segment IDs (batch_size, seq_len)
    /// - attention_mask: Optional mask (batch_size, seq_len), 1 for real tokens, 0 for padding
    ///
    /// Returns:
    /// - Hidden states (batch_size, seq_len, hidden_size)
    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let embeddings = self.embeddings.forward(input_ids, token_type_ids, None)?;
        self.encoder.forward(&embeddings, attention_mask)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Device;

    fn test_config() -> MetalBertConfig {
        MetalBertConfig {
            vocab_size: 30522,
            hidden_size: 768,
            num_hidden_layers: 2, // Reduced for testing
            num_attention_heads: 12,
            intermediate_size: 3072,
            hidden_act: HiddenAct::Gelu,
            hidden_dropout_prob: 0.1,
            max_position_embeddings: 512,
            type_vocab_size: 2,
            layer_norm_eps: 1e-12,
        }
    }

    #[test]
    fn test_config_head_dim() {
        let cfg = test_config();
        assert_eq!(cfg.head_dim(), 64); // 768 / 12 = 64
    }

    #[test]
    fn test_hidden_act_default() {
        let act: HiddenAct = Default::default();
        assert!(matches!(act, HiddenAct::Gelu));
    }
}
