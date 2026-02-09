//! NomicBert model implementation for Candle
//!
//! Based on nomic-ai/nomic-embed-text-v1.5
//! Key differences from standard BERT:
//! - Rotary Position Embeddings (RoPE) instead of absolute positional embeddings
//! - SwiGLU activation instead of GELU
//! - Combined QKV projection
//! - Pre-norm architecture

use anyhow::Result;
use candle_core::{Device, IndexOp, Tensor, D};
use candle_nn::{Embedding, Module, VarBuilder};
use serde::Deserialize;

use crate::metal_compat::{metal_safe_layer_norm, MetalSafeLayerNorm};

/// NomicBert configuration
#[derive(Debug, Clone, Deserialize)]
pub struct NomicBertConfig {
    #[serde(default = "default_vocab_size")]
    pub vocab_size: usize,
    #[serde(alias = "n_embd")]
    pub hidden_size: usize,
    #[serde(alias = "n_layer")]
    pub num_hidden_layers: usize,
    #[serde(alias = "n_head")]
    pub num_attention_heads: usize,
    #[serde(alias = "n_inner")]
    pub intermediate_size: usize,
    #[serde(alias = "n_positions")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_type_vocab_size")]
    pub type_vocab_size: usize,
    #[serde(default = "default_layer_norm_eps")]
    pub layer_norm_epsilon: f64,
    #[serde(default = "default_rotary_emb_base")]
    pub rotary_emb_base: f32,
    #[serde(default = "default_rotary_emb_fraction")]
    pub rotary_emb_fraction: f32,
}

fn default_vocab_size() -> usize { 30528 }
fn default_type_vocab_size() -> usize { 2 }
fn default_layer_norm_eps() -> f64 { 1e-12 }
fn default_rotary_emb_base() -> f32 { 1000.0 }
fn default_rotary_emb_fraction() -> f32 { 1.0 }

impl NomicBertConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
    
    pub fn rotary_dim(&self) -> usize {
        (self.head_dim() as f32 * self.rotary_emb_fraction) as usize
    }
}

/// Rotary Position Embedding
struct RotaryEmbedding {
    cos: Tensor,
    sin: Tensor,
    dim: usize,
}

impl RotaryEmbedding {
    fn new(cfg: &NomicBertConfig, device: &Device) -> Result<Self> {
        let dim = cfg.rotary_dim();
        let max_seq_len = cfg.max_position_embeddings;
        let base = cfg.rotary_emb_base;
        
        // Compute inverse frequencies
        let inv_freq: Vec<f32> = (0..dim)
            .step_by(2)
            .map(|i| 1.0 / base.powf(i as f32 / dim as f32))
            .collect();
        let inv_freq = Tensor::from_vec(inv_freq, (dim / 2,), device)?;
        
        // Compute position indices
        let positions: Vec<f32> = (0..max_seq_len).map(|i| i as f32).collect();
        let positions = Tensor::from_vec(positions, (max_seq_len,), device)?;
        
        // Outer product: positions x inv_freq
        let freqs = positions.unsqueeze(1)?.matmul(&inv_freq.unsqueeze(0)?)?;
        
        // Duplicate for complex rotation
        let freqs = Tensor::cat(&[&freqs, &freqs], 1)?;
        
        let cos = freqs.cos()?;
        let sin = freqs.sin()?;
        
        Ok(Self { cos, sin, dim })
    }
    
    fn apply(&self, x: &Tensor, seq_len: usize) -> Result<Tensor> {
        let (_, _, _, head_dim) = x.dims4()?;
        
        // Only apply to first `dim` dimensions
        if self.dim < head_dim {
            let x_rot = x.narrow(D::Minus1, 0, self.dim)?;
            let x_pass = x.narrow(D::Minus1, self.dim, head_dim - self.dim)?;
            let x_rot = self.apply_rotary(&x_rot, seq_len)?;
            Tensor::cat(&[&x_rot, &x_pass], D::Minus1).map_err(Into::into)
        } else {
            self.apply_rotary(x, seq_len)
        }
    }
    
    fn apply_rotary(&self, x: &Tensor, seq_len: usize) -> Result<Tensor> {
        let cos = self.cos.i(..seq_len)?;
        let sin = self.sin.i(..seq_len)?;
        
        // Reshape cos/sin for broadcasting: (seq, dim) -> (1, 1, seq, dim)
        let cos = cos.unsqueeze(0)?.unsqueeze(0)?;
        let sin = sin.unsqueeze(0)?.unsqueeze(0)?;
        
        // Rotate: x * cos + rotate_half(x) * sin
        let x_rotated = self.rotate_half(x)?;
        let result = x.broadcast_mul(&cos)?.broadcast_add(&x_rotated.broadcast_mul(&sin)?)?;
        
        Ok(result)
    }
    
    fn rotate_half(&self, x: &Tensor) -> Result<Tensor> {
        let dim = x.dim(D::Minus1)?;
        let half = dim / 2;
        let x1 = x.narrow(D::Minus1, 0, half)?;
        let x2 = x.narrow(D::Minus1, half, half)?;
        Tensor::cat(&[&x2.neg()?, &x1], D::Minus1).map_err(Into::into)
    }
}

/// SwiGLU MLP
struct SwiGluMlp {
    fc11: Tensor,  // Gate projection
    fc12: Tensor,  // Value projection
    fc2: Tensor,   // Output projection
}

impl SwiGluMlp {
    fn new(vb: VarBuilder, cfg: &NomicBertConfig) -> Result<Self> {
        let fc11 = vb.get((cfg.intermediate_size, cfg.hidden_size), "fc11.weight")?;
        let fc12 = vb.get((cfg.intermediate_size, cfg.hidden_size), "fc12.weight")?;
        let fc2 = vb.get((cfg.hidden_size, cfg.intermediate_size), "fc2.weight")?;
        Ok(Self { fc11, fc12, fc2 })
    }
    
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // SwiGLU: fc2(SiLU(fc11(x)) * fc12(x))
        // Reshape to 2D for matmul, then back to 3D
        let (batch_size, seq_len, hidden_size) = x.dims3()?;
        let x_2d = x.reshape((batch_size * seq_len, hidden_size))?;
        
        let gate = x_2d.matmul(&self.fc11.t()?)?;
        let gate = candle_nn::ops::silu(&gate)?;
        let value = x_2d.matmul(&self.fc12.t()?)?;
        let hidden = gate.mul(&value)?;
        let out = hidden.matmul(&self.fc2.t()?)?;
        
        out.reshape((batch_size, seq_len, hidden_size)).map_err(Into::into)
    }
}

/// NomicBert Attention
struct NomicAttention {
    wqkv: Tensor,
    out_proj: Tensor,
    num_heads: usize,
    head_dim: usize,
    rotary: RotaryEmbedding,
}

impl NomicAttention {
    fn new(vb: VarBuilder, cfg: &NomicBertConfig, rotary: RotaryEmbedding) -> Result<Self> {
        let hidden_size = cfg.hidden_size;
        let wqkv = vb.get((3 * hidden_size, hidden_size), "Wqkv.weight")?;
        let out_proj = vb.get((hidden_size, hidden_size), "out_proj.weight")?;
        
        Ok(Self {
            wqkv,
            out_proj,
            num_heads: cfg.num_attention_heads,
            head_dim: cfg.head_dim(),
            rotary,
        })
    }
    
    fn forward(&self, x: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        let (batch_size, seq_len, hidden) = x.dims3()?;
        
        // Compute Q, K, V: reshape to 2D, matmul, reshape back
        let x_2d = x.reshape((batch_size * seq_len, hidden))?;
        let qkv = x_2d.matmul(&self.wqkv.t()?)?;
        let qkv = qkv.reshape((batch_size, seq_len, 3 * self.num_heads * self.head_dim))?;
        let qkv = qkv.reshape((batch_size, seq_len, 3, self.num_heads, self.head_dim))?;
        let qkv = qkv.permute((2, 0, 3, 1, 4))?;  // (3, batch, heads, seq, head_dim)
        
        let q = qkv.i(0)?;
        let k = qkv.i(1)?;
        let v = qkv.i(2)?;
        
        // Apply rotary embeddings
        let q = self.rotary.apply(&q, seq_len)?;
        let k = self.rotary.apply(&k, seq_len)?;
        
        // Scaled dot-product attention
        // Make tensors contiguous to avoid striding issues in matmul
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
                // mask=1 for real tokens, mask=0 for padding
                // We want to add 0 for real tokens, -inf for padding
                let ones = Tensor::ones_like(&mask)?;
                let inv_mask = ones.sub(&mask)?;  // 0 for real, 1 for padding
                let neg_inf = Tensor::new(&[-1e9f32], mask.device())?;
                let mask = inv_mask.broadcast_mul(&neg_inf)?;
                scores.broadcast_add(&mask)?
            }
            None => scores,
        };
        
        let attn_weights = candle_nn::ops::softmax(&scores, D::Minus1)?;
        // Make v contiguous to avoid striding issues
        let attn_output = attn_weights.matmul(&v.contiguous()?)?;
        
        // Reshape back
        let attn_output = attn_output.permute((0, 2, 1, 3))?;  // (batch, seq, heads, head_dim)
        let attn_output = attn_output.reshape((batch_size, seq_len, self.num_heads * self.head_dim))?;
        
        // Output projection: reshape to 2D, matmul, reshape back
        let attn_2d = attn_output.reshape((batch_size * seq_len, self.num_heads * self.head_dim))?;
        let out = attn_2d.matmul(&self.out_proj.t()?)?;
        out.reshape((batch_size, seq_len, self.num_heads * self.head_dim)).map_err(Into::into)
    }
}

/// NomicBert Encoder Layer
struct NomicBertLayer {
    norm1: MetalSafeLayerNorm,
    attn: NomicAttention,
    norm2: MetalSafeLayerNorm,
    mlp: SwiGluMlp,
}

impl NomicBertLayer {
    fn new(vb: VarBuilder, cfg: &NomicBertConfig, rotary: RotaryEmbedding) -> Result<Self> {
        let norm1 = metal_safe_layer_norm(cfg.hidden_size, cfg.layer_norm_epsilon, vb.pp("norm1"))?;
        let attn = NomicAttention::new(vb.pp("attn"), cfg, rotary)?;
        let norm2 = metal_safe_layer_norm(cfg.hidden_size, cfg.layer_norm_epsilon, vb.pp("norm2"))?;
        let mlp = SwiGluMlp::new(vb.pp("mlp"), cfg)?;
        
        Ok(Self { norm1, attn, norm2, mlp })
    }
    
    fn forward(&self, x: &Tensor, attention_mask: Option<&Tensor>) -> Result<Tensor> {
        // Pre-norm attention with residual
        let normed = self.norm1.forward(x)?;
        let attn_out = self.attn.forward(&normed, attention_mask)?;
        let x = (x + attn_out)?;
        
        // Pre-norm MLP with residual
        let normed = self.norm2.forward(&x)?;
        let mlp_out = self.mlp.forward(&normed)?;
        (x + mlp_out).map_err(Into::into)
    }
}

/// NomicBert Embeddings
struct NomicBertEmbeddings {
    word_embeddings: Embedding,
    token_type_embeddings: Embedding,
    layer_norm: MetalSafeLayerNorm,
}

impl NomicBertEmbeddings {
    fn new(vb: VarBuilder, cfg: &NomicBertConfig) -> Result<Self> {
        let word_embeddings = candle_nn::embedding(
            cfg.vocab_size,
            cfg.hidden_size,
            vb.pp("embeddings.word_embeddings"),
        )?;
        let token_type_embeddings = candle_nn::embedding(
            cfg.type_vocab_size,
            cfg.hidden_size,
            vb.pp("embeddings.token_type_embeddings"),
        )?;
        let layer_norm = metal_safe_layer_norm(cfg.hidden_size, cfg.layer_norm_epsilon, vb.pp("emb_ln"))?;
        
        Ok(Self {
            word_embeddings,
            token_type_embeddings,
            layer_norm,
        })
    }
    
    fn forward(&self, input_ids: &Tensor, token_type_ids: &Tensor) -> Result<Tensor> {
        let word_emb = self.word_embeddings.forward(input_ids)?;
        let type_emb = self.token_type_embeddings.forward(token_type_ids)?;
        let emb = (word_emb + type_emb)?;
        self.layer_norm.forward(&emb).map_err(Into::into)
    }
}

/// NomicBert Model
pub struct NomicBertModel {
    embeddings: NomicBertEmbeddings,
    layers: Vec<NomicBertLayer>,
}

impl NomicBertModel {
    pub fn load(vb: VarBuilder, cfg: &NomicBertConfig) -> Result<Self> {
        let embeddings = NomicBertEmbeddings::new(vb.clone(), cfg)?;
        
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        for i in 0..cfg.num_hidden_layers {
            let rotary = RotaryEmbedding::new(cfg, vb.device())?;
            let layer = NomicBertLayer::new(vb.pp(format!("encoder.layers.{}", i)), cfg, rotary)?;
            layers.push(layer);
        }
        
        Ok(Self { embeddings, layers })
    }
    
    pub fn forward(
        &self,
        input_ids: &Tensor,
        token_type_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let mut hidden = self.embeddings.forward(input_ids, token_type_ids)?;
        
        for layer in &self.layers {
            hidden = layer.forward(&hidden, attention_mask)?;
        }
        
        Ok(hidden)
    }
}
