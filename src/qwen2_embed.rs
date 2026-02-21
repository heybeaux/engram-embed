//! Qwen2-based embedding model (bidirectional, no causal mask)
//!
//! Adapted from candle-transformers Qwen2 implementation for embedding use.
//! Key modifications from the causal LM version:
//! - Removed causal attention mask → fully bidirectional attention
//! - Removed KV cache (not needed for encoding)
//! - Returns hidden states for mean pooling
//!
//! Used by KaLM-Embedding-V2 (HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v2)

use anyhow::Result;
use candle_core::{DType, Device, IndexOp, Module, Tensor, D};
use candle_nn::{Activation, Embedding, VarBuilder};
use serde::Deserialize;

use crate::metal_compat::MetalSafeLayerNorm;

// ============================================================================
// Config
// ============================================================================

#[derive(Debug, Clone, Deserialize)]
pub struct Qwen2EmbedConfig {
    pub vocab_size: usize,
    pub hidden_size: usize,
    pub intermediate_size: usize,
    pub num_hidden_layers: usize,
    pub num_attention_heads: usize,
    pub num_key_value_heads: usize,
    #[serde(default = "default_max_position_embeddings")]
    pub max_position_embeddings: usize,
    #[serde(default = "default_rope_theta")]
    pub rope_theta: f64,
    #[serde(default = "default_rms_norm_eps")]
    pub rms_norm_eps: f64,
    #[serde(default = "default_hidden_act")]
    pub hidden_act: Activation,
}

fn default_max_position_embeddings() -> usize { 32768 }
fn default_rope_theta() -> f64 { 1000000.0 }
fn default_rms_norm_eps() -> f64 { 1e-6 }
fn default_hidden_act() -> Activation { Activation::Silu }

impl Qwen2EmbedConfig {
    pub fn head_dim(&self) -> usize {
        self.hidden_size / self.num_attention_heads
    }
}

// ============================================================================
// RMSNorm (Metal-compatible)
// ============================================================================

#[derive(Debug, Clone)]
struct RmsNorm {
    weight: Tensor,
    eps: f64,
}

impl RmsNorm {
    fn load(size: usize, eps: f64, vb: VarBuilder) -> candle_core::Result<Self> {
        let weight = vb.get(size, "weight")?;
        Ok(Self { weight, eps })
    }
}

impl Module for RmsNorm {
    fn forward(&self, x: &Tensor) -> candle_core::Result<Tensor> {
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        let x = x.to_dtype(internal_dtype)?;
        let variance = x.sqr()?.mean_keepdim(D::Minus1)?;
        let x_normed = x.broadcast_div(&(variance + self.eps)?.sqrt()?)?;
        x_normed.to_dtype(x_dtype)?.broadcast_mul(&self.weight)
    }
}

// ============================================================================
// Rotary Embeddings
// ============================================================================

#[derive(Debug, Clone)]
struct RotaryEmbedding {
    sin: Tensor,
    cos: Tensor,
}

impl RotaryEmbedding {
    fn new(dtype: DType, cfg: &Qwen2EmbedConfig, dev: &Device) -> candle_core::Result<Self> {
        let dim = cfg.head_dim();
        let max_seq_len = cfg.max_position_embeddings;
        let inv_freq: Vec<_> = (0..dim)
            .step_by(2)
            .map(|i| 1f32 / cfg.rope_theta.powf(i as f64 / dim as f64) as f32)
            .collect();
        let inv_freq_len = inv_freq.len();
        let inv_freq = Tensor::from_vec(inv_freq, (1, inv_freq_len), dev)?.to_dtype(dtype)?;
        let t = Tensor::arange(0u32, max_seq_len as u32, dev)?
            .to_dtype(dtype)?
            .reshape((max_seq_len, 1))?;
        let freqs = t.matmul(&inv_freq)?;
        Ok(Self {
            sin: freqs.sin()?,
            cos: freqs.cos()?,
        })
    }

    fn apply_rotary_emb(
        &self,
        q: &Tensor,
        k: &Tensor,
    ) -> candle_core::Result<(Tensor, Tensor)> {
        let (_b_sz, _h, seq_len, _n_embd) = q.dims4()?;
        let cos = self.cos.narrow(0, 0, seq_len)?;
        let sin = self.sin.narrow(0, 0, seq_len)?;
        let q_embed = candle_nn::rotary_emb::rope(&q.contiguous()?, &cos, &sin)?;
        let k_embed = candle_nn::rotary_emb::rope(&k.contiguous()?, &cos, &sin)?;
        Ok((q_embed, k_embed))
    }
}

// ============================================================================
// MLP
// ============================================================================

#[derive(Debug, Clone)]
struct MLP {
    gate_proj: candle_nn::Linear,
    up_proj: candle_nn::Linear,
    down_proj: candle_nn::Linear,
    act_fn: Activation,
}

impl MLP {
    fn load(cfg: &Qwen2EmbedConfig, vb: VarBuilder) -> candle_core::Result<Self> {
        let h = cfg.hidden_size;
        let i = cfg.intermediate_size;
        let gate_proj = candle_nn::linear_no_bias(h, i, vb.pp("gate_proj"))?;
        let up_proj = candle_nn::linear_no_bias(h, i, vb.pp("up_proj"))?;
        let down_proj = candle_nn::linear_no_bias(i, h, vb.pp("down_proj"))?;
        Ok(Self {
            gate_proj,
            up_proj,
            down_proj,
            act_fn: cfg.hidden_act,
        })
    }
}

impl Module for MLP {
    fn forward(&self, xs: &Tensor) -> candle_core::Result<Tensor> {
        let lhs = xs.apply(&self.gate_proj)?.apply(&self.act_fn)?;
        let rhs = xs.apply(&self.up_proj)?;
        (lhs * rhs)?.apply(&self.down_proj)
    }
}

// ============================================================================
// Bidirectional Attention (no causal mask, no KV cache)
// ============================================================================

#[derive(Debug, Clone)]
struct BidirectionalAttention {
    q_proj: candle_nn::Linear,
    k_proj: candle_nn::Linear,
    v_proj: candle_nn::Linear,
    o_proj: candle_nn::Linear,
    num_heads: usize,
    num_kv_heads: usize,
    num_kv_groups: usize,
    head_dim: usize,
    hidden_size: usize,
    rotary_emb: RotaryEmbedding,
}

impl BidirectionalAttention {
    fn load(
        rotary_emb: RotaryEmbedding,
        cfg: &Qwen2EmbedConfig,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let h = cfg.hidden_size;
        let num_heads = cfg.num_attention_heads;
        let num_kv_heads = cfg.num_key_value_heads;
        let head_dim = cfg.head_dim();
        // Qwen2 uses bias on q_proj and k_proj but not on o_proj
        let q_proj = candle_nn::linear(h, num_heads * head_dim, vb.pp("q_proj"))?;
        let k_proj = candle_nn::linear(h, num_kv_heads * head_dim, vb.pp("k_proj"))?;
        let v_proj = candle_nn::linear(h, num_kv_heads * head_dim, vb.pp("v_proj"))?;
        let o_proj = candle_nn::linear_no_bias(num_heads * head_dim, h, vb.pp("o_proj"))?;
        Ok(Self {
            q_proj,
            k_proj,
            v_proj,
            o_proj,
            num_heads,
            num_kv_heads,
            num_kv_groups: num_heads / num_kv_heads,
            head_dim,
            hidden_size: h,
            rotary_emb,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        let (b_sz, seq_len, _) = xs.dims3()?;

        let q = self.q_proj.forward(xs)?;
        let k = self.k_proj.forward(xs)?;
        let v = self.v_proj.forward(xs)?;

        let q = q
            .reshape((b_sz, seq_len, self.num_heads, self.head_dim))?
            .transpose(1, 2)?;
        let k = k
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;
        let v = v
            .reshape((b_sz, seq_len, self.num_kv_heads, self.head_dim))?
            .transpose(1, 2)?;

        // Apply rotary embeddings
        let (q, k) = self.rotary_emb.apply_rotary_emb(&q, &k)?;

        // Repeat KV heads for GQA
        let k = repeat_kv(k, self.num_kv_groups)?.contiguous()?;
        let v = repeat_kv(v, self.num_kv_groups)?.contiguous()?;

        // Scaled dot-product attention — NO causal mask (bidirectional)
        let scale = 1f64 / f64::sqrt(self.head_dim as f64);
        let attn_weights = (q.matmul(&k.transpose(2, 3)?)? * scale)?;

        // Only apply padding mask if provided (not causal mask)
        let attn_weights = match attention_mask {
            Some(mask) => attn_weights.broadcast_add(mask)?,
            None => attn_weights,
        };

        let attn_weights = candle_nn::ops::softmax_last_dim(&attn_weights)?;
        let attn_output = attn_weights.matmul(&v)?;

        attn_output
            .transpose(1, 2)?
            .reshape((b_sz, seq_len, self.hidden_size))?
            .apply(&self.o_proj)
    }
}

/// Repeat KV heads for grouped query attention
fn repeat_kv(xs: Tensor, n_rep: usize) -> candle_core::Result<Tensor> {
    if n_rep == 1 {
        Ok(xs)
    } else {
        let (b_sz, num_kv_heads, seq_len, head_dim) = xs.dims4()?;
        xs.unsqueeze(2)?
            .expand((b_sz, num_kv_heads, n_rep, seq_len, head_dim))?
            .reshape((b_sz, num_kv_heads * n_rep, seq_len, head_dim))
    }
}

// ============================================================================
// Transformer Layer
// ============================================================================

#[derive(Debug, Clone)]
struct Qwen2EmbedLayer {
    self_attn: BidirectionalAttention,
    mlp: MLP,
    input_layernorm: RmsNorm,
    post_attention_layernorm: RmsNorm,
}

impl Qwen2EmbedLayer {
    fn load(
        rotary_emb: RotaryEmbedding,
        cfg: &Qwen2EmbedConfig,
        vb: VarBuilder,
    ) -> candle_core::Result<Self> {
        let self_attn = BidirectionalAttention::load(rotary_emb, cfg, vb.pp("self_attn"))?;
        let mlp = MLP::load(cfg, vb.pp("mlp"))?;
        let input_layernorm =
            RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb.pp("input_layernorm"))?;
        let post_attention_layernorm = RmsNorm::load(
            cfg.hidden_size,
            cfg.rms_norm_eps,
            vb.pp("post_attention_layernorm"),
        )?;
        Ok(Self {
            self_attn,
            mlp,
            input_layernorm,
            post_attention_layernorm,
        })
    }

    fn forward(
        &self,
        xs: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        let residual = xs;
        let xs = self.input_layernorm.forward(xs)?;
        let xs = self.self_attn.forward(&xs, attention_mask)?;
        let xs = (xs + residual)?;
        let residual = &xs;
        let xs = xs
            .apply(&self.post_attention_layernorm)?
            .apply(&self.mlp)?;
        residual + xs
    }
}

// ============================================================================
// Qwen2 Embedding Model
// ============================================================================

#[derive(Debug, Clone)]
pub struct Qwen2EmbedModel {
    embed_tokens: Embedding,
    layers: Vec<Qwen2EmbedLayer>,
    norm: RmsNorm,
    device: Device,
    dtype: DType,
}

impl Qwen2EmbedModel {
    pub fn load(vb: VarBuilder, cfg: &Qwen2EmbedConfig) -> candle_core::Result<Self> {
        let vb_m = vb.pp("model");
        let embed_tokens =
            candle_nn::embedding(cfg.vocab_size, cfg.hidden_size, vb_m.pp("embed_tokens"))?;
        let rotary_emb = RotaryEmbedding::new(vb.dtype(), cfg, vb_m.device())?;
        let mut layers = Vec::with_capacity(cfg.num_hidden_layers);
        let vb_l = vb_m.pp("layers");
        for layer_idx in 0..cfg.num_hidden_layers {
            let layer = Qwen2EmbedLayer::load(rotary_emb.clone(), cfg, vb_l.pp(layer_idx))?;
            layers.push(layer);
        }
        let norm = RmsNorm::load(cfg.hidden_size, cfg.rms_norm_eps, vb_m.pp("norm"))?;
        Ok(Self {
            embed_tokens,
            layers,
            norm,
            device: vb.device().clone(),
            dtype: vb.dtype(),
        })
    }

    /// Forward pass returning hidden states (for mean pooling)
    ///
    /// `attention_mask`: (batch, seq_len) with 1 for real tokens, 0 for padding
    pub fn forward(
        &self,
        input_ids: &Tensor,
        _token_type_ids: &Tensor,
        attention_mask: Option<&Tensor>,
    ) -> candle_core::Result<Tensor> {
        let mut xs = self.embed_tokens.forward(input_ids)?;

        // Convert padding mask (batch, seq_len) → (batch, 1, 1, seq_len) additive mask
        let attn_mask = match attention_mask {
            Some(mask) => {
                let (b_sz, seq_len) = mask.dims2()?;
                // Expand to (b, 1, seq_len, seq_len) for broadcast with attention weights
                // We only mask padding positions (columns), not causal positions
                let mask = mask.unsqueeze(1)?.unsqueeze(2)?; // (b, 1, 1, seq_len)
                let mask = mask.expand((b_sz, 1, seq_len, seq_len))?;
                let on_true = mask.zeros_like()?.to_dtype(self.dtype)?;
                let on_false = Tensor::new(f32::NEG_INFINITY, &self.device)?
                    .broadcast_as(mask.shape())?
                    .to_dtype(self.dtype)?;
                Some(mask.where_cond(&on_true, &on_false)?)
            }
            None => None,
        };

        for layer in &self.layers {
            xs = layer.forward(&xs, attn_mask.as_ref())?;
        }

        xs.apply(&self.norm)
    }
}
