//! Metal compatibility layer for operations that don't have native Metal implementations
//!
//! Candle's Metal backend is missing certain CustomOp implementations (notably layer_norm).
//! This module provides fallback implementations using basic tensor operations that ARE
//! supported on Metal.

use candle_core::{DType, Device, Module, Result, Tensor, D};
use candle_nn::VarBuilder;

/// A Metal-compatible LayerNorm that always uses the "slow path" implementation.
/// 
/// The standard candle_nn::LayerNorm tries a CustomOp fast path first, which fails on Metal.
/// This implementation uses only basic tensor operations (sum, broadcast, sqrt, etc.)
/// that are fully supported on Metal.
#[derive(Clone, Debug)]
pub struct MetalSafeLayerNorm {
    weight: Tensor,
    bias: Option<Tensor>,
    eps: f64,
}

impl MetalSafeLayerNorm {
    /// Create a new MetalSafeLayerNorm with weight and bias
    pub fn new(weight: Tensor, bias: Tensor, eps: f64) -> Self {
        Self {
            weight,
            bias: Some(bias),
            eps,
        }
    }

    /// Create a new MetalSafeLayerNorm without bias (RMSNorm-like)
    pub fn new_no_bias(weight: Tensor, eps: f64) -> Self {
        Self {
            weight,
            bias: None,
            eps,
        }
    }

    /// Load from VarBuilder (compatible with candle_nn::layer_norm signature)
    pub fn load(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get(size, "weight")?;
        let bias = vb.get(size, "bias").ok();
        Ok(Self { weight, bias, eps })
    }
}

impl Module for MetalSafeLayerNorm {
    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // Always use the slow path that works on Metal
        // This is the same logic as candle_nn::LayerNorm's fallback path
        let x_dtype = x.dtype();
        let internal_dtype = match x_dtype {
            DType::F16 | DType::BF16 => DType::F32,
            d => d,
        };
        
        let hidden_size = x.dim(D::Minus1)?;
        let x = x.to_dtype(internal_dtype)?;
        
        // Compute mean and center
        let mean_x = (x.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_centered = x.broadcast_sub(&mean_x)?;
        
        // Compute variance and normalize
        let var_x = (x_centered.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
        let x_normed = x_centered.broadcast_div(&(var_x + self.eps)?.sqrt()?)?;
        
        // Apply scale and shift
        let x = x_normed.to_dtype(x_dtype)?.broadcast_mul(&self.weight)?;
        match &self.bias {
            None => Ok(x),
            Some(bias) => x.broadcast_add(bias),
        }
    }
}

/// Helper function to create a MetalSafeLayerNorm from VarBuilder
pub fn metal_safe_layer_norm(size: usize, eps: f64, vb: VarBuilder) -> Result<MetalSafeLayerNorm> {
    MetalSafeLayerNorm::load(size, eps, vb)
}

/// Check if a device is Metal
pub fn is_metal_device(device: &Device) -> bool {
    matches!(device, Device::Metal(_))
}

/// Select the best available device with Metal support
/// 
/// Returns Metal device if available and EMBED_DEVICE env var is not set to "cpu".
/// Falls back to CPU if Metal is not available or explicitly disabled.
pub fn select_device_with_metal() -> Result<Device> {
    // Check environment variable
    let force_cpu = std::env::var("EMBED_DEVICE")
        .map(|v| v.to_lowercase() == "cpu")
        .unwrap_or(false);
    
    if force_cpu {
        tracing::info!("EMBED_DEVICE=cpu, using CPU");
        return Ok(Device::Cpu);
    }
    
    // Try Metal first on macOS
    #[cfg(target_os = "macos")]
    {
        match Device::new_metal(0) {
            Ok(device) => {
                tracing::info!("🔩 Using Metal GPU acceleration");
                return Ok(device);
            }
            Err(e) => {
                tracing::warn!("Metal not available: {}, falling back to CPU", e);
            }
        }
    }
    
    Ok(Device::Cpu)
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_core::Tensor;

    #[test]
    fn test_metal_safe_layer_norm_cpu() -> Result<()> {
        let device = Device::Cpu;
        let hidden_size = 768;
        
        let weight = Tensor::ones((hidden_size,), DType::F32, &device)?;
        let bias = Tensor::zeros((hidden_size,), DType::F32, &device)?;
        let layer_norm = MetalSafeLayerNorm::new(weight, bias, 1e-5);
        
        let x = Tensor::randn(0f32, 1.0, (2, 4, hidden_size), &device)?;
        let output = layer_norm.forward(&x)?;
        
        assert_eq!(output.dims(), &[2, 4, hidden_size]);
        Ok(())
    }

    #[test]
    fn test_metal_safe_layer_norm_values() -> Result<()> {
        let device = Device::Cpu;
        
        // Simple test case: [1, 2, 3] normalized should have mean=0, var=1
        let weight = Tensor::ones((3,), DType::F32, &device)?;
        let bias = Tensor::zeros((3,), DType::F32, &device)?;
        let layer_norm = MetalSafeLayerNorm::new(weight, bias, 1e-5);
        
        let x = Tensor::new(&[[1f32, 2., 3.]], &device)?;
        let output = layer_norm.forward(&x)?;
        let output_vec = output.to_vec2::<f32>()?;
        
        // Expected: approximately [-1.2247, 0, 1.2247]
        let expected = [-1.2247, 0.0, 1.2247];
        for (i, (got, exp)) in output_vec[0].iter().zip(expected.iter()).enumerate() {
            assert!(
                (got - exp).abs() < 0.01,
                "Mismatch at {}: got {}, expected {}",
                i, got, exp
            );
        }
        
        Ok(())
    }
}
