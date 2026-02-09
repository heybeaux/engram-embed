//! Metal GPU test - check if BERT ops work on Metal backend
//! 
//! Run with: cargo run --example metal_test --release

use anyhow::Result;
use candle_core::{DType, Device, Tensor, D};

fn main() -> Result<()> {
    test_metal_ops()
}

/// Test basic Metal operations used by BERT
fn test_metal_ops() -> Result<()> {
    println!("\n=== Metal GPU Test for BERT Operations ===\n");
    
    let device = match Device::new_metal(0) {
        Ok(d) => {
            println!("✅ Metal device created successfully");
            d
        }
        Err(e) => {
            println!("❌ Failed to create Metal device: {}", e);
            println!("   Falling back to CPU for comparison...");
            Device::Cpu
        }
    };
    
    let is_metal = matches!(&device, Device::Metal(_));
    
    // Test operations used by BERT models
    test_basic_ops(&device, is_metal)?;
    test_layer_norm_slow_path(&device, is_metal)?;
    test_layer_norm_fast_path(&device, is_metal)?;
    test_attention_ops(&device, is_metal)?;
    
    println!("\n=== Metal Test Complete ===\n");
    
    if is_metal {
        println!("📊 SUMMARY: Metal GPU is available and basic ops work.");
        println!("   Layer norm fast path may fail but slow path works.");
        println!("   Recommendation: Use slow path for Metal compatibility.");
    }
    
    Ok(())
}

fn test_basic_ops(device: &Device, is_metal: bool) -> Result<()> {
    let device_name = if is_metal { "Metal" } else { "CPU" };
    println!("\n--- Basic Operations ({}) ---", device_name);
    
    // Tensor creation
    let a = Tensor::new(&[1.0f32, 2.0, 3.0, 4.0], device)?;
    println!("✅ Tensor creation");
    
    // Reshape
    let b = a.reshape((2, 2))?;
    println!("✅ Reshape");
    
    // Matmul
    let c = b.matmul(&b.t()?)?;
    println!("✅ Matmul");
    
    // Elementwise ops
    let _ = c.sqr()?;
    println!("✅ Square");
    
    let _ = c.sqrt()?;
    println!("✅ Sqrt");
    
    // Softmax
    let _ = candle_nn::ops::softmax(&c, 1)?;
    println!("✅ Softmax");
    
    // Sum with keepdim
    let _ = c.sum_keepdim(1)?;
    println!("✅ Sum keepdim");
    
    // Broadcast operations
    let ones = Tensor::ones((2, 1), DType::F32, device)?;
    let _ = c.broadcast_mul(&ones)?;
    println!("✅ Broadcast mul");
    
    let _ = c.broadcast_add(&ones)?;
    println!("✅ Broadcast add");
    
    let _ = c.broadcast_sub(&ones)?;
    println!("✅ Broadcast sub");
    
    let _ = c.broadcast_div(&ones)?;
    println!("✅ Broadcast div");
    
    Ok(())
}

fn test_layer_norm_slow_path(device: &Device, is_metal: bool) -> Result<()> {
    let device_name = if is_metal { "Metal" } else { "CPU" };
    println!("\n--- Layer Norm (Slow Path - basic ops) [{}] ---", device_name);
    
    // Simulate layer norm using basic operations (what the slow path does)
    let hidden_size = 768usize;
    let batch_size = 2usize;
    let seq_len = 4usize;
    let eps = 1e-5f64;
    
    let x = Tensor::randn(0f32, 1.0, (batch_size, seq_len, hidden_size), device)?;
    let weight = Tensor::ones((hidden_size,), DType::F32, device)?;
    let bias = Tensor::zeros((hidden_size,), DType::F32, device)?;
    
    // Manual layer norm (slow path implementation)
    let mean_x = (x.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
    println!("✅ Mean calculation");
    
    let x_centered = x.broadcast_sub(&mean_x)?;
    println!("✅ Centering");
    
    let var_x = (x_centered.sqr()?.sum_keepdim(D::Minus1)? / hidden_size as f64)?;
    println!("✅ Variance calculation");
    
    let std_x = (var_x + eps)?.sqrt()?;
    println!("✅ Std calculation");
    
    let x_norm = x_centered.broadcast_div(&std_x)?;
    println!("✅ Normalization");
    
    let output = x_norm.broadcast_mul(&weight)?.broadcast_add(&bias)?;
    println!("✅ Scale and shift");
    
    // Verify output shape
    let shape = output.dims();
    assert_eq!(shape, &[batch_size, seq_len, hidden_size]);
    println!("✅ Output shape correct: {:?}", shape);
    
    // Transfer back to CPU to verify
    let _cpu_output = output.to_device(&Device::Cpu)?.to_vec3::<f32>()?;
    println!("✅ Layer norm slow path WORKS on {}!", device_name);
    
    Ok(())
}

fn test_layer_norm_fast_path(device: &Device, is_metal: bool) -> Result<()> {
    let device_name = if is_metal { "Metal" } else { "CPU" };
    println!("\n--- Layer Norm (Fast Path - CustomOp) [{}] ---", device_name);
    
    let hidden_size = 768usize;
    let batch_size = 2usize;
    let seq_len = 4usize;
    
    let x = Tensor::randn(0f32, 1.0, (batch_size, seq_len, hidden_size), device)?;
    let weight = Tensor::ones((hidden_size,), DType::F32, device)?;
    let bias = Tensor::zeros((hidden_size,), DType::F32, device)?;
    
    // Try the fast path (may fail on Metal)
    match candle_nn::ops::layer_norm(&x, &weight, &bias, 1e-5) {
        Ok(output) => {
            let _ = output.to_device(&Device::Cpu)?;
            println!("✅ Layer norm fast path WORKS on {}!", device_name);
        }
        Err(e) => {
            println!("❌ Layer norm fast path FAILS on {}: {}", device_name, e);
            if is_metal {
                println!("   (This is expected - will use slow path fallback)");
            }
        }
    }
    
    Ok(())
}

fn test_attention_ops(device: &Device, is_metal: bool) -> Result<()> {
    let device_name = if is_metal { "Metal" } else { "CPU" };
    println!("\n--- Attention Operations [{}] ---", device_name);
    
    let batch = 2usize;
    let heads = 12usize;
    let seq_len = 32usize;
    let head_dim = 64usize;
    
    // Q, K, V
    let q = Tensor::randn(0f32, 0.1, (batch, heads, seq_len, head_dim), device)?;
    let k = Tensor::randn(0f32, 0.1, (batch, heads, seq_len, head_dim), device)?;
    let v = Tensor::randn(0f32, 0.1, (batch, heads, seq_len, head_dim), device)?;
    
    // QK^T
    let scale = 1.0 / (head_dim as f64).sqrt();
    let k_t = k.transpose(2, 3)?;
    println!("✅ Transpose");
    
    let attn_weights = q.matmul(&k_t)?;
    println!("✅ Q @ K^T matmul");
    
    let attn_weights = (attn_weights * scale)?;
    println!("✅ Scale");
    
    let attn_weights = candle_nn::ops::softmax(&attn_weights, D::Minus1)?;
    println!("✅ Softmax on attention");
    
    let output = attn_weights.matmul(&v)?;
    println!("✅ Attention @ V matmul");
    
    // to_vec4 doesn't exist, use flatten + to_vec1
    let _ = output.to_device(&Device::Cpu)?.flatten_all()?.to_vec1::<f32>()?;
    println!("✅ Attention ops WORK on {}!", device_name);
    
    Ok(())
}
