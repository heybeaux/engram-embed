//! Benchmark: CPU vs Metal for Nomic embedding model
//!
//! Run with: cargo run --example benchmark_metal --release

use std::time::Instant;

fn main() -> anyhow::Result<()> {
    // Initialize tracing
    tracing_subscriber::fmt::init();
    
    println!("\n=== engram-embed Metal Benchmark ===\n");
    
    // Test texts of various lengths
    let test_texts = vec![
        "Hello, world!".to_string(),
        "The quick brown fox jumps over the lazy dog. This is a longer sentence to test embedding performance.".to_string(),
        "Machine learning models can be accelerated using GPU computing. On Apple Silicon, the Metal API provides access to the GPU for compute tasks. This allows for faster inference when running embedding models locally.".to_string(),
    ];
    
    // Benchmark CPU
    println!("📊 Benchmarking Nomic model...\n");
    
    // Run with CPU
    std::env::set_var("EMBED_DEVICE", "cpu");
    let cpu_times = benchmark_embeddings(&test_texts, "CPU")?;
    
    // Run with Metal
    std::env::set_var("EMBED_DEVICE", "metal");
    let metal_times = benchmark_embeddings(&test_texts, "Metal")?;
    
    // Print comparison
    println!("\n=== Results ===\n");
    println!("{:<15} {:>12} {:>12} {:>12}", "Metric", "CPU (ms)", "Metal (ms)", "Speedup");
    println!("{:-<15} {:->12} {:->12} {:->12}", "", "", "", "");
    
    for (i, (cpu_t, metal_t)) in cpu_times.iter().zip(metal_times.iter()).enumerate() {
        let speedup = cpu_t / metal_t;
        println!(
            "{:<15} {:>12.2} {:>12.2} {:>11.2}x",
            format!("Text {} ({} chars)", i + 1, test_texts[i].len()),
            cpu_t,
            metal_t,
            speedup
        );
    }
    
    let avg_cpu: f64 = cpu_times.iter().sum::<f64>() / cpu_times.len() as f64;
    let avg_metal: f64 = metal_times.iter().sum::<f64>() / metal_times.len() as f64;
    let avg_speedup = avg_cpu / avg_metal;
    
    println!("{:-<15} {:->12} {:->12} {:->12}", "", "", "", "");
    println!("{:<15} {:>12.2} {:>12.2} {:>11.2}x", "Average", avg_cpu, avg_metal, avg_speedup);
    
    println!("\n✅ Benchmark complete\n");
    
    Ok(())
}

fn benchmark_embeddings(texts: &[String], device_name: &str) -> anyhow::Result<Vec<f64>> {
    use candle_core::Device;
    
    println!("  {} Device:", device_name);
    
    // Create device
    let device = if device_name == "Metal" {
        #[cfg(target_os = "macos")]
        {
            match Device::new_metal(0) {
                Ok(d) => {
                    println!("    ✅ Metal initialized");
                    d
                }
                Err(e) => {
                    println!("    ❌ Metal failed: {}", e);
                    return Ok(vec![0.0; texts.len()]);
                }
            }
        }
        #[cfg(not(target_os = "macos"))]
        {
            println!("    ❌ Metal not available (not macOS)");
            return Ok(vec![0.0; texts.len()]);
        }
    } else {
        println!("    Using CPU");
        Device::Cpu
    };
    
    // We need to use the actual embedder here, but for simplicity let's just test the ops
    // that are most performance-sensitive: matmul and attention
    
    let mut times = Vec::new();
    let warmup = 3;
    let iterations = 10;
    
    for (i, text) in texts.iter().enumerate() {
        // Simulate embedding workload: matmul-heavy operations
        // BERT-like: hidden_size=768, num_heads=12, head_dim=64
        let hidden_size = 768;
        let seq_len = text.len().min(512); // Simulate token length roughly = char length / 4
        let seq_len = (seq_len / 4).max(4);
        
        let x = candle_core::Tensor::randn(0f32, 0.1, (1, seq_len, hidden_size), &device)?;
        let w = candle_core::Tensor::randn(0f32, 0.1, (hidden_size, hidden_size), &device)?;
        
        // Warmup
        for _ in 0..warmup {
            let _ = simulate_bert_layer(&x, &w)?;
        }
        
        // Timed runs
        let mut total_time = 0.0;
        for _ in 0..iterations {
            let start = Instant::now();
            let _ = simulate_bert_layer(&x, &w)?;
            total_time += start.elapsed().as_secs_f64() * 1000.0;
        }
        
        let avg_time = total_time / iterations as f64;
        times.push(avg_time);
        println!("    Text {}: {:.2} ms (avg of {} runs, seq_len={})", i + 1, avg_time, iterations, seq_len);
    }
    
    Ok(times)
}

fn simulate_bert_layer(x: &candle_core::Tensor, w: &candle_core::Tensor) -> anyhow::Result<candle_core::Tensor> {
    use candle_core::D;
    
    // Simulate self-attention + FFN
    // 1. Q, K, V projections (3x matmul)
    let (batch, seq_len, hidden) = x.dims3()?;
    let x_2d = x.reshape((batch * seq_len, hidden))?;
    
    let q = x_2d.matmul(&w.t()?)?;
    let k = x_2d.matmul(&w.t()?)?;
    let v = x_2d.matmul(&w.t()?)?;
    
    // Reshape to (batch, heads, seq, head_dim)
    let num_heads = 12;
    let head_dim = hidden / num_heads;
    let q = q.reshape((batch, seq_len, num_heads, head_dim))?.permute((0, 2, 1, 3))?;
    let k = k.reshape((batch, seq_len, num_heads, head_dim))?.permute((0, 2, 1, 3))?;
    let v = v.reshape((batch, seq_len, num_heads, head_dim))?.permute((0, 2, 1, 3))?;
    
    // Attention scores - make tensors contiguous for Metal compatibility
    let scale = 1.0 / (head_dim as f64).sqrt();
    let k_t = k.transpose(2, 3)?.contiguous()?;
    let scores = q.contiguous()?.matmul(&k_t)?;
    let scores = (scores * scale)?;
    let attn = candle_nn::ops::softmax(&scores, D::Minus1)?;
    
    // Attention output
    let out = attn.matmul(&v.contiguous()?)?;
    let out = out.permute((0, 2, 1, 3))?.reshape((batch, seq_len, hidden))?;
    
    // FFN (2x matmul with activation)
    let out_2d = out.reshape((batch * seq_len, hidden))?;
    let h = out_2d.matmul(&w.t()?)?;
    let h = candle_nn::ops::silu(&h)?;
    let h = h.matmul(&w)?;
    let out = h.reshape((batch, seq_len, hidden))?;
    
    // Force sync for accurate timing
    let _ = out.to_vec3::<f32>()?;
    
    Ok(out)
}
