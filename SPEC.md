# engram-embed

**Local embedding server in Rust — drop-in replacement for OpenAI embeddings**

## Why

| OpenAI Embeddings | engram-embed |
|-------------------|--------------|
| $0.0001/1K tokens | Free (local compute) |
| Network latency | Sub-millisecond |
| Rate limits | None |
| Privacy concerns | Data never leaves machine |
| Vendor lock-in | Self-contained |

At scale (1M+ embeddings/day): **$100+/day → $0/day**

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                   engram-embed                       │
│                                                      │
│  ┌──────────┐    ┌──────────┐    ┌───────────────┐ │
│  │  Axum    │───▶│  Candle  │───▶│ bge-base-en   │ │
│  │  Server  │    │  Runtime │    │ v1.5 (440MB)  │ │
│  └──────────┘    └──────────┘    └───────────────┘ │
│       │                                ▲            │
│       │         ┌──────────┐          │            │
│       └────────▶│  Metal   │──────────┘            │
│                 │  Accel   │ (M1/M2/M3)            │
│                 └──────────┘                        │
└─────────────────────────────────────────────────────┘
         ▲
         │ POST /v1/embeddings (OpenAI-compatible)
         │
    ┌────┴────┐
    │ Engram  │
    └─────────┘
```

## API (OpenAI-compatible)

```bash
# Request (same as OpenAI)
curl http://localhost:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{
    "input": "Remember this important fact",
    "model": "bge-base-en-v1.5"
  }'

# Response (same shape as OpenAI)
{
  "object": "list",
  "data": [{
    "object": "embedding",
    "embedding": [0.123, -0.456, ...],  # 768 dimensions
    "index": 0
  }],
  "model": "bge-base-en-v1.5",
  "usage": {
    "prompt_tokens": 5,
    "total_tokens": 5
  }
}
```

## Tech Stack

| Component | Choice | Why |
|-----------|--------|-----|
| Language | Rust | Performance, single binary, learning goal |
| HTTP | Axum | Async, ergonomic, Tokio-based |
| ML Runtime | Candle | HuggingFace's Rust ML, Metal support |
| Model | all-MiniLM-L6-v2 | 384-dim, 23MB, great quality/speed |
| Tokenizer | tokenizers (HF) | Rust-native, fast |

## Model Options

| Model | Dimensions | Size | Speed | Quality |
|-------|------------|------|-------|---------|
| all-MiniLM-L6-v2 | 384 | 23MB | ⚡⚡⚡ | Good |
| bge-small-en-v1.5 | 384 | 130MB | ⚡⚡⚡ | Great |
| **bge-base-en-v1.5** | **768** | **440MB** | **⚡⚡** | **Excellent** ← SELECTED |
| bge-large-en-v1.5 | 1024 | 1.3GB | ⚡ | Best |
| nomic-embed-text-v1.5 | 768 | 550MB | ⚡ | Excellent |

**Selected:** bge-base-en-v1.5 — top-tier open-source embeddings, BERT architecture (native Candle support).

## MVP Scope

### Phase 1: Hello Embedding (Week 1)
- [ ] Rust project setup with Cargo
- [ ] Load model with Candle
- [ ] Tokenize input text
- [ ] Generate embedding vector
- [ ] CLI that embeds a string

### Phase 2: HTTP Server (Week 2)
- [ ] Axum server on port 8080
- [ ] POST /v1/embeddings endpoint
- [ ] OpenAI-compatible request/response
- [ ] Batch embedding support
- [ ] Health check endpoint

### Phase 3: Integration (Week 3)
- [ ] Engram config for local embeddings
- [ ] Fallback to OpenAI if local unavailable
- [ ] Benchmarks (latency, throughput)
- [ ] Metal acceleration verification

### Phase 4: Production Ready
- [ ] Model caching / warm start
- [ ] Graceful shutdown
- [ ] Metrics endpoint
- [ ] Docker build (optional)
- [ ] LaunchAgent for macOS

## Engram Integration

```typescript
// engram/.env
EMBEDDING_PROVIDER=local           # or 'openai'
EMBEDDING_LOCAL_URL=http://127.0.0.1:8080
EMBEDDING_MODEL=bge-base-en-v1.5
EMBEDDING_DIMENSIONS=768           # Must match Pinecone index

// Falls back to OpenAI if local is down
```

**Note:** Switching to 768-dim embeddings requires a new Pinecone index. See migration section.

## Performance Expectations

On M2 Mac (CPU):
- Single embedding: ~5ms
- Batch of 100: ~200ms
- Throughput: ~500 embeddings/sec

With Metal acceleration:
- Single embedding: ~1ms
- Batch of 100: ~50ms
- Throughput: ~2000 embeddings/sec

## Directory Structure

```
engram-embed/
├── Cargo.toml
├── src/
│   ├── main.rs          # Entry point
│   ├── server.rs        # Axum routes
│   ├── embedder.rs      # Candle model loading + inference
│   ├── tokenizer.rs     # Text tokenization
│   └── types.rs         # OpenAI-compatible types
├── models/              # Downloaded model files
│   └── all-MiniLM-L6-v2/
├── tests/
│   └── integration.rs
└── README.md
```

## Getting Started

```bash
# Prerequisites
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
rustup default stable

# Create project
cargo new engram-embed
cd engram-embed

# Add dependencies
cargo add axum tokio --features full
cargo add candle-core candle-nn candle-transformers
cargo add tokenizers
cargo add serde serde_json --features derive
```

## Resources

- [Candle Examples](https://github.com/huggingface/candle/tree/main/candle-examples)
- [Axum Guide](https://docs.rs/axum/latest/axum/)
- [Sentence Transformers Models](https://huggingface.co/sentence-transformers)
- [The Rust Book](https://doc.rust-lang.org/book/)

---

*Created: 2026-02-05*
*Status: Specification*
