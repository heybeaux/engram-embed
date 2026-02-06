# engram-embed

Local embedding server in Rust — drop-in replacement for OpenAI's embeddings API.

**Ecosystem:** [Core API](https://github.com/heybeaux/engram) • [Dashboard](https://github.com/heybeaux/engram-dashboard) • [Local Embeddings](https://github.com/heybeaux/engram-embed)

## Why?

| OpenAI | engram-embed |
|--------|--------------|
| $0.0001/1K tokens | Free |
| ~100ms latency | ~10ms latency |
| Rate limits | Unlimited |
| Data sent to cloud | Fully local |

## Quick Start

```bash
# Install Rust (if needed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Build and run
cargo run --release

# Test it
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, world!"}'
```

## Models

Three embedding models available:

| Model | Dimensions | Max Tokens | Use Case |
|-------|------------|------------|----------|
| `bge-base` (default) | 768 | 512 | General purpose, best quality |
| `minilm` | 384 | 256 | Fast, good for short text |
| `gte-base` | 768 | 512 | Alternative 768-dim, good for similarity |

### Enable Multiple Models

```bash
# Single model (default)
EMBED_MODELS=bge-base cargo run --release

# Multiple models
EMBED_MODELS=bge-base,minilm,gte-base cargo run --release

# All models
EMBED_MODELS=all cargo run --release
```

Models are loaded **lazily** on first request to save memory.

## API

### Embed Text (OpenAI-compatible)

```bash
POST /v1/embeddings
{
  "input": "text to embed",      # or ["text1", "text2", ...]
  "model": "bge-base"            # optional, defaults to bge-base
}

# Response
{
  "object": "list",
  "data": [{ "embedding": [0.1, -0.2, ...], "index": 0 }],
  "model": "bge-base",
  "usage": { "prompt_tokens": 3, "total_tokens": 3 }
}
```

### Multi-Model Embedding

Use `model: "*"` or `model: "all"` to get embeddings from all enabled models:

```bash
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, world!", "model": "*"}'

# Response
{
  "object": "list",
  "embeddings": [
    { "model": "bge-base", "dimensions": 768, "data": [...] },
    { "model": "minilm", "dimensions": 384, "data": [...] },
    { "model": "gte-base", "dimensions": 768, "data": [...] }
  ],
  "timing": {
    "total_ms": 45,
    "per_model": { "bge-base": 15, "minilm": 10, "gte-base": 20 }
  }
}
```

### List Models

```bash
GET /v1/models

# Response
{
  "object": "list",
  "data": [
    { "id": "bge-base", "dimensions": 768, "max_tokens": 512, "loaded": true },
    { "id": "minilm", "dimensions": 384, "max_tokens": 256, "loaded": false },
    { "id": "gte-base", "dimensions": 768, "max_tokens": 512, "loaded": false }
  ]
}
```

### Health Check

```bash
GET /health

# Response
{
  "status": "ok",
  "models": [...],
  "loaded_count": 1,
  "version": "0.1.0"
}
```

## Engram Integration

```env
# In engram/.env
EMBEDDING_PROVIDER=local
EMBEDDING_LOCAL_URL=http://127.0.0.1:8080
EMBEDDING_DIMENSIONS=768
```

**Note:** Requires Pinecone index matching your model dimensions — see SPEC.md for migration.

## Multi-Model Ensemble Usage

For improved retrieval accuracy, Engram's ensemble retrieval system uses multiple embedding models simultaneously. Enable multi-model mode:

```bash
# Start with all models enabled
EMBED_MODELS=bge-base,minilm,gte-base cargo run --release
```

### How It Works

1. **Embedding**: Each memory is embedded using all active models
2. **Query**: Queries are sent to all models in parallel
3. **Fusion**: Results are combined using Reciprocal Rank Fusion (RRF)
4. **Consensus**: Memories found by multiple models score higher

### Dashboard Visibility

The [Engram Dashboard](https://github.com/heybeaux/engram-dashboard) provides visibility into multi-model embeddings:

- **Memory Detail → Embeddings Tab**: See which models have embeddings per memory
- **Ensemble Overview Page**: Model status, coverage stats, A/B test results
- **Re-embedding Management**: Trigger and monitor batch re-embedding jobs

### Model Selection

Different models excel at different query types:

| Model | Best For | Trade-off |
|-------|----------|-----------|
| `bge-base` | General purpose | Balanced quality/speed |
| `minilm` | Short queries | Fastest, good precision |
| `gte-base` | Long documents | Similar to bge-base |
| `nomic`* | Very long context | 8K tokens, requires API |

*nomic-embed-text requires the Nomic API.

### Configuration in Engram

```env
# Enable ensemble retrieval
ENSEMBLE_ENABLED=true
ENSEMBLE_MODELS=bge-base,minilm
ENSEMBLE_WEIGHTS={"bge-base": 1.0, "minilm": 0.8}
ENSEMBLE_RRF_K=60
ENSEMBLE_CONSENSUS_BOOST=true
ENSEMBLE_CONSENSUS_FACTOR=1.2
```

## Performance

On M2 MacBook Pro (CPU mode):

| Model | Single Text | Batch of 100 |
|-------|-------------|--------------|
| bge-base | ~10ms | ~400ms |
| minilm | ~5ms | ~200ms |
| gte-base | ~10ms | ~400ms |

Models are loaded lazily. First request for each model incurs ~2-5s load time.

## Memory

Each model uses approximately:

| Model | Memory |
|-------|--------|
| bge-base | ~450MB |
| minilm | ~90MB |
| gte-base | ~450MB |

The server keeps up to 3 models loaded with LRU eviction.

## License

MIT
