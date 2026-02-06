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

## API

OpenAI-compatible endpoint:

```bash
POST /v1/embeddings
{
  "input": "text to embed",      # or ["text1", "text2", ...]
  "model": "bge-base-en-v1.5"    # optional, we use local model
}

# Response
{
  "object": "list",
  "data": [{ "embedding": [0.1, -0.2, ...], "index": 0 }],  # 768 dims
  "model": "bge-base-en-v1.5",
  "usage": { "prompt_tokens": 3, "total_tokens": 3 }
}
```

## Engram Integration

```env
# In engram/.env
EMBEDDING_PROVIDER=local
EMBEDDING_LOCAL_URL=http://127.0.0.1:8080
EMBEDDING_DIMENSIONS=768
```

**Note:** Requires new Pinecone index (768-dim) — see SPEC.md for migration.

## Performance

On M2 MacBook Pro:
- Single embedding: ~10ms
- Batch of 100: ~400ms  
- With Metal: ~3ms / ~100ms

## Model

Uses `bge-base-en-v1.5` (768 dimensions, 440MB) — top-tier open-source embeddings.
First run downloads from HuggingFace.

## License

MIT
