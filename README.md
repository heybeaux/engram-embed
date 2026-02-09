<p align="center">
  <h1 align="center">engram-embed</h1>
  <p align="center"><strong>Local embedding server in Rust.</strong></p>
  <p align="center">
    <strong>Ecosystem:</strong>&nbsp;
    <a href="https://github.com/heybeaux/engram">Memory API</a> •
    <a href="https://github.com/heybeaux/engram-code">Code Search</a> •
    <a href="https://github.com/heybeaux/engram-dashboard">Dashboard</a> •
    <b>Local Embeddings</b>
  </p>
</p>

Drop-in replacement for OpenAI's embeddings API. Zero cost, sub-10ms latency, data never leaves your machine.

---

## Why Local Embeddings?

| | OpenAI | engram-embed |
|--|--------|--------------|
| **Cost** | $0.0001/1K tokens | Free |
| **Latency** | ~100ms (network) | ~10ms (local) |
| **Rate limits** | Yes | None |
| **Privacy** | Data sent to cloud | Data stays local |
| **Offline** | No | Yes |

At scale: **$100+/day → $0/day**

## Quick Start

```bash
# Install Rust (if needed)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Clone and build
git clone https://github.com/heybeaux/engram-embed
cd engram-embed
cargo build --release

# Run (models download on first request)
cargo run --release

# Test it
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, world!"}'
```

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                       engram-embed                           │
│                                                              │
│  ┌──────────────────────────────────────────────────────┐   │
│  │                    Axum Server                        │   │
│  │              POST /v1/embeddings                      │   │
│  │               GET /v1/models                          │   │
│  │                GET /health                            │   │
│  └───────────────────────┬──────────────────────────────┘   │
│                          │                                   │
│  ┌───────────────────────▼──────────────────────────────┐   │
│  │               ModelRegistry (lazy loading)            │   │
│  │                                                       │   │
│  │   ┌─────────────┐ ┌─────────────┐ ┌─────────────┐    │   │
│  │   │  bge-base   │ │   minilm    │ │  gte-base   │    │   │
│  │   │   768-dim   │ │   384-dim   │ │   768-dim   │    │   │
│  │   │   512 tok   │ │   256 tok   │ │   512 tok   │    │   │
│  │   └─────────────┘ └─────────────┘ └─────────────┘    │   │
│  │                                                       │   │
│  │                    ┌─────────────┐                    │   │
│  │                    │    nomic    │                    │   │
│  │                    │   768-dim   │                    │   │
│  │                    │   8192 tok  │                    │   │
│  │                    └─────────────┘                    │   │
│  └───────────────────────────────────────────────────────┘   │
│                          │                                   │
│  ┌───────────────────────▼──────────────────────────────┐   │
│  │                  Candle Runtime                       │   │
│  │           HuggingFace's Rust ML Framework            │   │
│  │               (CPU / Metal acceleration)              │   │
│  └───────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

## Models

| Model | Dimensions | Max Tokens | Best For | Memory |
|-------|------------|------------|----------|--------|
| `bge-base` | 768 | 512 | General purpose, best quality | ~450MB |
| `minilm` | 384 | 256 | Fast, short text | ~90MB |
| `gte-base` | 768 | 512 | Alternative semantic space | ~450MB |
| `nomic` | 768 | 8192 | Long documents, code | ~550MB |

**Default:** `bge-base` — top-tier open-source embeddings, excellent quality/speed tradeoff.

### Enable Multiple Models

```bash
# Single model (default)
EMBED_MODELS=bge-base cargo run --release

# Multiple models for ensemble
EMBED_MODELS=bge-base,minilm,nomic cargo run --release

# All available models
EMBED_MODELS=all cargo run --release
```

Models are loaded **lazily** on first request to save memory. Up to 3 models kept loaded with LRU eviction.

## API Reference

### OpenAI-Compatible Endpoint

```bash
POST /v1/embeddings
```

**Request:**
```json
{
  "input": "text to embed",      // string or array of strings
  "model": "bge-base"            // optional, defaults to bge-base
}
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "object": "embedding",
      "embedding": [0.123, -0.456, ...],
      "index": 0
    }
  ],
  "model": "bge-base",
  "usage": {
    "prompt_tokens": 3,
    "total_tokens": 3
  }
}
```

### Multi-Model Embedding

Use `model: "*"` or `model: "all"` to embed with all enabled models at once:

```bash
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -H "Content-Type: application/json" \
  -d '{"input": "Hello, world!", "model": "*"}'
```

**Response:**
```json
{
  "object": "list",
  "embeddings": [
    {
      "model": "bge-base",
      "dimensions": 768,
      "data": [{ "embedding": [...], "index": 0 }]
    },
    {
      "model": "minilm",
      "dimensions": 384,
      "data": [{ "embedding": [...], "index": 0 }]
    }
  ],
  "timing": {
    "total_ms": 25,
    "per_model": { "bge-base": 12, "minilm": 8 }
  }
}
```

### List Models

```bash
GET /v1/models
```

**Response:**
```json
{
  "object": "list",
  "data": [
    {
      "id": "bge-base",
      "dimensions": 768,
      "max_tokens": 512,
      "loaded": true
    },
    {
      "id": "minilm",
      "dimensions": 384,
      "max_tokens": 256,
      "loaded": false
    }
  ]
}
```

### Health Check

```bash
GET /health
```

**Response:**
```json
{
  "status": "ok",
  "models": [
    { "id": "bge-base", "dimensions": 768, "max_tokens": 512, "loaded": true, "default": true }
  ],
  "loaded_count": 1,
  "version": "0.1.0"
}
```

## The Truncation Fix

BERT-based models have a maximum sequence length (typically 512 tokens). Without truncation, long inputs cause a panic:

```
thread 'main' panicked at 'index out of bounds: position embeddings only support 512 tokens'
```

**engram-embed handles this automatically:**

```rust
// Truncation enabled on tokenizer initialization
tokenizer.with_truncation(Some(TruncationParams {
    max_length: model.max_tokens(),      // 512 for bge-base
    strategy: TruncationStrategy::LongestFirst,
    direction: TruncationDirection::Right,
}));
```

This means:
- Long text is automatically truncated to fit the model
- No panics or errors on long inputs
- Truncation happens from the right (keeps the beginning)
- Works for all models with their respective limits

**For very long content (code files, documents):** Use the `nomic` model which supports 8192 tokens.

## Integration with Engram

### Engram (Memory API)

```env
# In engram/.env
EMBEDDING_PROVIDER=local
EMBEDDING_LOCAL_URL=http://127.0.0.1:8080
EMBEDDING_DIMENSIONS=768
```

### engram-code (Code Search)

```env
# In engram-code/.env
ENGRAM_EMBED_URL=http://127.0.0.1:8080
```

Both services share the same embedding server for consistent vector representations.

## Ensemble Retrieval

For improved search accuracy, use multiple models together:

```
┌─────────────────────────────────────────────────────┐
│  Query: "user authentication"                       │
│                                                     │
│  ┌─────────────┐     ┌─────────────┐               │
│  │  bge-base   │     │   nomic     │               │
│  │  General    │     │  Long ctx   │               │
│  │  purpose    │     │  semantic   │               │
│  └──────┬──────┘     └──────┬──────┘               │
│         │                   │                       │
│         └─────────┬─────────┘                       │
│                   ▼                                 │
│         ┌─────────────────┐                         │
│         │   RRF Fusion    │                         │
│         │  (in engram /   │                         │
│         │   engram-code)  │                         │
│         └─────────────────┘                         │
│                   │                                 │
│         Better recall than single model             │
└─────────────────────────────────────────────────────┘
```

**Why multiple models?**
- Different models capture different semantic aspects
- Consensus (found by multiple models) increases confidence
- Reduces single-model blind spots
- Nomic's 8K context catches patterns bge-base might miss

### Configuration for Ensemble

```env
# In engram/.env
ENSEMBLE_ENABLED=true
ENSEMBLE_MODELS=bge-base,nomic
ENSEMBLE_WEIGHTS={"bge-base": 1.0, "nomic": 0.8}
ENSEMBLE_RRF_K=60
```

## Performance

On M2 MacBook Pro (CPU):

| Operation | bge-base | minilm | nomic |
|-----------|----------|--------|-------|
| Single text | ~10ms | ~5ms | ~15ms |
| Batch of 100 | ~400ms | ~200ms | ~600ms |
| First request (load) | ~3s | ~2s | ~5s |

**Memory usage:**
- 1 model loaded: ~500MB
- 2 models loaded: ~1GB
- 3 models loaded: ~1.5GB

Models are loaded lazily and evicted LRU when memory limit reached.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `EMBED_MODELS` | `bge-base` | Models to enable (comma-separated or `all`) |
| `PORT` | `8080` | Server port |

## Tech Stack

| Component | Technology | Why |
|-----------|------------|-----|
| Language | Rust | Performance, single binary, memory safety |
| HTTP | Axum | Async, ergonomic, Tokio-based |
| ML Runtime | Candle | HuggingFace's Rust ML, Apple Silicon support |
| Tokenizer | tokenizers | Rust-native, fast |

## Building

```bash
# Debug build (faster compile, slower runtime)
cargo build

# Release build (slower compile, optimized runtime)
cargo build --release

# Run tests
cargo test

# Run with specific models
EMBED_MODELS=bge-base,minilm cargo run --release
```

## Running as a Service (macOS)

Create `~/Library/LaunchAgents/com.engram.embed.plist`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>com.engram.embed</string>
    <key>ProgramArguments</key>
    <array>
        <string>/path/to/engram-embed/target/release/engram-embed</string>
    </array>
    <key>EnvironmentVariables</key>
    <dict>
        <key>EMBED_MODELS</key>
        <string>bge-base,nomic</string>
    </dict>
    <key>RunAtLoad</key>
    <true/>
    <key>KeepAlive</key>
    <true/>
    <key>StandardOutPath</key>
    <string>/tmp/engram-embed.log</string>
    <key>StandardErrorPath</key>
    <string>/tmp/engram-embed.err</string>
</dict>
</plist>
```

```bash
# Load the service
launchctl load ~/Library/LaunchAgents/com.engram.embed.plist

# Check status (PID shown means running)
launchctl list | grep engram

# View logs
tail -f /tmp/engram-embed.log

# Restart the service
launchctl unload ~/Library/LaunchAgents/com.engram.embed.plist
launchctl load ~/Library/LaunchAgents/com.engram.embed.plist
```

### Uninstall

```bash
# Stop and unload the service
launchctl unload ~/Library/LaunchAgents/com.engram.embed.plist

# Remove the plist file
rm ~/Library/LaunchAgents/com.engram.embed.plist

# Optional: remove log files
rm /tmp/engram-embed.log /tmp/engram-embed.err

# Optional: remove cached model files
rm -rf ~/.cache/huggingface/hub/models--BAAI--bge-base-en-v1.5
rm -rf ~/.cache/huggingface/hub/models--nomic-ai--nomic-embed-text-v1.5
```

## Troubleshooting

### Model download fails

Models are downloaded from HuggingFace Hub on first request. If download fails:

```bash
# Check network connectivity
curl -I https://huggingface.co

# Pre-download model manually
huggingface-cli download BAAI/bge-base-en-v1.5
```

### Out of memory

Reduce the number of loaded models:

```bash
EMBED_MODELS=bge-base cargo run --release
```

### Slow first request

First request for each model triggers download + load (~3-5s). Subsequent requests are fast (~10ms).

To pre-warm models on startup:
```bash
# After starting server, hit each model once
curl -X POST http://127.0.0.1:8080/v1/embeddings \
  -d '{"input": "warmup", "model": "bge-base"}'
```

## License

MIT

---

<p align="center">
  <em>Embeddings, locally.</em>
</p>
