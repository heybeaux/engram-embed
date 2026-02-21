# KaLM-Embedding-V2 Integration Spec

**Status:** Proposed
**Author:** Rook ♜
**Date:** 2026-02-20
**Priority:** Medium
**Estimate:** 2-3 days

---

## Summary

Add KaLM-Embedding-V2 (0.5B, Qwen2-based) as an optional embedding model in engram-embed. Users choose which models to enable via config — KaLM-V2 is another option in the ensemble, not a forced default.

## Motivation

KaLM-V2 (sub-1B params) rivals models 3-26x larger on MTEB benchmarks. It gives users a significant quality upgrade for semantic recall without requiring cloud GPU infrastructure. Apache 2.0 licensed, fully local.

The goal is **user choice** — some users may enable it as their only model, some may add it as 1 of 5 in an ensemble, and others may skip it entirely. The point is that the option exists.

## Model Specs

| Property | Value |
|---|---|
| Base architecture | Qwen2-0.5B (modified with bidirectional attention) |
| Parameters | ~0.5B |
| Embedding dimensions | 896 |
| Max context | 512 tokens |
| Pooling | Mean pooling (already supported) |
| Key modification | Causal attention mask removed → fully bidirectional |
| HuggingFace ID | `HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v2` |
| License | Apache 2.0 |
| Instruction prefix | Yes — task-specific (see below) |

### Performance Expectations (Apple Silicon Metal)

| Metric | Current (bge-base, 110M) | KaLM-V2 (0.5B) |
|---|---|---|
| Memory | ~450MB | ~1GB (FP16) |
| Latency (single text) | ~5-15ms | ~50-150ms (est.) |
| Metal GPU | ✅ via MetalBert | ✅ via Candle Qwen2 |

## Implementation Plan

### 1. New source file: `src/qwen2_embed.rs`

- Copy Candle's `candle_transformers::models::qwen2` module
- Remove causal attention mask → fully bidirectional
- Expose `forward()` returning hidden states for mean pooling
- Load as FP16 by default (`DType::F16`)

### 2. Extend `ModelId` enum in `src/embedder.rs`

```rust
// Add to enum
KalmV2,

// from_str matches
"kalm-v2" | "kalm" | "kalm-embedding-v2"
    | "hit-tmg/kalm-embedding-multilingual-mini-instruct-v2" => Some(Self::KalmV2),

// to_hf_id
Self::KalmV2 => "HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v2",

// dimensions
Self::KalmV2 => 896,

// max_tokens
Self::KalmV2 => 512,

// display_name
Self::KalmV2 => "kalm-v2",
```

### 3. Extend `Embedder::new()` model loading

- Add `ModelId::KalmV2` arm that loads via `Qwen2Embed`
- Verify tensor name mapping (Qwen2 safetensors → Candle expectations)
- May need a weight renaming layer if tensor names don't match 1:1

### 4. Instruction prefix handling

KaLM-V2 uses task-aware prefixes (more complex than Nomic's static prefix):

- **Query prefix:** `"Instruct: Given a query, retrieve relevant passages\nQuery: "`
- **Document prefix:** None (or minimal)

This requires extending the current `prefix()` method to distinguish between query and document embeddings. Options:
- Add `query_prefix()` / `document_prefix()` methods to `ModelId`
- Or accept a `prefix_type` parameter in the embedding request

### 5. Update `ModelId::all()`

Include `KalmV2` so it appears in model listings. Users still control which models are enabled via `EMBED_MODELS` env var.

### 6. User experience

- **Disabled by default** — not in the default `EMBED_MODELS` list
- Users opt in: `EMBED_MODELS=bge-base,kalm-v2` or `EMBED_MODELS=kalm-v2`
- Works with existing ensemble weighting and LRU model eviction
- No migration needed — existing embeddings from other models are unaffected

## Risks

| Risk | Severity | Notes |
|---|---|---|
| Bidirectional attention mod | 🟡 Medium | Main unknown — removing causal mask from Candle Qwen2. Well-understood change but needs verification. |
| Tensor name mapping | 🟡 Medium | KaLM-V2 weights may not map 1:1 to Candle's Qwen2 expectations. May need rename layer. |
| Memory-aware LRU | 🟡 Low | 0.5B model weighs more than BERT models — LRU eviction currently treats all models equally. Consider size-aware eviction. |
| Tokenizer | 🟢 Low | Qwen2 BPE tokenizer, standard HF tokenizers crate — should work out of the box. |

## Future Work

- Q8 quantization for lower memory footprint (~500MB)
- KaLM-V2.5 variant support
- Matryoshka dimension support (flexible output dims) if KaLM adds it
- Consider as cloud-tier default for hosted Engram instances

## References

- [KaLM-V2 Paper](https://arxiv.org/abs/2506.20923)
- [KaLM-V1 Paper](https://arxiv.org/abs/2501.01028)
- [HuggingFace Model](https://huggingface.co/HIT-TMG/KaLM-embedding-multilingual-mini-instruct-v2)
- [GitHub](https://github.com/HITsz-TMG/KaLM-Embedding)
- [Candle Qwen2 Module](https://github.com/huggingface/candle/tree/main/candle-transformers/src/models/qwen2.rs)
