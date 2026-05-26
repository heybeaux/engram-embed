# Embed Pipeline Overhaul

**Status:** Draft
**Branch:** `feat/metal-embed-load` (pipeline fixes) + `feat/embed-per-model-tables` (schema)
**Owner:** Rook
**Co-investigators:** Pax (initial fix 228c8ea), Kit
**Date:** 2026-05-25

## Why

Engram's embedding layer is the source of the majority of production incidents in the memory faculty stack. Symptoms surface elsewhere (pgvector 22P02, missed recall, empty result sets, ingest tx-closed errors) but the root causes live in `engram-embed`. A targeted patch (commit 228c8ea by Pax) adds a defensive guard that rejects non-finite values before serialization. This is necessary but insufficient: it catches a symptom (`NaN`/`Inf` floats) downstream of the real problem (corrupted tensor strides producing finite-but-garbage memory reads that serialize as `[,,,,]` sparse arrays).

Two independent investigations (request-lifecycle trace + adversarial sweep) confirm the issues compound:

1. **Correctness bugs in the model forward passes** that silently corrupt embeddings.
2. **Async/Metal concurrency hazards** that surface only under load (3.5k parallel chunk ingest).
3. **No input bounds** on the HTTP layer, making the service trivially DoSable.
4. **Sequential ensemble fan-out** with silent error swallowing, masking root cause and starving the runtime.

A fifth root cause was identified on 2026-05-25 during schema review:

5. **Dim-agnostic single embedding table can't be pgvector-indexed.** The current `embeddings` table stores vectors from models with different dimensions (1536, 768, 384) in a single `vector` column. pgvector requires a fixed declared dimension for IVFFlat/HNSW index creation. Without a real index, every recall query is a sequential scan — this kills recall performance at scale and is the proximate cause of recall latency spikes observed in cloud Engram at 10k+ memories.

The goal of this change is not a rewrite. The goal is to make `feat/metal-embed-load` the new baseline by fixing the architectural debt that produced 228c8ea in the first place, so the defensive guard becomes redundant rather than load-bearing — and to add per-model embedding tables so pgvector indexing is actually possible.

## What Changes

### 1. Correctness Fixes (model forward passes)

- **CRITICAL** `metal_bert.rs:315` — replace `reshape((batch_size, seq_len, ()))` with the explicit `cfg.intermediate_size`. This is the proximate cause of the `[,,,,]` corruption observed in production.
- **CRITICAL** `nomic_bert.rs:263-271` — verify Nomic v1.5 architecture against HuggingFace config. Code currently implements post-norm; HF config has `prenorm=true`. Fix the layer ordering.
- Audit all `.reshape()`, `.permute()`, `.transpose()` call sites across `metal_bert.rs`, `nomic_bert.rs`, `qwen2_embed.rs`, `metal_compat.rs` for missing `.contiguous()` calls before matmul.
- Add runtime config validation: `assert hidden_size % num_attention_heads == 0` on model load, fail fast with a clear error rather than producing a silently-wrong `head_dim`.
- Reconcile epsilon constants: pooling mask uses `1e-9`, LayerNorm uses `1e-12`. Pick one source of truth per model and document the choice.

### 2. Async / Concurrency

- Restore `spawn_blocking()` around the inference forward pass. CPU-bound and Metal-bound work must not run on Tokio worker threads. (This was previously fixed and has regressed.)
- Add a per-model semaphore guarding concurrent Metal device invocations. Candle's Metal backend is not safe for concurrent GPU command submission on the same device.
- Fix the `get_or_load` double-load race: hold the write lock for the load-or-insert sequence, or use `tokio::sync::OnceCell` per model.
- Parallel ensemble fan-out: `embed_all()` should run each model's `spawn_blocking` task concurrently (bounded by the per-model semaphores), then collect with `try_join_all`. Errors propagate; no silent `None` swallowing.

### 3. HTTP Layer Hardening

- **CRITICAL** Bound request size: max batch count (configurable, default 256), max single-input length in bytes (configurable, default 1 MiB), max total batch bytes.
- Reject oversized requests with HTTP 413 before deserialization where possible (axum `RequestBodyLimit`).
- Add per-request timeout (configurable, default 60s).
- Replace `.unwrap()` calls in handlers (`main.rs:169, 240, 286`) with proper error responses.
- Optional API key check via env var (`EMBED_API_KEY`), default off for backward compat.

### 4. Validation Defense-in-Depth

- Keep Pax's `validate_embedding_batch` (228c8ea) but extend it:
  - Tensor shape/stride sanity check **before** `to_vec2()` extraction, not after.
  - Stronger detection: variance check (a fully-zero or fully-identical vector is also garbage), L2 norm bounds (post-pooled embeddings should have a known norm range per model).
- Emit a structured `embed.validation.rejected` metric tagged by model + failure mode so we can see corruption rates per model in prod.

### 5. Per-Model Embedding Tables (Schema)

The current single `embeddings` table cannot carry a pgvector dimension declaration because it mixes models with different output dimensions. The fix is one table per model, each with a fixed declared dimension:

| Table | Dimension | Models | Status |
|---|---|---|---|
| `embeddings_openai_small` | 1536 | text-embedding-3-small | active (primary) |
| `embeddings_bge_base` | 768 | BAAI/bge-base-en-v1.5 | active |
| `embeddings_minilm` | 384 | all-MiniLM-L6-v2 | active |
| `embeddings_nomic` | 768 | nomic-embed-text-v1.5 | quarantined (opt-in) |

Each table has the same shape:

```sql
CREATE TABLE embeddings_<model> (
    memory_id   UUID        NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    vector      vector(<N>) NOT NULL,
    model_version TEXT      NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (memory_id)
);
CREATE INDEX ON embeddings_<model> USING hnsw (vector vector_cosine_ops);
```

The `memories` table remains the canonical row. Embedding rows are satellites: a memory may have zero or more embedding rows across tables (1:N across tables). The Engram API reads from whichever tables are active for the configured ensemble and UNION ALLs when cross-model search is needed.

A thin discriminator service layer (~30 LOC) in the Engram API routes `model → table` and is the single place where the model-to-table mapping lives. This avoids scattering the mapping across query builders.

**Migration strategy:**
1. Create the four new tables alongside the existing `embeddings` table (non-destructive).
2. Backfill existing rows: copy any bge-base-dimension (768) rows from `embeddings` into `embeddings_bge_base`.
3. Re-embed all memories into `embeddings_openai_small` as the new primary (this is a separate backfill workstream, estimated 13k memories in cloud).
4. Once the backfill is verified, deprecate reads from the old `embeddings` table and eventually drop it.

### 6. Test Coverage

- Add integration tests that actually run against real models and exercise:
  - Concurrent requests against the same model (race detection)
  - Large batches (256 inputs)
  - Maximum-length single inputs at each model's max_tokens
  - Multi-model ensemble fan-out
- A load test harness that reproduces the 3.5k parallel-chunk scenario locally. Until this passes cleanly, we don't declare victory.
- Property test: for any valid input, embedding output passes the strengthened validation (no NaN, no Inf, no zero-vec, correct dim).

### 7. Observability

- Per-model latency histogram (p50/p95/p99), not just rolling average.
- Validation rejection counter tagged by failure mode.
- Metal device queue depth (if extractable).
- Structured logs at error boundaries — every model failure logs the input shape, model, and underlying error.

### Out of Scope

- Single-model architecture change. Ensemble + rank fusion stays (per product decision 2026-05-25). The user picks open vs closed source, local vs cloud; the runtime supports both.
- The 13k bge-base-only backfill in cloud Engram (separate workstream, blocked on ingest pipeline tx-closed bug). The schema migration here creates the target tables; the backfill is a separate agent task.
- Rewriting the service in a different framework. Axum + Candle stays.
- Adding new models. KaLM-V2 was the last addition; we stabilize before extending.
- Dropping the old `embeddings` table. That's a separate migration after backfill is verified.

## Impact

- **Affected code:** `src/main.rs`, `src/embedder.rs`, `src/metal_bert.rs`, `src/nomic_bert.rs`, `src/qwen2_embed.rs`, `src/metal_compat.rs`, `tests/integration.rs`, new `tests/load_test.rs`.
- **Affected schema:** `engram/prisma/schema.prisma` — four new model tables, HNSW indexes, discriminator service layer in Engram API.
- **Affected callers:** All Engram instances (local + cloud) consuming engram-embed. No breaking API change planned, but embedding *values* will change for Nomic once the pre-norm fix lands. Existing Nomic-embedded memories will need re-embedding to remain consistent — this is a known and accepted cost.
- **Risk:** Medium. The reshape fix is small but its blast radius is large (it changes what bge-base, minilm, gte-base actually compute). We need before/after embedding comparison on a held-out fixture set to confirm the new output matches reference implementations (sentence-transformers Python) within tolerance. The schema migration is non-destructive (additive tables) so rollback risk is low.

## Success Criteria

1. `[,,,,]` sparse-vector corruption rate drops to zero under 3.5k-parallel load test.
2. Pax's `validate_embedding_batch` rejects nothing in normal operation (the guard becomes proof of correctness, not a band-aid).
3. Multi-model ensemble at 10k+ memories completes without bailing.
4. Embeddings produced match reference sentence-transformers output within cosine ≥ 0.999 per model on a fixture set.
5. `feat/metal-embed-load` merges to main as the new baseline.
6. Four per-model embedding tables exist in schema with declared-dimension HNSW indexes.
7. Recall queries hit the index (no sequential scans on `embeddings_*` tables) under EXPLAIN ANALYZE.
8. Discriminator service layer maps every active model to its table in ≤ 30 LOC.
