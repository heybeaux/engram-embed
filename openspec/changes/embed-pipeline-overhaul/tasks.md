# Tasks — Embed Pipeline Overhaul

Ordered for safe incremental landing on `feat/metal-embed-load`. Each task ships independently behind tests.

## Phase 1 — Correctness (no behavior change for callers, but embeddings change value)

- [ ] T1. Fix `metal_bert.rs:315` reshape — use `cfg.intermediate_size` explicitly. Add regression test that asserts intermediate tensor shape post-reshape.
- [ ] T2. Audit all `reshape`/`permute`/`transpose` sites across model files; add `.contiguous()` where matmul follows. Document each fix with a comment naming the matmul that requires it.
- [ ] T3. Add config validation on model load: `hidden_size % num_attention_heads == 0`, dim/intermediate sanity. Fail fast with named error.
- [ ] T4. Verify Nomic v1.5 pre-norm vs post-norm against HF config; fix layer ordering in `nomic_bert.rs`. Compare a fixture's embedding to sentence-transformers reference.
- [ ] T5. Reconcile epsilon constants — document per-model norm epsilon and pooling clamp choices.
- [ ] T6. Build a fixture-comparison test: for each supported model, embed a fixed corpus and assert cosine ≥ 0.999 vs a checked-in reference vector set (generated from sentence-transformers Python on the same model weights).

## Phase 2 — Async / Concurrency

- [ ] T7. Wrap forward pass in `tokio::task::spawn_blocking`. Verify no `RwLock` held across the await.
- [ ] T8. Add per-model `tokio::sync::Semaphore` for Metal device serialization. Permit count configurable per model (default 1 for Metal-backed, higher for CPU).
- [ ] T9. Fix `get_or_load` race using `tokio::sync::OnceCell` per ModelId, or restructured locking.
- [ ] T10. Rewrite `embed_all()` to run model tasks concurrently via `try_join_all` with per-model semaphore. Surface errors instead of swallowing.
- [ ] T11. Add a stress test: N=1000 concurrent requests against one model, then N=1000 against the ensemble. Assert no panics, no malformed responses, latency p99 stable.

## Phase 3 — HTTP Hardening

- [ ] T12. Add `RequestBodyLimit` middleware (configurable, default 8 MiB).
- [ ] T13. Validate `texts.len()` ≤ max_batch (default 256), each `text.len()` ≤ max_text_bytes (default 1 MiB) in handler. Return HTTP 400 with structured error.
- [ ] T14. Add per-request `tower::timeout` (default 60s).
- [ ] T15. Replace `.unwrap()` calls in `main.rs:169, 240, 286, 240` with proper error responses.
- [ ] T16. Optional `EMBED_API_KEY` env-var Bearer check. Default off (backward compat).

## Phase 4 — Validation Defense-in-Depth

- [ ] T17. Move shape/stride sanity check to before `to_vec2()` extraction in `embedder.rs`.
- [ ] T18. Extend `validate_embedding_batch` (228c8ea) — variance check, L2 norm bounds per model. Tag rejections by failure mode.
- [ ] T19. Add `embed.validation.rejected{model, reason}` metric and structured log.

## Phase 5 — Per-Model Embedding Tables (Schema)

Targets `engram/prisma/schema.prisma` in the Engram API repo. Branch: `feat/embed-per-model-tables`.

- [ ] T26. Create `embeddings_openai_small` table: `memory_id UUID PK FK→memories`, `vector vector(1536) NOT NULL`, `model_version TEXT NOT NULL`, `created_at TIMESTAMPTZ DEFAULT now()`. Add HNSW index: `USING hnsw (vector vector_cosine_ops)`.
- [ ] T27. Create `embeddings_bge_base` table: same shape, `vector vector(768)`. Add HNSW index.
- [ ] T28. Create `embeddings_minilm` table: same shape, `vector vector(384)`. Add HNSW index.
- [ ] T29. Create `embeddings_nomic` table: same shape, `vector vector(768)`, marked quarantined in schema comment. Add HNSW index (available when quarantine is lifted). Quarantine flag is enforced at the service layer (opt-in env var), not by dropping the table.
- [ ] T30. Write discriminator service layer (`src/embedding/discriminator.ts` or equivalent in Engram API, ≤ 30 LOC): `modelId → tableName` map, `getTableForModel(model: string): string` throws on unknown model. This is the single source of truth for the mapping; no other code hardcodes table names.
- [ ] T31. Update Engram API query builder to route SELECT/INSERT through the discriminator. Single-model queries target the model's table directly. Cross-model search uses `UNION ALL` across active (non-quarantined) tables with a `source_model` discriminator column added in the SELECT.
- [ ] T32. Write Prisma migration file (or raw SQL migration) for the four tables + indexes. Non-destructive — old `embeddings` table is not touched in this migration.
- [ ] T33. Backfill script (one-shot, not production code): copy bge-base-dimension rows from `embeddings` → `embeddings_bge_base` where the model_version column matches. Verify row count before and after. Log any rows skipped due to missing/mismatched dimension.
- [ ] T34. Update `EXPLAIN ANALYZE` snapshot in docs to confirm index scans on `embeddings_*` tables post-migration (no sequential scans).
- [ ] T35. Integration test: insert a memory, write an embedding row via discriminator, read it back, assert the correct table was hit (query `pg_stat_user_tables` for table scan counts before/after).

## Phase 6 — Observability

- [ ] T20. Replace rolling-average latency in `metrics.rs` with proper histogram (use `hdrhistogram` or similar). Expose p50/p95/p99 via `/metrics`.
- [ ] T21. Structured-log every error path with input shape, model, underlying error.

## Phase 7 — Load Test Harness & Sign-off

- [ ] T22. Build `tests/load_test.rs` that reproduces the 3.5k-parallel-chunk ingest scenario against a running server. Mark `#[ignore]` for CI but documented in README.
- [ ] T23. Run the load test 10x against the fully-fixed branch. Zero `[,,,,]` corruption, zero validation rejects, ensemble completes.
- [ ] T24. Update `SPEC.md` and `README.md` to reflect ensemble architecture, request limits, validation semantics, and per-model table routing.
- [ ] T25. Merge `feat/metal-embed-load` → `main`. Tag a release. Bump engram clients.
- [ ] T36. After T25 lands and backfill is verified: deprecate reads from old `embeddings` table, schedule drop in a follow-on migration.
