# Design Notes — Embed Pipeline Overhaul

## Investigation Trail

Two independent sub-agent investigations on 2026-05-25 produced overlapping findings on different layers of the stack.

### Investigation 1 — Request lifecycle trace

Traced an embedding request from HTTP → response. Smoking gun: `metal_bert.rs:315` uses `reshape((batch_size, seq_len, ()))` with `()` as a dimension placeholder. Candle's reshape API does not interpret `()` as a wildcard in this position; the call produces a tensor with incorrect strides. Downstream layers read the wrong memory offsets, and `to_vec2::<f32>()` extraction produces a Vec<f32> with garbage values that serialize as `[,,,,]` because intermediate slots are missing. Pax's commit 228c8ea catches `NaN`/`Inf` after extraction but passes finite garbage through.

### Investigation 2 — Adversarial sweep

Found additional issues:

- **Nomic v1.5 architectural mismatch.** Code does post-norm; HF config specifies pre-norm. All Nomic embeddings to date are likely wrong, just plausible enough to not raise alarms.
- **No HTTP input bounds.** Single-request OOM is trivial (`{"input": [<10k strings>]}` or one giant string).
- **`get_or_load` double-load race.** Two simultaneous requests for an unloaded model both pass the read-check, both call load. Wasteful, not corrupting, but contributes to startup latency spikes.
- **`embed_all` silent error swallowing.** A failed model logs and returns `None`; clients receive a partial response with no indication.
- **Unwrap panics in handler hot paths.** `serde_json::to_value(...).unwrap()` will crash the entire request on serialization failure.
- **No integration tests in CI.** All integration tests are `#[ignore]`. The defensive guard at 228c8ea is only unit-tested against synthetic NaN inputs; the actual production failure mode (corrupt strides → sparse vec) is not in the test suite.

### Investigation 3 — Schema review (2026-05-25)

Identified the pgvector indexing problem: the current `embeddings` table stores `vector` without a declared dimension because it mixes models. pgvector requires `vector(N)` with a fixed N to build IVFFlat or HNSW indexes. Without an index, every similarity search is O(n) across all rows — at 13k+ memories with 4 models each, that's 52k+ rows scanned per query with no index pruning.

The fix is schema honesty: one table per model, declared dimension per table.

## Why this isn't a rewrite

Per product direction (2026-05-25, Beaux): ensemble + rank fusion stays. The codebase's architecture is sound — multi-model registry, lazy loading, LRU eviction, OpenAI-compatible API. The bugs are localized: one bad reshape, one architectural mismatch in one model, one missing async boundary, one missing input validator. Treating this as a rewrite would discard months of correct work on Metal compat, tokenization, and model integration.

## Per-Model Table Design

### Why one table per model, not one row with a JSON embedding

A single table with a `jsonb` or `float[]` embedding column is not indexable by pgvector. pgvector's `vector(N)` type requires a fixed declared N at DDL time. Mixing dims in one column means you either:
1. Declare a max dim (e.g. `vector(1536)`) and zero-pad smaller models → wastes storage, corrupts cosine similarity for smaller models.
2. Use `float[]` → no pgvector index at all, O(n) scans forever.
3. One table per model → correct declared dim, real index, independent tuning.

Option 3 is the only one that doesn't trade correctness for convenience.

### Schema shape

```sql
-- openai text-embedding-3-small: 1536 dims
CREATE TABLE embeddings_openai_small (
    memory_id     UUID        NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    vector        vector(1536) NOT NULL,
    model_version TEXT         NOT NULL,  -- e.g. "text-embedding-3-small"
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
    PRIMARY KEY (memory_id)
);
CREATE INDEX ON embeddings_openai_small USING hnsw (vector vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- BAAI/bge-base-en-v1.5: 768 dims
CREATE TABLE embeddings_bge_base (
    memory_id     UUID        NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    vector        vector(768)  NOT NULL,
    model_version TEXT         NOT NULL,
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
    PRIMARY KEY (memory_id)
);
CREATE INDEX ON embeddings_bge_base USING hnsw (vector vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- all-MiniLM-L6-v2: 384 dims
CREATE TABLE embeddings_minilm (
    memory_id     UUID        NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    vector        vector(384)  NOT NULL,
    model_version TEXT         NOT NULL,
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
    PRIMARY KEY (memory_id)
);
CREATE INDEX ON embeddings_minilm USING hnsw (vector vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- nomic-embed-text-v1.5: 768 dims — QUARANTINED (opt-in only, see quarantine notes)
CREATE TABLE embeddings_nomic (
    memory_id     UUID        NOT NULL REFERENCES memories(id) ON DELETE CASCADE,
    vector        vector(768)  NOT NULL,
    model_version TEXT         NOT NULL,  -- e.g. "nomic-embed-text-v1.5"
    created_at    TIMESTAMPTZ  NOT NULL DEFAULT now(),
    PRIMARY KEY (memory_id)
);
CREATE INDEX ON embeddings_nomic USING hnsw (vector vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);
```

### HNSW vs IVFFlat

HNSW is preferred here over IVFFlat:
- IVFFlat requires a two-pass build (cluster then index) and degrades if the list count is misconfigured relative to row count.
- HNSW builds incrementally, no training pass, works well at 10k–100k rows.
- `m=16, ef_construction=64` is a safe conservative default; tunable per-model independently.

If row count grows beyond 1M for a given model, revisit IVFFlat with proper `lists` tuning. That's a separate migration decision per table.

### Discriminator service layer

A single module owns the model→table mapping. No other code hardcodes table names.

```typescript
// src/embedding/discriminator.ts (~25 LOC)
const MODEL_TABLE_MAP: Record<string, string> = {
  'text-embedding-3-small': 'embeddings_openai_small',
  'BAAI/bge-base-en-v1.5':  'embeddings_bge_base',
  'all-MiniLM-L6-v2':       'embeddings_minilm',
  // nomic is quarantined; only routed when NOMIC_EMBED_OPT_IN=true
  'nomic-embed-text-v1.5':  'embeddings_nomic',
};

const QUARANTINED = new Set(['nomic-embed-text-v1.5']);

export function getTableForModel(model: string): string {
  if (QUARANTINED.has(model) && !process.env.NOMIC_EMBED_OPT_IN) {
    throw new Error(`Model ${model} is quarantined. Set NOMIC_EMBED_OPT_IN=true to use.`);
  }
  const table = MODEL_TABLE_MAP[model];
  if (!table) throw new Error(`Unknown model: ${model}`);
  return table;
}

export function activeModels(): string[] {
  return Object.keys(MODEL_TABLE_MAP).filter(m => !QUARANTINED.has(m) || !!process.env.NOMIC_EMBED_OPT_IN);
}
```

### Memory row as canonical; embeddings as satellites

The `memories` table remains the source of truth for content, metadata, and timestamps. An embedding row is a derived artifact: it can be deleted and regenerated without data loss. This means:
- `ON DELETE CASCADE` on the FK: deleting a memory automatically removes its embedding rows.
- A memory with no embedding rows is valid (not yet embedded, or in a quarantined-model gap).
- The API reads from whichever tables are configured for the active ensemble, joining back to `memories` as needed.

### Cross-model search

When the ensemble search needs results from multiple models, it uses `UNION ALL` with an added discriminator column:

```sql
SELECT memory_id, 1 - (vector <=> $1) AS score, 'openai_small' AS source_model
FROM embeddings_openai_small
ORDER BY vector <=> $1
LIMIT 20

UNION ALL

SELECT memory_id, 1 - (vector <=> $2) AS score, 'bge_base' AS source_model
FROM embeddings_bge_base
ORDER BY vector <=> $2
LIMIT 20
```

Each sub-query hits its own HNSW index. The outer result is merged and re-ranked by the existing rank fusion layer (RRF or score-based). Note: $1 and $2 are different vectors (different models produce different-dimensional embeddings from the same query text), so the query must embed the query text once per active model.

### Migration strategy

The migration is non-destructive and can be applied to a live database:

1. **Add tables** (T26–T29): `CREATE TABLE IF NOT EXISTS` for all four. Safe to run on prod; no existing data touched.
2. **Add indexes** (T26–T29): HNSW builds are concurrent in PG 15+ (`CREATE INDEX CONCURRENTLY`). No table lock.
3. **Backfill bge-base** (T33): one-shot script copies from `embeddings` → `embeddings_bge_base`. Uses `INSERT INTO ... SELECT` with a dimension check (rows where `array_length(vector, 1) = 768`). Idempotent: `ON CONFLICT (memory_id) DO NOTHING`.
4. **Re-embed into openai_small** (separate workstream): this is the 13k-memory backfill blocked on the ingest tx-closed fix. The tables must exist before this can run, which is why the schema migration lands first.
5. **Cutover reads** (after backfill): flip the query builder to read from `embeddings_openai_small` as primary. Keep reading from `embeddings_bge_base` as secondary ensemble member.
6. **Deprecate old table** (follow-on migration, T36): after reads are verified against new tables for 1+ week, drop `embeddings`. Not in this PR.

## Risk: changing embedding values

T1 (the reshape fix) and T4 (Nomic pre-norm) will change what the models actually compute. This breaks consistency with any embeddings already written to cloud Engram. Mitigation:

1. T6 — fixture comparison vs sentence-transformers Python reference. Must hit cosine ≥ 0.999 per model. If a fix lands and the fixture moves more than that, it indicates a deeper issue.
2. Coordinate with the separate 13k-backfill workstream — once Phase 1 is verified correct, that backfill should re-embed against the fixed runtime, not the broken one.
3. The fixture set is checked in; any future regression to the model forward passes is caught in CI.

## Ensemble fan-out concurrency

Current: sequential. Proposed: parallel via `try_join_all`. Concern — running 4 models concurrently on a single Metal device may serialize at the GPU level anyway, providing no real speedup while adding scheduler overhead. Mitigation:

- Per-model semaphore (T8) defaults to permit=1 for Metal-backed models. Effectively this means only one Metal model runs at a time even with `try_join_all`, but a CPU-backed model can interleave. This is the right shape for mixed local/cloud (local Metal serialized, cloud API parallel).
- A future enhancement (out of scope here) is multi-device or async Metal command queues; not needed for v1.

## Validation philosophy

Pax's 228c8ea is the right idea — defense-in-depth at the response boundary. The fix is to make sure the validation **never trips** in normal operation. If T1+T4 land correctly, `validate_embedding_batch` should reject nothing. The metric `embed.validation.rejected` becomes a canary: any non-zero rate indicates regression somewhere upstream.

## Quarantine model independence

One of the explicit goals of per-model tables is that a broken model can be quarantined without affecting recall for other models. The quarantine enforcement:
- In `engram-embed`: the `--quarantine` flag (added 2026-05-25 commit 9e7bde3) prevents a model from being loaded.
- In Engram API discriminator: `QUARANTINED` set prevents routing writes/reads to the model's table unless `NOMIC_EMBED_OPT_IN=true`.
- In Engram search: if a model's table is empty or skipped, the rank fusion layer still produces results from the remaining models. No silent degradation — metrics should show which models contributed to each result set.

The combination means: a model can be quarantined independently at the embed-service level, at the schema-routing level, or both. No single-table cross-model design offers this.

## Open Questions

1. **Should we add a query vs document distinction to the API?** Nomic recommends `search_query:` vs `search_document:` prefixes for optimal retrieval. Currently all texts get the document prefix. Adding a `task: "query" | "document"` field in the request would unlock real Nomic performance — but it's an API change. Defer to a separate proposal.

2. **OnceCell vs sharded RwLock for `get_or_load`?** Both work. `OnceCell` is cleaner but requires per-ModelId static or wrapped state. Sharded `RwLock<HashMap>` is closer to current shape. Pick during T9 implementation.

3. **Histogram lib choice.** `hdrhistogram` is the standard but heavy; `metrics-util` has lighter options. Defer to T20.

4. **Do we need a circuit breaker** between Engram ingest and engram-embed? Out of scope here, but worth raising in the Engram side. If embed is the bottleneck, ingest should backpressure rather than fan out 3.5k concurrent calls.

5. **HNSW index parameters per model.** The defaults `m=16, ef_construction=64` are conservative. At scale, openai_small (1536 dims) may benefit from higher `m` for recall quality. Revisit when row count exceeds 100k per table; track as a follow-on task.

6. **Backfill ordering.** Should bge-base backfill (T33) run before or after Phase 1 correctness fixes land? After: the bge-base output changes with T1, so backfilling from the old `embeddings` table captures corrupted vectors. Backfill should run against the corrected runtime. Coordination needed with the 13k-memories workstream owner.
