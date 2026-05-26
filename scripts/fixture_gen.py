#!/usr/bin/env python3
"""
Generate embedding fixtures for Phase 1 fixture comparison test.

Computes embeddings for 20 short texts using sentence-transformers (reference
implementation) for each of the 4 ensemble models, and saves them as JSON
fixtures in tests/fixtures/.

Usage:
    python3 scripts/fixture_gen.py

Requirements:
    pip install sentence-transformers
"""

import json
import os
import sys
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("sentence-transformers not installed. Run: pip install sentence-transformers")
    sys.exit(1)

# 20 short texts covering diverse content types
TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "Machine learning models transform text into dense vector representations.",
    "Paris is the capital city of France.",
    "def fibonacci(n): return n if n < 2 else fibonacci(n-1) + fibonacci(n-2)",
    "The mitochondria is the powerhouse of the cell.",
    "To be or not to be, that is the question.",
    "Quantum entanglement enables instantaneous state correlation across distance.",
    "She sells seashells by the seashore.",
    "The Rust programming language guarantees memory safety without a garbage collector.",
    "Neural networks consist of layers of interconnected artificial neurons.",
    "Climate change is accelerating the frequency of extreme weather events.",
    "import numpy as np; arr = np.zeros((3, 4))",
    "The speed of light in a vacuum is approximately 299,792,458 meters per second.",
    "Embeddings capture semantic similarity between pieces of text.",
    "SELECT * FROM users WHERE created_at > '2024-01-01' ORDER BY id DESC;",
    "The Amazon rainforest produces 20% of the world's oxygen.",
    "async fn handle_request(req: Request) -> Response { ... }",
    "Transformer architecture revolutionized natural language processing in 2017.",
    "Water freezes at 0 degrees Celsius and boils at 100 degrees Celsius.",
    "The derivative of sin(x) with respect to x is cos(x).",
]

# Model mapping: our internal name -> HuggingFace model ID
MODELS = {
    "bge-base": ("BAAI/bge-base-en-v1.5", None),
    "minilm": ("sentence-transformers/all-MiniLM-L6-v2", None),
    "gte-base": ("thenlper/gte-base", None),
    # Nomic requires search_document: prefix for document embeddings
    "nomic": ("nomic-ai/nomic-embed-text-v1.5", "search_document: "),
}

def compute_fixtures(output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    all_fixtures = {}

    for model_name, (hf_id, prefix) in MODELS.items():
        print(f"\nLoading {model_name} ({hf_id})...")

        # Nomic requires trust_remote_code for the custom architecture
        trust_remote = model_name == "nomic"
        model = SentenceTransformer(hf_id, trust_remote_code=trust_remote)

        # Apply prefix if required
        texts_with_prefix = texts = TEXTS
        if prefix:
            texts_with_prefix = [f"{prefix}{t}" for t in TEXTS]

        print(f"  Computing embeddings for {len(TEXTS)} texts...")
        # batch_size=1: some BERT models (e.g. GTE) have non-deterministic embeddings
        # when texts of different lengths are batched together due to padding interactions
        # with absolute position embeddings. Single-text batches are the stable reference.
        embeddings = model.encode(
            texts_with_prefix,
            normalize_embeddings=True,
            show_progress_bar=False,
            batch_size=1,
        )

        fixture = {
            "model": model_name,
            "hf_id": hf_id,
            "prefix": prefix,
            "texts": TEXTS,
            "embeddings": [emb.tolist() for emb in embeddings],
            "dimensions": embeddings[0].shape[0],
        }

        # Save per-model fixture
        model_path = output_dir / f"{model_name}.json"
        with open(model_path, "w") as f:
            json.dump(fixture, f, separators=(",", ":"))
        print(f"  Saved {model_path} ({embeddings[0].shape[0]} dims)")

        all_fixtures[model_name] = fixture

    # Save combined fixture manifest
    manifest = {
        "generated_by": "scripts/fixture_gen.py",
        "sentence_transformers_version": __import__("sentence_transformers").__version__,
        "num_texts": len(TEXTS),
        "models": list(MODELS.keys()),
    }
    manifest_path = output_dir / "manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nSaved manifest: {manifest_path}")
    print("Done.")


if __name__ == "__main__":
    repo_root = Path(__file__).parent.parent
    output_dir = repo_root / "tests" / "fixtures"
    compute_fixtures(output_dir)
