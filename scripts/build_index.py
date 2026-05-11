"""
build_index.py — Step 2.2: Vector Embedding & FAISS Index Building

Embeds all catalog items using sentence-transformers `all-MiniLM-L6-v2`,
builds a FAISS IndexFlatIP (cosine similarity with normalized vectors),
and persists the index + metadata to `embeddings/faiss_index/`.

Run once:
    python -m scripts.build_index
"""

import json
import pickle
import sys
import time
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
PROCESSED_CATALOG = PROJECT_ROOT / "data" / "catalog_processed.json"
INDEX_DIR = PROJECT_ROOT / "embeddings" / "faiss_index"
INDEX_FILE = INDEX_DIR / "faiss_index.bin"
METADATA_FILE = INDEX_DIR / "metadata.pkl"

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


def load_catalog() -> list[dict]:
    """Load the preprocessed catalog."""
    print(f"Loading processed catalog from: {PROCESSED_CATALOG}")
    with open(PROCESSED_CATALOG, "r", encoding="utf-8") as f:
        data = json.load(f)
    print(f"  Loaded {len(data)} items")
    return data


def build_embeddings(catalog: list[dict], model: SentenceTransformer) -> np.ndarray:
    """Embed the text_for_embedding field for all catalog items."""
    texts = [item["text_for_embedding"] for item in catalog]
    print(f"Embedding {len(texts)} items with {EMBEDDING_MODEL}...")

    t0 = time.time()
    embeddings = model.encode(
        texts,
        show_progress_bar=True,
        normalize_embeddings=True,  # L2-normalize for cosine similarity via inner product
        batch_size=64,
    )
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s — shape: {embeddings.shape}")
    return embeddings


def build_faiss_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """Build a FAISS flat inner-product index (cosine sim with normalized vectors)."""
    dim = embeddings.shape[1]
    print(f"Building FAISS IndexFlatIP with dim={dim}...")
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings.astype(np.float32))
    print(f"  Index contains {index.ntotal} vectors")
    return index


def save_index(index: faiss.IndexFlatIP, metadata: dict):
    """Persist FAISS index and metadata to disk."""
    INDEX_DIR.mkdir(parents=True, exist_ok=True)

    print(f"Saving FAISS index to: {INDEX_FILE}")
    faiss.write_index(index, str(INDEX_FILE))

    print(f"Saving metadata to: {METADATA_FILE}")
    with open(METADATA_FILE, "wb") as f:
        pickle.dump(metadata, f)

    # Print file sizes
    idx_size = INDEX_FILE.stat().st_size / 1024
    meta_size = METADATA_FILE.stat().st_size / 1024
    print(f"  Index size: {idx_size:.1f} KB")
    print(f"  Metadata size: {meta_size:.1f} KB")


def validate_index(index: faiss.IndexFlatIP, metadata: dict, model: SentenceTransformer):
    """Run validation queries to verify the index works correctly."""
    print("\n===== INDEX VALIDATION =====")

    test_queries = [
        ("Java developer assessment", ["Java"]),
        ("personality test for executives", ["Personality", "OPQ", "Behavior"]),
        ("communication skills entry level", ["communication", "entry"]),
        ("cognitive ability test", ["Ability", "Aptitude", "cognitive", "Verify"]),
        ("short assessment under 10 minutes", []),  # Just check it returns results
    ]

    for query, expected_keywords in test_queries:
        # Embed query
        q_emb = model.encode([query], normalize_embeddings=True).astype(np.float32)

        # Search
        scores, indices = index.search(q_emb, 5)

        print(f"\n  Query: '{query}'")
        print(f"  Top-5 results:")
        match_found = False
        for rank, (score, idx) in enumerate(zip(scores[0], indices[0])):
            item = metadata[idx]
            name = item["name"]
            codes = item.get("type_codes", [])
            print(f"    {rank+1}. [{score:.4f}] {name} (codes: {codes})")

            # Check if any expected keyword appears in name or description
            item_text = f"{name} {item.get('description', '')}".lower()
            for kw in expected_keywords:
                if kw.lower() in item_text:
                    match_found = True

        if expected_keywords:
            status = "✅" if match_found else "⚠️"
            print(f"  {status} Expected keyword match: {match_found}")

    print("\n✅ INDEX VALIDATION COMPLETE")


def main():
    t_start = time.time()

    # Load catalog
    catalog = load_catalog()

    # Load embedding model
    print(f"\nLoading embedding model: {EMBEDDING_MODEL}")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print(f"  Model loaded (dim={model.get_sentence_embedding_dimension()})")

    # Build embeddings
    embeddings = build_embeddings(catalog, model)

    # Build FAISS index
    index = build_faiss_index(embeddings)

    # Build metadata dict: index_id → catalog entry
    metadata = {i: item for i, item in enumerate(catalog)}

    # Save
    save_index(index, metadata)

    # Validate
    validate_index(index, metadata, model)

    elapsed = time.time() - t_start
    print(f"\n🎉 Build complete in {elapsed:.1f}s")


if __name__ == "__main__":
    main()
