"""
STEP 3: FAISS Vector Index Construction

Builds and persists a FAISS index over precomputed medical document embeddings.

Design choices (per spec):
- Index type: FAISS IndexFlatIP (Inner Product)
- Vectors are already L2-normalized → IP equals cosine similarity
- Exact search only (no HNSW/IVF/PQ approximations)
- Preserve full metadata for citation and evaluation

Inputs (from STEP 2):
- embeddings/embeddings.npy        (shape: [num_docs, 1024], dtype: float32, normalized)
- embeddings/metadata.pkl          (list[dict]: id, text, topic, source, source_type)

Outputs (for STEP 4):
- retrieval/index.faiss            (serialized FAISS index)
- retrieval/metadata_lookup.pkl    (mapping: index_position → document metadata)

Important:
- DO NOT re-embed or modify text
- DO NOT perform retrieval logic here
- Query encoding rule (for future steps): When embedding queries later, prepend
  "Represent this question for retrieving relevant medical documents: "
"""

from pathlib import Path
from typing import Tuple, List, Dict, Any
import numpy as np
import pickle
import faiss

EMBEDDING_DIM = 1024


def load_embeddings_and_metadata(embeddings_dir: Path) -> Tuple[np.ndarray, List[Dict[str, Any]]]:
    """
    Load precomputed embeddings and metadata from disk.

    Args:
        embeddings_dir: Directory containing embeddings.npy and metadata.pkl

    Returns:
        Tuple of (embeddings [N, 1024] float32, metadata_list)
    """
    embeddings_path = embeddings_dir / "embeddings.npy"
    metadata_path = embeddings_dir / "metadata.pkl"

    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings file not found: {embeddings_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")

    embeddings = np.load(embeddings_path)
    with open(metadata_path, "rb") as f:
        metadata_list = pickle.load(f)

    return embeddings, metadata_list


def validate_embeddings(embeddings: np.ndarray) -> None:
    """
    Validate embeddings per spec.

    - No missing vectors
    - Dimensionality is 1024
    - dtype is float32
    - Already L2-normalized (sanity check norms ≈ 1.0)
    """
    if embeddings.ndim != 2:
        raise ValueError(f"Embeddings must be 2D, got shape {embeddings.shape}")

    n, d = embeddings.shape
    if d != EMBEDDING_DIM:
        raise ValueError(f"Embedding dimension mismatch: got {d}, expected {EMBEDDING_DIM}")

    if embeddings.dtype != np.float32:
        raise ValueError(f"Embeddings dtype must be float32, got {embeddings.dtype}")

    if n == 0:
        raise ValueError("No embeddings found (n=0)")

    # Sanity check normalization (not re-normalizing, just assert)
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        raise ValueError(f"Embeddings not normalized: norm stats min={norms.min():.6f} max={norms.max():.6f}")


def build_index(embeddings: np.ndarray) -> faiss.IndexFlatIP:
    """
    Create and populate an IndexFlatIP with given embeddings.

    Inner product equals cosine similarity when vectors are normalized.
    """
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    # Preserve insertion order: FAISS stores in the order added
    index.add(embeddings)

    # Validate index size
    if index.ntotal != embeddings.shape[0]:
        raise RuntimeError(
            f"Index population failed: index.ntotal={index.ntotal} vs embeddings={embeddings.shape[0]}"
        )

    return index


def optional_sanity_check(index: faiss.IndexFlatIP, embeddings: np.ndarray) -> None:
    """
    Optionally run a tiny similarity check to ensure correctness.

    For normalized vectors, the first vector should have highest similarity with itself.
    """
    # Query with the first vector
    q = embeddings[0:1]  # shape [1, d]
    distances, indices = index.search(q, k=5)
    top_ids = indices[0].tolist()
    top_scores = distances[0].tolist()
    # For normalized vectors, self-similarity should be ~1.0
    self_in_topk = 0 in top_ids
    self_score = float(top_scores[top_ids.index(0)]) if self_in_topk else None
    
    if not self_in_topk:
        # This can happen if another document has an identical vector (tie),
        # resulting in arbitrary ordering among equals.
        print(f"⚠ Sanity check: index 0 not in top-5. Top-5: {top_ids}, scores: {[f'{s:.6f}' for s in top_scores]}")
    else:
        print(f"✓ Sanity check: self in top-5 with score {self_score:.6f}")


def build_metadata_lookup(metadata_list: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    """
    Build mapping from FAISS index position → document metadata.

    Preserves:
    - id
    - text
    - topic
    - source
    - source_type
    """
    lookup: Dict[int, Dict[str, Any]] = {}
    for i, meta in enumerate(metadata_list):
        # Metadata from STEP 2 was preserved exactly; ensure keys exist
        lookup[i] = {
            "id": meta.get("id"),
            "text": meta.get("text"),
            "topic": meta.get("topic"),
            "source": meta.get("source"),
            "source_type": meta.get("source_type"),
        }
    return lookup


def persist_outputs(index: faiss.IndexFlatIP, lookup: Dict[int, Dict[str, Any]], output_dir: Path) -> None:
    """
    Persist FAISS index and metadata lookup to disk.

    - index.faiss: serialized FAISS index
    - metadata_lookup.pkl: pickled mapping (int → dict)
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    index_path = output_dir / "index.faiss"
    faiss.write_index(index, str(index_path))
    print(f"✓ FAISS index saved: {index_path}")

    lookup_path = output_dir / "metadata_lookup.pkl"
    with open(lookup_path, "wb") as f:
        pickle.dump(lookup, f)
    print(f"✓ Metadata lookup saved: {lookup_path} (entries: {len(lookup)})")


def build_and_save_index(
    embeddings_dir: Path = Path("embeddings"),
    output_dir: Path = Path("retrieval")
) -> None:
    """
    End-to-end: load, validate, index, and persist.
    """
    print("=" * 70)
    print("STEP 3: FAISS Index Construction")
    print("=" * 70)

    # 1) Load
    print("\n[1/5] Loading embeddings and metadata...")
    embeddings, metadata_list = load_embeddings_and_metadata(embeddings_dir)
    print(f"✓ Loaded embeddings: shape={embeddings.shape}, dtype={embeddings.dtype}")
    print(f"✓ Loaded metadata: {len(metadata_list)} entries")

    # 2) Validate
    print("\n[2/5] Validating embeddings...")
    validate_embeddings(embeddings)
    print("✓ Embeddings valid (dim=1024, float32, normalized)")

    # 3) Build index
    print("\n[3/5] Building FAISS IndexFlatIP...")
    index = build_index(embeddings)
    print(f"✓ Index built: ntotal={index.ntotal}")

    # 4) Sanity check (optional)
    print("\n[4/5] Running sanity check search...")
    optional_sanity_check(index, embeddings)

    # 5) Metadata mapping
    print("\n[5/5] Building metadata lookup...")
    lookup = build_metadata_lookup(metadata_list)
    print("✓ Metadata lookup ready")

    # Persist
    persist_outputs(index, lookup, output_dir)

    print("\n" + "=" * 70)
    print("✓ STEP 3 Complete: Index & metadata persisted")
    print("=" * 70)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Build and persist FAISS index over embeddings")
    parser.add_argument(
        "--embeddings-dir", type=str, default="embeddings",
        help="Directory containing embeddings.npy and metadata.pkl"
    )
    parser.add_argument(
        "--output-dir", type=str, default="retrieval",
        help="Directory to write index.faiss and metadata_lookup.pkl"
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Only verify existing index and lookup without rebuilding"
    )

    args = parser.parse_args()

    if args.verify:
        verify_artifacts(Path(args.output_dir))
    else:
        build_and_save_index(Path(args.embeddings_dir), Path(args.output_dir))


def verify_artifacts(output_dir: Path) -> None:
    """
    Verify existing index and lookup files.
    """
    index_path = output_dir / "index.faiss"
    lookup_path = output_dir / "metadata_lookup.pkl"

    if not index_path.exists() or not lookup_path.exists():
        print("✗ Missing artifacts. Run the builder to generate them.")
        return

    index = faiss.read_index(str(index_path))
    with open(lookup_path, "rb") as f:
        lookup = pickle.load(f)

    print("\nVerification:")
    print(f"  Index: ntotal={index.ntotal}")
    print(f"  Metadata entries: {len(lookup)}")
    # Simple consistency check
    if index.ntotal != len(lookup):
        print("  ✗ Mismatch: index count vs metadata entries")
    else:
        print("  ✓ Consistent: index count matches metadata entries")


if __name__ == "__main__":
    main()
