"""
Embedding Generation & Vector Preparation (STEP 2)

This module:
- Loads validated documents from STEP 1
- Generates embeddings using BAAI/bge-large-en-v1.5
- Normalizes embedding vectors
- Persists embeddings and metadata to disk

Embedding Model: BAAI/bge-large-en-v1.5
- Optimized for retrieval tasks
- Embedding dimension: 1024
- Strong performance on factual QA and citation-based RAG

IMPORTANT - Query Encoding Rule (for future retrieval):
When embedding queries, prepend:
"Represent this question for retrieving relevant medical documents: "

This is NOT applied to document embeddings, only to query embeddings.
"""

import numpy as np
import pickle
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import sys
import torch

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
from ingest.load_clean import load_and_validate_dataset


# Model configuration (FIXED - DO NOT CHANGE)
EMBEDDING_MODEL_NAME = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM = 1024

# Query instruction for future retrieval (NOT used for document embeddings)
QUERY_INSTRUCTION = "Represent this question for retrieving relevant medical documents: "


def load_embedding_model(model_name: str = EMBEDDING_MODEL_NAME) -> SentenceTransformer:
    """
    Load the sentence transformer embedding model.
    
    Args:
        model_name: Name of the model to load
    
    Returns:
        Loaded SentenceTransformer model
    """
    # Detect device (GPU if available, otherwise CPU)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Loading embedding model: {model_name}")
    print(f"Using device: {device.upper()}")
    if device == 'cuda':
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA Version: {torch.version.cuda}")
    
    model = SentenceTransformer(model_name, device=device)
    print(f"✓ Model loaded (embedding dimension: {EMBEDDING_DIM})")
    return model


def generate_embeddings(
    documents: List[Dict[str, Any]], 
    model: SentenceTransformer,
    batch_size: int = 32,
    normalize: bool = True
) -> np.ndarray:
    """
    Generate normalized embeddings for document texts.
    
    CRITICAL: Only embeds the 'text' field, NOT metadata.
    
    Args:
        documents: List of validated documents with 'text' and 'metadata' fields
        model: Loaded SentenceTransformer model
        batch_size: Batch size for encoding
        normalize: Whether to normalize embeddings (required for cosine similarity)
    
    Returns:
        Normalized embedding matrix of shape (num_documents, embedding_dim)
    """
    print(f"\nGenerating embeddings for {len(documents)} documents...")
    
    # Extract only the text field (DO NOT embed metadata)
    texts = [doc["text"] for doc in documents]
    
    # Generate embeddings with progress bar
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=normalize,  # Normalize for cosine similarity
        convert_to_numpy=True
    )
    
    print(f"✓ Embeddings generated: shape {embeddings.shape}")
    
    # Verify normalization
    if normalize:
        norms = np.linalg.norm(embeddings, axis=1)
        print(f"  Embedding norms: min={norms.min():.4f}, max={norms.max():.4f}, mean={norms.mean():.4f}")
        assert np.allclose(norms, 1.0, atol=1e-6), "Embeddings not properly normalized"
        print(f"  ✓ All embeddings are normalized (L2 norm ≈ 1.0)")
    
    return embeddings


def extract_metadata(documents: List[Dict[str, Any]]) -> List[Dict[str, str]]:
    """
    Extract and preserve metadata exactly as-is.
    
    Args:
        documents: List of validated documents
    
    Returns:
        List of metadata dictionaries
    """
    metadata = []
    for doc in documents:
        # Preserve metadata exactly (no modifications)
        metadata.append({
            "id": doc["metadata"]["id"],
            "topic": doc["metadata"]["topic"],
            "source": doc["metadata"]["source"],
            "source_type": doc["metadata"]["source_type"],
            "text": doc["text"]  # Store original text for retrieval
        })
    return metadata


def save_embeddings_and_metadata(
    embeddings: np.ndarray,
    metadata: List[Dict[str, str]],
    output_dir: Path
) -> None:
    """
    Persist embeddings and metadata to disk.
    
    Args:
        embeddings: Normalized embedding matrix
        metadata: List of metadata dictionaries
        output_dir: Directory to save files
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save embeddings as NumPy array
    embeddings_path = output_dir / "embeddings.npy"
    np.save(embeddings_path, embeddings)
    print(f"\n✓ Embeddings saved to: {embeddings_path}")
    print(f"  Shape: {embeddings.shape}")
    print(f"  Size: {embeddings.nbytes / (1024**2):.2f} MB")
    
    # Save metadata as pickle
    metadata_path = output_dir / "metadata.pkl"
    with open(metadata_path, "wb") as f:
        pickle.dump(metadata, f)
    print(f"\n✓ Metadata saved to: {metadata_path}")
    print(f"  Documents: {len(metadata)}")
    
    # Save model configuration
    config_path = output_dir / "config.pkl"
    config = {
        "model_name": EMBEDDING_MODEL_NAME,
        "embedding_dim": EMBEDDING_DIM,
        "num_documents": len(metadata),
        "query_instruction": QUERY_INSTRUCTION,
        "normalized": True
    }
    with open(config_path, "wb") as f:
        pickle.dump(config, f)
    print(f"\n✓ Configuration saved to: {config_path}")


def load_embeddings_and_metadata(
    embeddings_dir: Path
) -> Tuple[np.ndarray, List[Dict[str, str]], Dict[str, Any]]:
    """
    Load persisted embeddings and metadata from disk.
    
    Args:
        embeddings_dir: Directory containing embeddings and metadata
    
    Returns:
        Tuple of (embeddings, metadata, config)
    """
    embeddings_path = embeddings_dir / "embeddings.npy"
    metadata_path = embeddings_dir / "metadata.pkl"
    config_path = embeddings_dir / "config.pkl"
    
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Embeddings not found: {embeddings_path}")
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata not found: {metadata_path}")
    
    embeddings = np.load(embeddings_path)
    
    with open(metadata_path, "rb") as f:
        metadata = pickle.load(f)
    
    config = {}
    if config_path.exists():
        with open(config_path, "rb") as f:
            config = pickle.load(f)
    
    return embeddings, metadata, config


def build_and_save_embeddings(
    dataset_path: str,
    output_dir: str,
    batch_size: int = 32
) -> Tuple[np.ndarray, List[Dict[str, str]]]:
    """
    Complete pipeline: Load documents, generate embeddings, and persist.
    
    Args:
        dataset_path: Path to medical_knowledge.jsonl
        output_dir: Directory to save embeddings and metadata
        batch_size: Batch size for embedding generation
    
    Returns:
        Tuple of (embeddings, metadata)
    """
    print("=" * 70)
    print("STEP 2: Embedding Generation & Vector Preparation")
    print("=" * 70)
    
    # Step 1: Load validated documents
    print("\n[1/4] Loading validated documents...")
    documents, total, dropped = load_and_validate_dataset(dataset_path)
    print(f"✓ Loaded {len(documents)} valid documents (dropped {dropped})")
    
    # Step 2: Load embedding model
    print("\n[2/4] Loading embedding model...")
    model = load_embedding_model()
    
    # Step 3: Generate embeddings
    print("\n[3/4] Generating embeddings...")
    embeddings = generate_embeddings(documents, model, batch_size=batch_size)
    
    # Step 4: Extract metadata and save
    print("\n[4/4] Extracting metadata and saving...")
    metadata = extract_metadata(documents)
    
    output_path = Path(output_dir)
    save_embeddings_and_metadata(embeddings, metadata, output_path)
    
    print("\n" + "=" * 70)
    print("✓ STEP 2 Complete: Embeddings and metadata persisted")
    print("=" * 70)
    print(f"\nOutput files:")
    print(f"  - {output_path / 'embeddings.npy'}")
    print(f"  - {output_path / 'metadata.pkl'}")
    print(f"  - {output_path / 'config.pkl'}")
    print(f"\nReady for STEP 3: Vector Index & Retriever")
    
    return embeddings, metadata


def verify_embeddings(embeddings_dir: str) -> None:
    """
    Verify that embeddings were correctly generated and saved.
    
    Args:
        embeddings_dir: Directory containing embeddings
    """
    print("Verifying embeddings...")
    
    embeddings, metadata, config = load_embeddings_and_metadata(Path(embeddings_dir))
    
    print(f"\n✓ Verification Report:")
    print(f"  Model: {config.get('model_name', 'Unknown')}")
    print(f"  Embedding dimension: {config.get('embedding_dim', 'Unknown')}")
    print(f"  Number of documents: {config.get('num_documents', len(metadata))}")
    print(f"  Embeddings shape: {embeddings.shape}")
    print(f"  Normalized: {config.get('normalized', False)}")
    print(f"  Query instruction: {config.get('query_instruction', 'Not set')}")
    
    # Verify normalization
    norms = np.linalg.norm(embeddings, axis=1)
    print(f"\n  Embedding norms:")
    print(f"    Min: {norms.min():.6f}")
    print(f"    Max: {norms.max():.6f}")
    print(f"    Mean: {norms.mean():.6f}")
    
    # Sample metadata
    print(f"\n  Sample metadata (first document):")
    print(f"    ID: {metadata[0]['id']}")
    print(f"    Topic: {metadata[0]['topic']}")
    print(f"    Source: {metadata[0]['source']}")
    print(f"    Text length: {len(metadata[0]['text'])} chars")


def main():
    """Main entry point for embedding generation."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate embeddings for medical knowledge documents")
    parser.add_argument(
        "--dataset",
        type=str,
        default="data/medical_knowledge.jsonl",
        help="Path to medical knowledge dataset"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="embeddings",
        help="Output directory for embeddings and metadata"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for embedding generation"
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="Verify existing embeddings instead of generating new ones"
    )
    
    args = parser.parse_args()
    
    if args.verify:
        verify_embeddings(args.output)
    else:
        build_and_save_embeddings(
            dataset_path=args.dataset,
            output_dir=args.output,
            batch_size=args.batch_size
        )


if __name__ == "__main__":
    main()
