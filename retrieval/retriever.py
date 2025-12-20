"""
STEP 4: Semantic Retriever (Top-K + Scoring)

Retrieves high-quality medical evidence chunks from FAISS vector index.
Designed for citation-grounded answer generation with precision and reproducibility.

Key Features:
- Deterministic exact search (IndexFlatIP)
- Cosine similarity via inner product on normalized vectors
- Top-K retrieval (default k=6, configurable)
- No LLM calls, no text modification, no re-ranking

Query Encoding Rule (MANDATORY):
When embedding queries, prepend: "Represent this question for retrieving relevant medical documents: "

Input: User query (raw string)
Output: Top-K documents with scores and metadata

Design choices:
- Load index/metadata once at initialization (efficiency)
- GPU acceleration for query encoding if available
- Deterministic output (same query → same results)
- Low-confidence flagging if top score < 0.2
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np
import pickle
import faiss
from sentence_transformers import SentenceTransformer
import torch


# Mandatory query instruction prefix for BGE model
QUERY_INSTRUCTION = "Represent this question for retrieving relevant medical documents: "

# Model configuration (must match STEP 2)
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
EMBEDDING_DIM = 1024

# Default retrieval parameters
DEFAULT_TOP_K = 6
LOW_CONFIDENCE_THRESHOLD = 0.2


class MedicalRetriever:
    """
    Semantic retriever for medical knowledge chunks.
    
    Loads FAISS index and metadata at initialization.
    Provides deterministic top-K retrieval with cosine similarity scoring.
    """
    
    def __init__(
        self,
        index_path: str = "retrieval/index.faiss",
        metadata_path: str = "retrieval/metadata_lookup.pkl",
        model_name: str = EMBEDDING_MODEL
    ):
        """
        Initialize retriever with FAISS index and embedding model.
        
        Args:
            index_path: Path to serialized FAISS index
            metadata_path: Path to metadata lookup pickle
            model_name: Sentence transformer model name
        """
        print(f"Initializing MedicalRetriever...")
        
        # Load FAISS index
        index_path = Path(index_path)
        if not index_path.exists():
            raise FileNotFoundError(f"Index not found: {index_path}")
        
        self.index = faiss.read_index(str(index_path))
        print(f"[OK] Loaded FAISS index: {self.index.ntotal} documents")
        
        # Load metadata lookup
        metadata_path = Path(metadata_path)
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        with open(metadata_path, "rb") as f:
            self.metadata_lookup = pickle.load(f)
        print(f"[OK] Loaded metadata: {len(self.metadata_lookup)} entries")
        
        # Validate consistency
        if self.index.ntotal != len(self.metadata_lookup):
            raise ValueError(
                f"Index size ({self.index.ntotal}) != metadata size ({len(self.metadata_lookup)})"
            )
        
        # Load embedding model with GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading embedding model: {model_name} (device: {device})")
        self.model = SentenceTransformer(model_name, device=device)
        print(f"[OK] Model loaded")
        
        print("[OK] MedicalRetriever ready\n")
    
    def encode_query(self, query: str) -> np.ndarray:
        """
        Encode user query with mandatory instruction prefix.
        
        CRITICAL: Always prepends the query instruction for optimal retrieval.
        Returns L2-normalized embedding vector.
        
        Args:
            query: Raw user query (unmodified)
        
        Returns:
            Normalized query embedding [1, 1024] float32
        """
        # Prepend mandatory instruction (per BGE model requirements)
        prefixed_query = QUERY_INSTRUCTION + query
        
        # Encode with normalization (cosine similarity requirement)
        embedding = self.model.encode(
            [prefixed_query],
            normalize_embeddings=True,  # L2 normalization for cosine similarity
            convert_to_numpy=True,
            show_progress_bar=False
        )
        
        return embedding.astype(np.float32)
    
    def search(
        self,
        query_embedding: np.ndarray,
        k: int = DEFAULT_TOP_K
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Perform FAISS search on query embedding.
        
        Args:
            query_embedding: Normalized query vector [1, d]
            k: Number of results to retrieve
        
        Returns:
            Tuple of (distances [1, k], indices [1, k])
            Distances are cosine similarities (higher = more similar)
        """
        # FAISS IndexFlatIP returns inner product = cosine similarity (normalized vectors)
        distances, indices = self.index.search(query_embedding, k)
        return distances, indices
    
    def map_results_to_documents(
        self,
        indices: np.ndarray,
        distances: np.ndarray
    ) -> List[Dict[str, Any]]:
        """
        Map FAISS results to document metadata with scores.
        
        Preserves document text verbatim.
        Returns results in strict output format.
        
        Args:
            indices: FAISS result indices [1, k]
            distances: Similarity scores [1, k]
        
        Returns:
            List of documents with format:
            {
                "id": str,
                "text": str (original, unmodified),
                "score": float,
                "metadata": {
                    "topic": str,
                    "source": str,
                    "source_type": str
                }
            }
        """
        results = []
        
        for idx, score in zip(indices[0], distances[0]):
            idx = int(idx)  # Convert numpy int to Python int
            score = float(score)  # Convert numpy float to Python float
            
            # Retrieve document metadata
            doc = self.metadata_lookup[idx]
            
            # Build result in strict format (per spec)
            result = {
                "id": doc["id"],
                "text": doc["text"],  # Preserve verbatim (no modification)
                "score": round(score, 6),  # Round for cleaner output
                "metadata": {
                    "topic": doc["topic"],
                    "source": doc["source"],
                    "source_type": doc["source_type"]
                }
            }
            results.append(result)
        
        return results
    
    def retrieve(
        self,
        query: str,
        k: int = DEFAULT_TOP_K,
        return_scores: bool = True
    ) -> List[Dict[str, Any]]:
        """
        End-to-end retrieval: encode query, search, and return top-K documents.
        
        This is the main interface for retrieval.
        
        Args:
            query: Raw user query string (not modified)
            k: Number of documents to retrieve (default: 6)
            return_scores: Include similarity scores (always True per spec)
        
        Returns:
            List of top-K documents sorted by score (descending)
            Each document contains: id, text, score, metadata
            
        Example:
            >>> retriever = MedicalRetriever()
            >>> results = retriever.retrieve("What are symptoms of diabetes?", k=5)
            >>> print(results[0]["id"], results[0]["score"])
        """
        if not query or not query.strip():
            raise ValueError("Query cannot be empty")
        
        # 1. Encode query with instruction prefix
        query_embedding = self.encode_query(query)
        
        # 2. Search FAISS index
        distances, indices = self.search(query_embedding, k=k)
        
        # 3. Map to documents
        results = self.map_results_to_documents(indices, distances)
        
        # 4. Sort by score (descending) - should already be sorted, but ensure
        results = sorted(results, key=lambda x: x["score"], reverse=True)
        
        # 5. Optional: Flag low-confidence results
        if results and results[0]["score"] < LOW_CONFIDENCE_THRESHOLD:
            print(f"⚠ Low confidence: top score = {results[0]['score']:.3f}")
        
        return results
    
    def batch_retrieve(
        self,
        queries: List[str],
        k: int = DEFAULT_TOP_K
    ) -> List[List[Dict[str, Any]]]:
        """
        Retrieve for multiple queries (useful for evaluation).
        
        Args:
            queries: List of query strings
            k: Number of results per query
        
        Returns:
            List of result lists (one per query)
        """
        return [self.retrieve(q, k=k) for q in queries]


def load_retriever(
    index_path: str = "retrieval/index.faiss",
    metadata_path: str = "retrieval/metadata_lookup.pkl"
) -> MedicalRetriever:
    """
    Convenience function to load a retriever instance.
    
    Args:
        index_path: Path to FAISS index
        metadata_path: Path to metadata lookup
    
    Returns:
        Initialized MedicalRetriever
    """
    return MedicalRetriever(index_path, metadata_path)


def main():
    """
    Demo: Interactive retrieval CLI for testing.
    """
    import sys
    
    print("=" * 70)
    print("Medical Knowledge Retriever - Interactive Demo")
    print("=" * 70)
    
    # Initialize retriever
    retriever = MedicalRetriever()
    
    # Example queries
    example_queries = [
        "What are the symptoms of type 2 diabetes?",
        "How is lung cancer treated?",
        "What causes high blood pressure?"
    ]
    
    print("\nExample Queries:")
    for i, q in enumerate(example_queries, 1):
        print(f"  {i}. {q}")
    
    print("\n" + "=" * 70)
    
    # Interactive loop
    while True:
        print("\nEnter a medical query (or 'quit' to exit, 'example N' for demo):")
        user_input = input("> ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ["quit", "exit", "q"]:
            break
        
        # Handle example queries
        if user_input.lower().startswith("example"):
            try:
                idx = int(user_input.split()[1]) - 1
                query = example_queries[idx]
            except (IndexError, ValueError):
                print("Invalid example number")
                continue
        else:
            query = user_input
        
        print(f"\nQuery: {query}")
        print("-" * 70)
        
        # Retrieve
        try:
            results = retriever.retrieve(query, k=5)
            
            print(f"\nTop-{len(results)} Results:\n")
            for i, doc in enumerate(results, 1):
                print(f"{i}. [{doc['score']:.4f}] {doc['id']}")
                print(f"   Topic: {doc['metadata']['topic'][:60]}...")
                print(f"   Source: {doc['metadata']['source']}")
                print(f"   Text preview: {doc['text'][:120]}...")
                print()
        
        except Exception as e:
            print(f"Error during retrieval: {e}")


if __name__ == "__main__":
    main()
