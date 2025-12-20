"""
Hybrid Retriever (BM25 + Dense Fusion)

Combines sparse (BM25) and dense (FAISS) retrieval strategies.
Uses union-based merging with score normalization.

Strategy:
1. Retrieve top-k from BM25
2. Retrieve top-k from Dense (FAISS)
3. Merge results (union of chunk IDs)
4. Normalize scores to [0, 1]
5. Deduplicate by chunk ID (keep highest score)
6. Sort by final score
7. Return top-k

Features:
- Wraps existing retrievers (no modifications)
- Configurable fusion weights (alpha for BM25, 1-alpha for Dense)
- Min-max score normalization
- Deterministic results
"""

from typing import List, Dict, Any, Optional, Set
from pathlib import Path
import numpy as np

from retrieval.retriever import MedicalRetriever
from retrieval.bm25_retriever import BM25Retriever


def normalize_scores(scores: List[float]) -> List[float]:
    """
    Normalize scores to [0, 1] using min-max normalization.
    
    Args:
        scores: List of raw scores
    
    Returns:
        List of normalized scores in [0, 1]
    """
    if not scores:
        return []
    
    scores_array = np.array(scores)
    min_score = scores_array.min()
    max_score = scores_array.max()
    
    # Avoid division by zero
    if max_score == min_score:
        return [1.0] * len(scores)
    
    normalized = (scores_array - min_score) / (max_score - min_score)
    return normalized.tolist()


class HybridRetriever:
    """
    Hybrid retriever combining BM25 (sparse) and FAISS (dense) retrieval.
    
    Uses union-based fusion with score normalization.
    Compatible with MedicalRetriever interface.
    """
    
    def __init__(
        self,
        dense_retriever: Optional[MedicalRetriever] = None,
        bm25_retriever: Optional[BM25Retriever] = None,
        alpha: float = 0.5,
        jsonl_path: str = "data/medical_knowledge.jsonl"
    ):
        """
        Initialize hybrid retriever.
        
        Args:
            dense_retriever: MedicalRetriever instance (creates new if None)
            bm25_retriever: BM25Retriever instance (creates new if None)
            alpha: Weight for BM25 scores (0.0-1.0). Dense weight = 1 - alpha
            jsonl_path: Path to medical knowledge JSONL (for BM25)
        """
        print("Initializing HybridRetriever...")
        
        # Validate alpha
        if not 0.0 <= alpha <= 1.0:
            raise ValueError(f"Alpha must be in [0, 1], got {alpha}")
        
        self.alpha = alpha
        print(f"  Fusion weights: BM25={alpha:.2f}, Dense={1-alpha:.2f}")
        
        # Initialize dense retriever
        if dense_retriever is None:
            print("  Loading dense retriever (FAISS)...")
            self.dense_retriever = MedicalRetriever()
        else:
            self.dense_retriever = dense_retriever
            print("  [OK] Using provided dense retriever")
        
        # Initialize BM25 retriever
        if bm25_retriever is None:
            print("  Loading BM25 retriever...")
            self.bm25_retriever = BM25Retriever(jsonl_path=jsonl_path)
        else:
            self.bm25_retriever = bm25_retriever
            print("  [OK] Using provided BM25 retriever")
        
        print("[OK] HybridRetriever ready\n")
    
    def retrieve(
        self,
        query: str,
        k: int = 6,
        retrieve_k_multiplier: float = 2.0
    ) -> List[Dict[str, Any]]:
        """
        Retrieve documents using hybrid fusion.
        
        Args:
            query: User query string
            k: Number of final results to return
            retrieve_k_multiplier: Multiplier for retrieval k (e.g., 2.0 retrieves 2*k from each)
        
        Returns:
            List of top-k documents with fused scores:
            [
                {
                    "id": str,
                    "text": str,
                    "score": float (normalized, fused),
                    "metadata": dict,
                    "source_method": str ("bm25", "dense", or "both")
                },
                ...
            ]
        """
        # Retrieve from both methods (retrieve more to ensure good coverage after fusion)
        retrieve_k = int(k * retrieve_k_multiplier)
        
        # BM25 retrieval
        bm25_results = self.bm25_retriever.retrieve(query, k=retrieve_k)
        
        # Dense retrieval
        dense_results = self.dense_retriever.retrieve(query, k=retrieve_k)
        
        # Normalize scores separately for each method
        bm25_scores = [r["score"] for r in bm25_results]
        dense_scores = [r["score"] for r in dense_results]
        
        bm25_normalized = normalize_scores(bm25_scores)
        dense_normalized = normalize_scores(dense_scores)
        
        # Create lookup dictionaries with normalized scores
        bm25_lookup = {
            bm25_results[i]["id"]: {
                "score": bm25_normalized[i],
                "doc": bm25_results[i]
            }
            for i in range(len(bm25_results))
        }
        
        dense_lookup = {
            dense_results[i]["id"]: {
                "score": dense_normalized[i],
                "doc": dense_results[i]
            }
            for i in range(len(dense_results))
        }
        
        # Merge: union of all chunk IDs
        all_chunk_ids: Set[str] = set(bm25_lookup.keys()) | set(dense_lookup.keys())
        
        # Compute fused scores
        fused_results = []
        for chunk_id in all_chunk_ids:
            # Get normalized scores (0 if not present)
            bm25_score = bm25_lookup[chunk_id]["score"] if chunk_id in bm25_lookup else 0.0
            dense_score = dense_lookup[chunk_id]["score"] if chunk_id in dense_lookup else 0.0
            
            # Weighted fusion
            fused_score = self.alpha * bm25_score + (1 - self.alpha) * dense_score
            
            # Determine source method
            in_bm25 = chunk_id in bm25_lookup
            in_dense = chunk_id in dense_lookup
            
            if in_bm25 and in_dense:
                source_method = "both"
                # Use dense doc as primary (more metadata)
                doc = dense_lookup[chunk_id]["doc"]
            elif in_bm25:
                source_method = "bm25"
                doc = bm25_lookup[chunk_id]["doc"]
            else:
                source_method = "dense"
                doc = dense_lookup[chunk_id]["doc"]
            
            # Build result entry
            fused_results.append({
                "id": chunk_id,
                "text": doc["text"],
                "score": fused_score,
                "metadata": doc.get("metadata", {}),
                "source_method": source_method,
                "bm25_score": bm25_score,
                "dense_score": dense_score
            })
        
        # Sort by fused score (descending)
        fused_results.sort(key=lambda x: x["score"], reverse=True)
        
        # Return top-k
        return fused_results[:k]
    
    def get_retrievers(self) -> tuple:
        """Get underlying retrievers for inspection."""
        return self.bm25_retriever, self.dense_retriever


def create_hybrid_retriever(
    alpha: float = 0.5,
    jsonl_path: str = "data/medical_knowledge.jsonl"
) -> HybridRetriever:
    """
    Create a hybrid retriever instance.
    
    Args:
        alpha: Weight for BM25 scores (0.0-1.0)
        jsonl_path: Path to medical knowledge JSONL
    
    Returns:
        HybridRetriever instance
    """
    return HybridRetriever(alpha=alpha, jsonl_path=jsonl_path)


# Test/demo code
if __name__ == "__main__":
    print("=" * 70)
    print("Hybrid Retriever Demo")
    print("=" * 70)
    
    # Initialize hybrid retriever
    try:
        retriever = create_hybrid_retriever(alpha=0.5)
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        exit(1)
    
    # Test queries
    test_queries = [
        "What are the symptoms of diabetes?",
        "How is lung cancer treated?",
        "Risk factors for heart disease"
    ]
    
    print("\n" + "=" * 70)
    print("Test Queries (Hybrid Retrieval)")
    print("=" * 70)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 70)
        
        results = retriever.retrieve(query, k=5)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. [{result['id']}] (Score: {result['score']:.4f})")
            print(f"   Source: {result['source_method']}")
            print(f"   BM25: {result['bm25_score']:.4f}, Dense: {result['dense_score']:.4f}")
            
            metadata = result.get('metadata', {})
            topic = metadata.get('topic', 'N/A') if isinstance(metadata, dict) else 'N/A'
            print(f"   Topic: {topic}")
            print(f"   Text: {result['text'][:100]}...")
    
    print("\n" + "=" * 70)
    print("✓ Hybrid Retriever Demo Complete")
    print("=" * 70)
