"""
BM25 Retriever for Medical Knowledge Base

Implements sparse retrieval using BM25 algorithm.
Complements dense retrieval with keyword matching.

Features:
- BM25 scoring via rank-bm25 library
- Tokenized document index
- Metadata preservation (id, text, topic, source, source_type)
- Top-K retrieval with scores

Design:
- Load documents from JSONL
- Tokenize with simple whitespace + lowercase
- Build BM25 index at initialization
- Query returns top-k with normalized scores
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
import orjson
from rank_bm25 import BM25Okapi
import numpy as np


def simple_tokenize(text: str) -> List[str]:
    """
    Simple tokenizer for BM25.
    
    Splits on whitespace, lowercases, removes punctuation.
    
    Args:
        text: Input text string
    
    Returns:
        List of tokens
    """
    # Lowercase and split on whitespace
    tokens = text.lower().split()
    
    # Remove basic punctuation
    cleaned_tokens = []
    for token in tokens:
        # Strip common punctuation from ends
        cleaned = token.strip('.,!?;:()[]{}"\'-')
        if cleaned:  # Keep non-empty tokens
            cleaned_tokens.append(cleaned)
    
    return cleaned_tokens


def load_documents_from_jsonl(jsonl_path: str) -> List[Dict[str, Any]]:
    """
    Load documents from JSONL file.
    
    Args:
        jsonl_path: Path to medical_knowledge.jsonl
    
    Returns:
        List of document dictionaries with id, text, topic, source, source_type
    """
    path = Path(jsonl_path)
    if not path.exists():
        raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")
    
    documents = []
    with open(path, 'rb') as f:
        for line in f:
            if line.strip():
                doc = orjson.loads(line)
                documents.append({
                    "id": doc["id"],
                    "text": doc["text"],
                    "topic": doc.get("topic", ""),
                    "source": doc.get("source", ""),
                    "source_type": doc.get("source_type", "")
                })
    
    return documents


class BM25Retriever:
    """
    BM25-based sparse retriever for medical documents.
    
    Uses keyword matching with BM25 scoring algorithm.
    Compatible with MedicalRetriever interface.
    """
    
    def __init__(self, jsonl_path: str = "data/medical_knowledge.jsonl"):
        """
        Initialize BM25 retriever.
        
        Args:
            jsonl_path: Path to medical knowledge JSONL file
        """
        print("Initializing BM25Retriever...")
        
        # Load documents
        print(f"  Loading documents from {jsonl_path}...")
        self.documents = load_documents_from_jsonl(jsonl_path)
        print(f"  [OK] Loaded {len(self.documents)} documents")
        
        # Tokenize all document texts
        print("  Tokenizing documents...")
        self.tokenized_corpus = [
            simple_tokenize(doc["text"]) for doc in self.documents
        ]
        
        # Build BM25 index
        print("  Building BM25 index...")
        self.bm25 = BM25Okapi(self.tokenized_corpus)
        print("  [OK] BM25 index ready")
        
        print("[OK] BM25Retriever initialized\n")
    
    def retrieve(self, query: str, k: int = 6) -> List[Dict[str, Any]]:
        """
        Retrieve top-k documents using BM25 scoring.
        
        Args:
            query: User query string
            k: Number of documents to retrieve
        
        Returns:
            List of top-k documents with scores and metadata:
            [
                {
                    "id": str,
                    "text": str,
                    "score": float,
                    "metadata": {
                        "topic": str,
                        "source": str,
                        "source_type": str
                    }
                },
                ...
            ]
        """
        # Tokenize query
        tokenized_query = simple_tokenize(query)
        
        # Get BM25 scores for all documents
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top-k indices
        top_k_indices = np.argsort(scores)[::-1][:k]
        
        # Build result documents
        results = []
        for idx in top_k_indices:
            doc = self.documents[idx]
            results.append({
                "id": doc["id"],
                "text": doc["text"],
                "score": float(scores[idx]),
                "metadata": {
                    "topic": doc["topic"],
                    "source": doc["source"],
                    "source_type": doc["source_type"]
                }
            })
        
        return results
    
    def get_document_count(self) -> int:
        """Get total number of indexed documents."""
        return len(self.documents)


# Standalone convenience function
def create_bm25_retriever(
    jsonl_path: str = "data/medical_knowledge.jsonl"
) -> BM25Retriever:
    """
    Create a BM25 retriever instance.
    
    Args:
        jsonl_path: Path to medical knowledge JSONL
    
    Returns:
        BM25Retriever instance
    """
    return BM25Retriever(jsonl_path=jsonl_path)


# Test/demo code
if __name__ == "__main__":
    print("=" * 70)
    print("BM25 Retriever Demo")
    print("=" * 70)
    
    # Initialize retriever
    try:
        retriever = create_bm25_retriever()
        print(f"Indexed {retriever.get_document_count()} documents")
    except Exception as e:
        print(f"✗ Failed to initialize: {e}")
        exit(1)
    
    # Test queries
    test_queries = [
        "What are the symptoms of lung cancer?",
        "How is diabetes treated?",
        "Risk factors for heart disease"
    ]
    
    print("\n" + "=" * 70)
    print("Test Queries")
    print("=" * 70)
    
    for query in test_queries:
        print(f"\nQuery: {query}")
        print("-" * 70)
        
        results = retriever.retrieve(query, k=3)
        
        for i, result in enumerate(results, 1):
            print(f"\n{i}. [{result['id']}] (Score: {result['score']:.4f})")
            print(f"   Topic: {result['metadata']['topic']}")
            print(f"   Text: {result['text'][:150]}...")
    
    print("\n" + "=" * 70)
    print("✓ BM25 Retriever Demo Complete")
    print("=" * 70)
