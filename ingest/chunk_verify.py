"""
Lightweight verification script for loaded documents.

This script:
- Loads validated documents
- Prints statistics on document chunks
- Verifies chunk sizes are within expected range
- No data modification
"""

from pathlib import Path
from typing import List, Dict, Any
import statistics


def count_words(text: str) -> int:
    """Count words in a string."""
    return len(text.split())


def verify_documents(documents: List[Dict[str, Any]]) -> None:
    """
    Verify and print statistics about loaded documents.
    
    Args:
        documents: List of validated document objects
    """
    if not documents:
        print("⚠ No documents to verify")
        return
    
    # Extract word counts
    word_counts = [count_words(doc["text"]) for doc in documents]
    
    total_docs = len(documents)
    min_words = min(word_counts)
    max_words = max(word_counts)
    avg_words = statistics.mean(word_counts)
    median_words = statistics.median(word_counts)
    
    print(f"\n✓ Document Verification Report")
    print(f"  Total documents: {total_docs}")
    print(f"\n  Word Count Statistics:")
    print(f"    Min:    {min_words} words")
    print(f"    Max:    {max_words} words")
    print(f"    Mean:   {avg_words:.1f} words")
    print(f"    Median: {median_words} words")
    
    # Verify all chunks meet minimum requirement (>40 words)
    below_threshold = sum(1 for wc in word_counts if wc <= 40)
    if below_threshold > 0:
        print(f"\n  ⚠ Warning: {below_threshold} documents with ≤40 words")
    else:
        print(f"\n  ✓ All documents exceed 40-word minimum")
    
    # Check metadata integrity
    metadata_issues = 0
    for i, doc in enumerate(documents):
        if "metadata" not in doc:
            print(f"  ⚠ Document {i}: missing metadata")
            metadata_issues += 1
        elif not all(k in doc["metadata"] for k in ["id", "topic", "source", "source_type"]):
            print(f"  ⚠ Document {i}: incomplete metadata")
            metadata_issues += 1
    
    if metadata_issues == 0:
        print(f"  ✓ All documents have complete metadata")
    
    # Unique topics
    topics = set(doc["metadata"]["topic"] for doc in documents if "metadata" in doc)
    print(f"\n  Unique topics: {len(topics)}")
    
    # Unique sources
    sources = set(doc["metadata"]["source"] for doc in documents if "metadata" in doc)
    print(f"  Unique sources: {len(sources)}")


def main():
    """Main entry point for verification."""
    # Import here to avoid circular dependency
    from load_clean import load_and_validate_dataset
    import sys
    
    if len(sys.argv) > 1:
        jsonl_path = sys.argv[1]
    else:
        jsonl_path = Path(__file__).parent.parent / "data" / "medical_knowledge.jsonl"
    
    print(f"Verifying dataset from: {jsonl_path}")
    
    documents, total, dropped = load_and_validate_dataset(str(jsonl_path))
    verify_documents(documents)
    
    print(f"\n  Validation Summary:")
    print(f"    Total records: {total}")
    print(f"    Valid documents: {len(documents)}")
    print(f"    Dropped records: {dropped}")
    print(f"    Retention rate: {100 * len(documents) / total:.1f}%")


if __name__ == "__main__":
    main()
