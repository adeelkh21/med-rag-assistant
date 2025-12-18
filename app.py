"""
RAG Medical Pipeline - Phase 2
Main entry point for the RAG system.
"""

from pathlib import Path
from ingest.load_clean import load_and_validate_dataset


def main():
    """Initialize the RAG pipeline."""
    print("RAG Medical Pipeline - Phase 2")
    print("=" * 50)
    
    # Load dataset
    data_path = Path(__file__).parent / "data" / "medical_knowledge.jsonl"
    print(f"\nLoading medical knowledge dataset...")
    documents, total, dropped = load_and_validate_dataset(str(data_path))
    
    print(f"✓ Dataset loaded: {len(documents)} valid documents")
    print(f"  Dropped records: {dropped}")
    
    return documents


if __name__ == "__main__":
    main()
