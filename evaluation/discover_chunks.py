"""
Helper script to discover actual chunk IDs for evaluation dataset.

Runs retrieval for each question and shows top retrieved chunk IDs.
Use this to populate evaluation_dataset.json with real chunk IDs.
"""

import sys
from pathlib import Path

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from retrieval.retriever import MedicalRetriever
import json


def discover_chunk_ids():
    """Retrieve top chunks for each evaluation question."""
    
    # Sample questions from different categories
    questions = [
        "What are the symptoms of type 2 diabetes?",
        "How can type 2 diabetes be prevented?",
        "What causes high blood pressure?",
        "What are the risk factors for heart disease?",
        "How is lung cancer diagnosed?",
        "What are the different types of stroke?",
        "What lifestyle changes help prevent heart disease?",
        "What are the symptoms of a stroke?",
        "What is cholesterol and why is it important?",
        "What are the complications of uncontrolled diabetes?",
    ]
    
    print("Initializing retriever...")
    retriever = MedicalRetriever()
    print("✓ Ready\n")
    
    print("=" * 70)
    print("DISCOVERING CHUNK IDs FOR EVALUATION DATASET")
    print("=" * 70)
    print()
    
    discovered = []
    
    for i, question in enumerate(questions, 1):
        print(f"[{i}/{len(questions)}] {question}")
        print()
        
        # Retrieve top-8 documents
        docs = retriever.retrieve(question, k=8)
        
        # Show top chunk IDs
        chunk_ids = [doc["id"] for doc in docs[:5]]
        scores = [doc["score"] for doc in docs[:5]]
        
        print("  Top 5 chunks:")
        for cid, score in zip(chunk_ids, scores):
            print(f"    {cid:<30} (score: {score:.4f})")
        
        discovered.append({
            "question": question,
            "suggested_chunk_ids": chunk_ids[:3],  # Top 3 as ground truth
            "all_top_ids": chunk_ids
        })
        
        print()
    
    # Save suggestions
    output_path = Path(__file__).parent / "discovered_chunks.json"
    with open(output_path, 'w') as f:
        json.dump(discovered, f, indent=2)
    
    print(f"✓ Saved chunk ID suggestions to: {output_path}")
    print("\nUse these IDs to update evaluation_dataset.json")


if __name__ == "__main__":
    discover_chunk_ids()
