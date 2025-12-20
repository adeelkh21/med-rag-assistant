"""
Hybrid Retrieval Evaluation Script

Evaluates and compares three retrieval strategies:
1. BM25 (sparse)
2. Dense (FAISS embeddings)
3. Hybrid (BM25 + Dense fusion)

Metrics:
- Recall@K (K=1, 3, 5, 10)
- MRR (Mean Reciprocal Rank)

Output:
- Console tables (comparison summary)
- JSON results file
- CSV results file
"""

import sys
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import json
import csv
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import numpy as np

from retrieval.bm25_retriever import BM25Retriever
from retrieval.retriever import MedicalRetriever
from retrieval.hybrid_retriever import HybridRetriever
from evaluation.retrieval_benchmark import get_benchmark_dataset


def compute_recall_at_k(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k: int
) -> float:
    """
    Compute Recall@K.
    
    Recall@K = (# relevant docs in top-K) / (# total relevant docs)
    
    Args:
        retrieved_ids: List of retrieved document IDs (in rank order)
        relevant_ids: List of ground truth relevant document IDs
        k: Cutoff position
    
    Returns:
        Recall@K score (0.0 to 1.0)
    """
    if not relevant_ids:
        return 0.0
    
    # Get top-k retrieved
    top_k = set(retrieved_ids[:k])
    relevant_set = set(relevant_ids)
    
    # Count how many relevant docs are in top-k
    hits = len(top_k & relevant_set)
    
    # Recall = hits / total relevant
    recall = hits / len(relevant_set)
    return recall


def compute_mrr(
    retrieved_ids: List[str],
    relevant_ids: List[str]
) -> float:
    """
    Compute Mean Reciprocal Rank (MRR).
    
    MRR = 1 / (rank of first relevant document)
    If no relevant document found, MRR = 0
    
    Args:
        retrieved_ids: List of retrieved document IDs (in rank order)
        relevant_ids: List of ground truth relevant document IDs
    
    Returns:
        Reciprocal rank (0.0 to 1.0)
    """
    relevant_set = set(relevant_ids)
    
    # Find rank of first relevant document (1-indexed)
    for rank, doc_id in enumerate(retrieved_ids, 1):
        if doc_id in relevant_set:
            return 1.0 / rank
    
    # No relevant document found
    return 0.0


def evaluate_retriever(
    retriever,
    questions: List[Dict[str, Any]],
    k_values: List[int] = [1, 3, 5, 10],
    retrieval_k: int = 10,
    retriever_name: str = "Retriever"
) -> Dict[str, Any]:
    """
    Evaluate a retriever on benchmark questions.
    
    Args:
        retriever: Retriever instance (BM25, Dense, or Hybrid)
        questions: List of benchmark questions
        k_values: List of K values for Recall@K
        retrieval_k: Number of documents to retrieve per query
        retriever_name: Name for display
    
    Returns:
        Dictionary with metrics and per-question results
    """
    print(f"\nEvaluating {retriever_name}...")
    
    # Storage for all metrics
    all_recall_scores = {k: [] for k in k_values}
    all_mrr_scores = []
    per_question_results = []
    
    # Evaluate each question
    for question in tqdm(questions, desc=f"  {retriever_name}"):
        query = question["question"]
        relevant_ids = question["relevant_chunks"]
        
        # Retrieve documents
        try:
            results = retriever.retrieve(query, k=retrieval_k)
            retrieved_ids = [r["id"] for r in results]
        except Exception as e:
            print(f"\n    Warning: Retrieval failed for question {question['id']}: {e}")
            retrieved_ids = []
        
        # Compute metrics
        recall_scores = {}
        for k in k_values:
            recall = compute_recall_at_k(retrieved_ids, relevant_ids, k)
            recall_scores[f"recall@{k}"] = recall
            all_recall_scores[k].append(recall)
        
        mrr = compute_mrr(retrieved_ids, relevant_ids)
        all_mrr_scores.append(mrr)
        
        # Store per-question result
        per_question_results.append({
            "question_id": question["id"],
            "question": query,
            "category": question["category"],
            "relevant_count": len(relevant_ids),
            "retrieved_count": len(retrieved_ids),
            **recall_scores,
            "mrr": mrr,
            "retrieved_ids": retrieved_ids[:5]  # Top-5 for inspection
        })
    
    # Compute average metrics
    avg_metrics = {
        f"recall@{k}": np.mean(all_recall_scores[k]) for k in k_values
    }
    avg_metrics["mrr"] = np.mean(all_mrr_scores)
    
    return {
        "retriever_name": retriever_name,
        "avg_metrics": avg_metrics,
        "per_question_results": per_question_results,
        "total_questions": len(questions)
    }


def print_comparison_table(results: List[Dict[str, Any]]):
    """
    Print comparison table of all retrievers.
    
    Args:
        results: List of evaluation results for each retriever
    """
    print("\n" + "=" * 70)
    print("RETRIEVAL COMPARISON RESULTS")
    print("=" * 70)
    
    # Header
    print(f"\n{'Retriever':<20} {'R@1':<10} {'R@3':<10} {'R@5':<10} {'R@10':<10} {'MRR':<10}")
    print("-" * 70)
    
    # Rows
    for result in results:
        name = result["retriever_name"]
        metrics = result["avg_metrics"]
        
        r1 = metrics.get("recall@1", 0.0)
        r3 = metrics.get("recall@3", 0.0)
        r5 = metrics.get("recall@5", 0.0)
        r10 = metrics.get("recall@10", 0.0)
        mrr = metrics.get("mrr", 0.0)
        
        print(f"{name:<20} {r1:<10.4f} {r3:<10.4f} {r5:<10.4f} {r10:<10.4f} {mrr:<10.4f}")
    
    print("=" * 70)


def save_results_json(results: List[Dict[str, Any]], output_path: str):
    """Save evaluation results to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"\n✓ Results saved to JSON: {output_path}")


def save_results_csv(results: List[Dict[str, Any]], output_path: str):
    """Save evaluation results to CSV file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)
    
    # Summary CSV
    with open(output_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(["Retriever", "Recall@1", "Recall@3", "Recall@5", "Recall@10", "MRR"])
        
        # Data rows
        for result in results:
            name = result["retriever_name"]
            metrics = result["avg_metrics"]
            
            writer.writerow([
                name,
                f"{metrics.get('recall@1', 0.0):.4f}",
                f"{metrics.get('recall@3', 0.0):.4f}",
                f"{metrics.get('recall@5', 0.0):.4f}",
                f"{metrics.get('recall@10', 0.0):.4f}",
                f"{metrics.get('mrr', 0.0):.4f}"
            ])
    
    print(f"✓ Summary saved to CSV: {output_path}")
    
    # Detailed per-question CSV
    detailed_path = output_path.parent / f"{output_path.stem}_detailed.csv"
    
    all_rows = []
    for result in results:
        retriever_name = result["retriever_name"]
        for q_result in result["per_question_results"]:
            row = {
                "retriever": retriever_name,
                "question_id": q_result["question_id"],
                "category": q_result["category"],
                "question": q_result["question"],
                "recall@1": f"{q_result.get('recall@1', 0.0):.4f}",
                "recall@3": f"{q_result.get('recall@3', 0.0):.4f}",
                "recall@5": f"{q_result.get('recall@5', 0.0):.4f}",
                "recall@10": f"{q_result.get('recall@10', 0.0):.4f}",
                "mrr": f"{q_result.get('mrr', 0.0):.4f}",
                "relevant_count": q_result["relevant_count"],
                "retrieved_count": q_result["retrieved_count"]
            }
            all_rows.append(row)
    
    if all_rows:
        with open(detailed_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=all_rows[0].keys())
            writer.writeheader()
            writer.writerows(all_rows)
        
        print(f"✓ Detailed results saved to CSV: {detailed_path}")


def main():
    """Main evaluation pipeline."""
    print("=" * 70)
    print("HYBRID RETRIEVAL EVALUATION")
    print("=" * 70)
    
    # Load benchmark dataset
    print("\nLoading benchmark dataset...")
    benchmark = get_benchmark_dataset()
    questions = benchmark["questions"]
    print(f"✓ Loaded {len(questions)} questions")
    
    # Initialize retrievers
    print("\nInitializing retrievers...")
    
    try:
        print("\n[1/3] BM25 Retriever")
        bm25_retriever = BM25Retriever()
    except Exception as e:
        print(f"✗ Failed to initialize BM25 retriever: {e}")
        return
    
    try:
        print("\n[2/3] Dense Retriever (FAISS)")
        dense_retriever = MedicalRetriever()
    except Exception as e:
        print(f"✗ Failed to initialize Dense retriever: {e}")
        return
    
    try:
        print("\n[3/3] Hybrid Retriever")
        hybrid_retriever = HybridRetriever(
            dense_retriever=dense_retriever,
            bm25_retriever=bm25_retriever,
            alpha=0.5  # Equal weight to BM25 and Dense
        )
    except Exception as e:
        print(f"✗ Failed to initialize Hybrid retriever: {e}")
        return
    
    print("\n✓ All retrievers initialized")
    
    # Evaluation parameters
    k_values = [1, 3, 5, 10]
    retrieval_k = 10  # Retrieve top-10 for evaluation
    
    # Evaluate each retriever
    print("\n" + "=" * 70)
    print("RUNNING EVALUATIONS")
    print("=" * 70)
    
    all_results = []
    
    # BM25
    bm25_results = evaluate_retriever(
        bm25_retriever,
        questions,
        k_values=k_values,
        retrieval_k=retrieval_k,
        retriever_name="BM25 (Sparse)"
    )
    all_results.append(bm25_results)
    
    # Dense
    dense_results = evaluate_retriever(
        dense_retriever,
        questions,
        k_values=k_values,
        retrieval_k=retrieval_k,
        retriever_name="Dense (FAISS)"
    )
    all_results.append(dense_results)
    
    # Hybrid
    hybrid_results = evaluate_retriever(
        hybrid_retriever,
        questions,
        k_values=k_values,
        retrieval_k=retrieval_k,
        retriever_name="Hybrid (BM25+Dense)"
    )
    all_results.append(hybrid_results)
    
    # Print comparison table
    print_comparison_table(all_results)
    
    # Save results
    output_dir = Path("evaluation")
    output_dir.mkdir(exist_ok=True)
    
    save_results_json(
        all_results,
        output_path="evaluation/hybrid_retrieval_results.json"
    )
    
    save_results_csv(
        all_results,
        output_path="evaluation/hybrid_retrieval_results.csv"
    )
    
    print("\n" + "=" * 70)
    print("✓ EVALUATION COMPLETE")
    print("=" * 70)
    
    # Best performer analysis
    print("\nBest Performer Analysis:")
    
    for metric in ["recall@1", "recall@3", "recall@5", "recall@10", "mrr"]:
        best_score = -1
        best_retriever = None
        
        for result in all_results:
            score = result["avg_metrics"].get(metric, 0.0)
            if score > best_score:
                best_score = score
                best_retriever = result["retriever_name"]
        
        print(f"  {metric.upper():<15}: {best_retriever} ({best_score:.4f})")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
