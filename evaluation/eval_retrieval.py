"""
STEP 10: Evaluation Pipeline for Medical RAG System

Comprehensive evaluation including:
1. Retrieval Quality (Recall@K)
2. Answer Faithfulness
3. Safety & Scope Compliance
4. Citation Quality
5. Answer Quality

Usage:
    python evaluation/eval_retrieval.py
    python evaluation/eval_retrieval.py --quick  # Run subset only
"""

import json
import sys
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import defaultdict
import time

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from retrieval.retriever import MedicalRetriever
from generation.answer_generator import MedicalAnswerGenerator
from generation.safety_filter import filter_query


class RetrievalEvaluator:
    """Evaluate retrieval quality using Recall@K metrics."""
    
    def __init__(self, retriever: MedicalRetriever):
        self.retriever = retriever
        self.results = []
    
    def evaluate_query(
        self,
        question: str,
        expected_chunk_ids: List[str],
        k_values: List[int] = [5, 8]
    ) -> Dict[str, Any]:
        """
        Evaluate retrieval for a single query.
        
        Args:
            question: Test question
            expected_chunk_ids: Ground truth relevant chunk IDs
            k_values: List of K values to test
        
        Returns:
            Dict with recall scores and retrieved IDs
        """
        # Retrieve documents
        max_k = max(k_values)
        retrieved_docs = self.retriever.retrieve(question, k=max_k)
        retrieved_ids = [doc["id"] for doc in retrieved_docs]
        
        # Calculate Recall@K for each K
        recalls = {}
        for k in k_values:
            top_k_ids = set(retrieved_ids[:k])
            expected_set = set(expected_chunk_ids)
            
            # Recall@K = 1 if any expected chunk found in top-K, else 0
            recalls[f"recall@{k}"] = 1.0 if top_k_ids & expected_set else 0.0
        
        # Find which expected chunks were found (if any)
        found_chunks = [cid for cid in expected_chunk_ids if cid in retrieved_ids]
        missing_chunks = [cid for cid in expected_chunk_ids if cid not in retrieved_ids]
        
        result = {
            "question": question,
            "expected_chunks": expected_chunk_ids,
            "retrieved_ids": retrieved_ids,
            "found_chunks": found_chunks,
            "missing_chunks": missing_chunks,
            "top_score": retrieved_docs[0]["score"] if retrieved_docs else 0.0,
            **recalls
        }
        
        self.results.append(result)
        return result
    
    def compute_metrics(self) -> Dict[str, Any]:
        """Compute aggregate retrieval metrics."""
        if not self.results:
            return {}
        
        # Average recall scores
        k_values = [key for key in self.results[0].keys() if key.startswith("recall@")]
        metrics = {}
        
        for k_key in k_values:
            avg_recall = sum(r[k_key] for r in self.results) / len(self.results)
            metrics[k_key] = avg_recall
        
        # Failed queries
        failed_queries = [r for r in self.results if all(r[k] == 0.0 for k in k_values)]
        metrics["failed_count"] = len(failed_queries)
        metrics["failed_rate"] = len(failed_queries) / len(self.results)
        metrics["total_queries"] = len(self.results)
        
        return metrics
    
    def get_failed_queries(self) -> List[Dict[str, Any]]:
        """Get list of queries where retrieval completely failed."""
        k_values = [key for key in self.results[0].keys() if key.startswith("recall@")]
        return [r for r in self.results if all(r[k] == 0.0 for k in k_values)]


class FaithfulnessEvaluator:
    """Evaluate answer faithfulness using LLM-based judge."""
    
    def __init__(self, generator: MedicalAnswerGenerator):
        self.generator = generator
        self.results = []
    
    def evaluate_answer(
        self,
        question: str,
        answer: str,
        retrieved_docs: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Evaluate faithfulness of a generated answer.
        
        Uses LLM to judge if answer is grounded in context.
        
        Args:
            question: Original question
            answer: Generated answer
            retrieved_docs: Documents retrieved for this question
        
        Returns:
            Dict with faithfulness scores
        """
        from generation.llm_client import GroqClient
        from generation.prompts import get_mandatory_disclaimer
        
        # Build context summary
        context_text = "\n\n".join([
            f"[{doc['id']}] {doc['text'][:200]}..."
            for doc in retrieved_docs
        ])
        
        # Remove disclaimer for evaluation
        disclaimer = get_mandatory_disclaimer()
        answer_without_disclaimer = answer.replace(disclaimer, "").strip()
        
        # Build faithfulness evaluation prompt
        eval_prompt = f"""You are evaluating if an answer is faithful to the provided context.

Context:
{context_text}

Question: {question}

Answer: {answer_without_disclaimer}

Evaluate each factual statement in the answer:
- Supported: Statement is directly supported by context
- Unsupported: Statement is plausible but not in context
- Hallucinated: Statement contradicts context or is fabricated

Respond with JSON:
{{
  "supported_count": <number>,
  "unsupported_count": <number>,
  "hallucinated_count": <number>,
  "faithfulness_score": <0.0-1.0>,
  "verdict": "faithful" or "unfaithful",
  "explanation": "<brief explanation>"
}}"""
        
        try:
            client = GroqClient()
            eval_response = client.generate(
                system_prompt="You are a medical fact-checking assistant. Evaluate answer faithfulness objectively.",
                user_prompt=eval_prompt,
                temperature=0.0
            )
            
            # Parse JSON response
            import re
            json_match = re.search(r'\{.*\}', eval_response, re.DOTALL)
            if json_match:
                eval_data = json.loads(json_match.group())
            else:
                eval_data = {"error": "Could not parse LLM response"}
        
        except Exception as e:
            eval_data = {"error": str(e)}
        
        result = {
            "question": question,
            "answer_length": len(answer_without_disclaimer),
            "evaluation": eval_data
        }
        
        self.results.append(result)
        return result
    
    def compute_metrics(self) -> Dict[str, Any]:
        """Compute aggregate faithfulness metrics."""
        if not self.results:
            return {}
        
        successful_evals = [r for r in self.results if "error" not in r["evaluation"]]
        
        if not successful_evals:
            return {"error": "No successful faithfulness evaluations"}
        
        avg_faithfulness = sum(
            r["evaluation"].get("faithfulness_score", 0.0)
            for r in successful_evals
        ) / len(successful_evals)
        
        faithful_count = sum(
            1 for r in successful_evals
            if r["evaluation"].get("verdict") == "faithful"
        )
        
        return {
            "average_faithfulness_score": avg_faithfulness,
            "faithful_count": faithful_count,
            "faithful_rate": faithful_count / len(successful_evals),
            "total_evaluated": len(successful_evals),
            "evaluation_failures": len(self.results) - len(successful_evals)
        }


class SafetyEvaluator:
    """Evaluate safety and scope compliance."""
    
    def __init__(self, generator: MedicalAnswerGenerator):
        self.generator = generator
        self.results = []
    
    def evaluate_unsafe_query(
        self,
        question: str,
        expected_behavior: str
    ) -> Dict[str, Any]:
        """
        Test if unsafe query is properly blocked.
        
        Args:
            question: Unsafe test question
            expected_behavior: Expected system behavior (e.g., "refuse_diagnosis")
        
        Returns:
            Dict with safety compliance result
        """
        # Check safety filter
        should_proceed, refusal = filter_query(question)
        
        result = {
            "question": question,
            "expected_behavior": expected_behavior,
            "blocked_by_filter": not should_proceed,
            "refusal_message": refusal if not should_proceed else None,
            "compliant": not should_proceed  # Compliant if blocked
        }
        
        # If not blocked by filter, check if generation happens
        if should_proceed:
            gen_result = self.generator.generate_answer(question, verbose=False)
            result["generated_answer"] = gen_result.get("answer", "")[:200]
            result["compliant"] = False  # Failed to block
        
        self.results.append(result)
        return result
    
    def compute_metrics(self) -> Dict[str, Any]:
        """Compute safety compliance metrics."""
        if not self.results:
            return {}
        
        compliant_count = sum(1 for r in self.results if r["compliant"])
        blocked_count = sum(1 for r in self.results if r["blocked_by_filter"])
        
        return {
            "total_unsafe_queries": len(self.results),
            "correctly_blocked": compliant_count,
            "incorrectly_answered": len(self.results) - compliant_count,
            "compliance_rate": compliant_count / len(self.results),
            "filter_block_rate": blocked_count / len(self.results)
        }


class CitationEvaluator:
    """Evaluate citation quality and completeness."""
    
    def __init__(self):
        self.results = []
    
    def evaluate_citations(
        self,
        answer: str,
        retrieved_docs: List[Dict[str, Any]],
        validation_passed: bool,
        citations_used: List[str]
    ) -> Dict[str, Any]:
        """
        Evaluate citation quality in answer.
        
        Args:
            answer: Generated answer
            retrieved_docs: Retrieved documents
            validation_passed: Whether validation passed
            citations_used: List of citation IDs used
        
        Returns:
            Dict with citation metrics
        """
        import re
        from generation.prompts import get_mandatory_disclaimer
        
        # Remove disclaimer
        disclaimer = get_mandatory_disclaimer()
        answer_text = answer.replace(disclaimer, "").strip()
        
        # Extract citations
        citation_pattern = r'\(([A-Z0-9_]+)\)'
        found_citations = re.findall(citation_pattern, answer_text)
        
        retrieved_ids = {doc["id"] for doc in retrieved_docs}
        
        # Check for hallucinated citations
        hallucinated = [cid for cid in found_citations if cid not in retrieved_ids]
        
        # Check if answer has factual statements without citations
        sentences = [s.strip() for s in answer_text.split('.') if s.strip()]
        sentences_with_citations = [s for s in sentences if '(' in s and ')' in s]
        
        citation_coverage = (
            len(sentences_with_citations) / len(sentences)
            if sentences else 0.0
        )
        
        result = {
            "total_citations": len(found_citations),
            "unique_citations": len(set(found_citations)),
            "hallucinated_citations": hallucinated,
            "hallucinated_count": len(hallucinated),
            "validation_passed": validation_passed,
            "citation_coverage": citation_coverage,
            "has_complete_citations": validation_passed and len(hallucinated) == 0
        }
        
        self.results.append(result)
        return result
    
    def compute_metrics(self) -> Dict[str, Any]:
        """Compute aggregate citation metrics."""
        if not self.results:
            return {}
        
        complete_count = sum(1 for r in self.results if r["has_complete_citations"])
        no_hallucinations = sum(1 for r in self.results if r["hallucinated_count"] == 0)
        
        avg_coverage = sum(r["citation_coverage"] for r in self.results) / len(self.results)
        avg_citations = sum(r["total_citations"] for r in self.results) / len(self.results)
        
        return {
            "complete_citations_rate": complete_count / len(self.results),
            "no_hallucination_rate": no_hallucinations / len(self.results),
            "average_citation_coverage": avg_coverage,
            "average_citations_per_answer": avg_citations,
            "total_answers_evaluated": len(self.results)
        }


def run_evaluation(dataset_path: str = None, quick_mode: bool = False):
    """
    Run complete evaluation pipeline.
    
    Args:
        dataset_path: Path to evaluation dataset JSON
        quick_mode: If True, run on subset only (5 queries)
    """
    if dataset_path is None:
        dataset_path = Path(__file__).parent / "evaluation_dataset.json"
    
    print("=" * 70)
    print("MEDICAL RAG SYSTEM - COMPREHENSIVE EVALUATION")
    print("=" * 70)
    print(f"\nDataset: {dataset_path}")
    print(f"Mode: {'Quick (5 queries)' if quick_mode else 'Full'}")
    print()
    
    # Load evaluation dataset
    with open(dataset_path, 'r') as f:
        eval_data = json.load(f)
    
    safe_queries = eval_data["safe_queries"]
    unsafe_queries = eval_data["unsafe_queries"]
    
    if quick_mode:
        safe_queries = safe_queries[:5]
        unsafe_queries = unsafe_queries[:3]
    
    print(f"Safe queries: {len(safe_queries)}")
    print(f"Unsafe queries: {len(unsafe_queries)}")
    print()
    
    # Initialize system
    print("Initializing system...")
    retriever = MedicalRetriever()
    generator = MedicalAnswerGenerator(retriever=retriever, top_k=8)
    print("✓ System ready\n")
    
    # Initialize evaluators
    retrieval_eval = RetrievalEvaluator(retriever)
    faithfulness_eval = FaithfulnessEvaluator(generator)
    safety_eval = SafetyEvaluator(generator)
    citation_eval = CitationEvaluator()
    
    # ========== RETRIEVAL QUALITY EVALUATION ==========
    print("=" * 70)
    print("1. RETRIEVAL QUALITY EVALUATION")
    print("=" * 70)
    print()
    
    for i, item in enumerate(safe_queries, 1):
        question = item["question"]
        expected_ids = item["expected_chunk_ids"]
        
        print(f"[{i}/{len(safe_queries)}] {question[:60]}...")
        
        result = retrieval_eval.evaluate_query(question, expected_ids, k_values=[5, 8])
        
        print(f"  Recall@5: {result['recall@5']:.2f} | Recall@8: {result['recall@8']:.2f}")
        if result['found_chunks']:
            print(f"  Found: {result['found_chunks'][:2]}")
        if result['missing_chunks']:
            print(f"  Missing: {result['missing_chunks'][:2]}")
        print()
    
    retrieval_metrics = retrieval_eval.compute_metrics()
    
    print("\nRETRIEVAL METRICS:")
    print(f"  Average Recall@5: {retrieval_metrics['recall@5']:.2%}")
    print(f"  Average Recall@8: {retrieval_metrics['recall@8']:.2%}")
    print(f"  Failed queries: {retrieval_metrics['failed_count']}/{retrieval_metrics['total_queries']}")
    print()
    
    # ========== ANSWER GENERATION & FAITHFULNESS ==========
    print("=" * 70)
    print("2. ANSWER FAITHFULNESS EVALUATION")
    print("=" * 70)
    print()
    
    # Generate answers for subset
    sample_queries = safe_queries[:min(10, len(safe_queries))] if not quick_mode else safe_queries[:3]
    
    for i, item in enumerate(sample_queries, 1):
        question = item["question"]
        
        print(f"[{i}/{len(sample_queries)}] {question[:60]}...")
        
        # Generate answer
        gen_result = generator.generate_answer(question, verbose=False)
        
        if gen_result["success"]:
            print(f"  ✓ Generated ({len(gen_result['answer'])} chars)")
            
            # Evaluate faithfulness
            faith_result = faithfulness_eval.evaluate_answer(
                question,
                gen_result["answer"],
                gen_result["retrieved_docs"]
            )
            
            if "error" not in faith_result["evaluation"]:
                score = faith_result["evaluation"].get("faithfulness_score", 0)
                verdict = faith_result["evaluation"].get("verdict", "unknown")
                print(f"  Faithfulness: {score:.2f} ({verdict})")
            
            # Evaluate citations
            cit_result = citation_eval.evaluate_citations(
                gen_result["answer"],
                gen_result["retrieved_docs"],
                gen_result["validation_passed"],
                gen_result.get("citations_used", [])
            )
            
            print(f"  Citations: {cit_result['total_citations']} (coverage: {cit_result['citation_coverage']:.0%})")
        else:
            print(f"  ✗ Failed: {gen_result.get('error')}")
        
        print()
    
    faithfulness_metrics = faithfulness_eval.compute_metrics()
    
    print("\nFAITHFULNESS METRICS:")
    if "error" not in faithfulness_metrics:
        print(f"  Average score: {faithfulness_metrics['average_faithfulness_score']:.2f}")
        print(f"  Faithful answers: {faithfulness_metrics['faithful_count']}/{faithfulness_metrics['total_evaluated']}")
        print(f"  Faithful rate: {faithfulness_metrics['faithful_rate']:.2%}")
    else:
        print(f"  Error: {faithfulness_metrics['error']}")
    print()
    
    # ========== CITATION QUALITY ==========
    citation_metrics = citation_eval.compute_metrics()
    
    print("CITATION QUALITY METRICS:")
    print(f"  Complete citations rate: {citation_metrics['complete_citations_rate']:.2%}")
    print(f"  No hallucinations rate: {citation_metrics['no_hallucination_rate']:.2%}")
    print(f"  Average citation coverage: {citation_metrics['average_citation_coverage']:.2%}")
    print(f"  Average citations per answer: {citation_metrics['average_citations_per_answer']:.1f}")
    print()
    
    # ========== SAFETY COMPLIANCE ==========
    print("=" * 70)
    print("3. SAFETY & SCOPE COMPLIANCE EVALUATION")
    print("=" * 70)
    print()
    
    for i, item in enumerate(unsafe_queries, 1):
        question = item["question"]
        expected_behavior = item["expected_behavior"]
        
        print(f"[{i}/{len(unsafe_queries)}] {question[:60]}...")
        
        result = safety_eval.evaluate_unsafe_query(question, expected_behavior)
        
        if result["compliant"]:
            print(f"  ✓ Correctly blocked")
        else:
            print(f"  ✗ FAILED TO BLOCK")
        print()
    
    safety_metrics = safety_eval.compute_metrics()
    
    print("\nSAFETY COMPLIANCE METRICS:")
    print(f"  Correctly blocked: {safety_metrics['correctly_blocked']}/{safety_metrics['total_unsafe_queries']}")
    print(f"  Incorrectly answered: {safety_metrics['incorrectly_answered']}")
    print(f"  Compliance rate: {safety_metrics['compliance_rate']:.2%}")
    print()
    
    # ========== FINAL SUMMARY ==========
    print("=" * 70)
    print("FINAL EVALUATION SUMMARY")
    print("=" * 70)
    print()
    
    summary = {
        "retrieval": retrieval_metrics,
        "faithfulness": faithfulness_metrics,
        "citations": citation_metrics,
        "safety": safety_metrics
    }
    
    print("OVERALL SYSTEM PERFORMANCE:")
    print(f"  ✓ Retrieval Recall@8: {retrieval_metrics['recall@8']:.1%}")
    print(f"  ✓ Safety Compliance: {safety_metrics['compliance_rate']:.1%}")
    print(f"  ✓ Citation Quality: {citation_metrics['complete_citations_rate']:.1%}")
    if "error" not in faithfulness_metrics:
        print(f"  ✓ Answer Faithfulness: {faithfulness_metrics['faithful_rate']:.1%}")
    print()
    
    # Save results
    results_path = Path(__file__).parent / "evaluation_results.json"
    with open(results_path, 'w') as f:
        json.dump({
            "summary": summary,
            "retrieval_details": retrieval_eval.results,
            "faithfulness_details": faithfulness_eval.results,
            "safety_details": safety_eval.results,
            "citation_details": citation_eval.results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }, f, indent=2)
    
    print(f"✓ Detailed results saved to: {results_path}")
    print()
    
    return summary


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run RAG system evaluation")
    parser.add_argument("--quick", action="store_true", help="Run on subset only")
    parser.add_argument("--dataset", type=str, help="Path to evaluation dataset")
    
    args = parser.parse_args()
    
    run_evaluation(dataset_path=args.dataset, quick_mode=args.quick)
