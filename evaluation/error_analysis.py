"""
STEP 11: Results Analysis & Error Diagnosis

Analyzes evaluation results to identify:
- Retrieval failures
- Faithfulness issues
- Citation problems
- Error patterns
- Improvement opportunities
"""

import json
import sys
from pathlib import Path
from collections import defaultdict, Counter
from typing import Dict, List, Any


def load_evaluation_results(results_path: str = None) -> Dict[str, Any]:
    """Load evaluation results from JSON file."""
    if results_path is None:
        results_path = Path(__file__).parent / "evaluation_results.json"
    
    with open(results_path, 'r') as f:
        return json.load(f)


def analyze_retrieval_failures(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze retrieval quality and failure patterns."""
    
    print("=" * 70)
    print("RETRIEVAL FAILURE ANALYSIS")
    print("=" * 70)
    print()
    
    retrieval_details = results["retrieval_details"]
    
    # Identify failed queries (Recall@8 = 0)
    failed_queries = [r for r in retrieval_details if r["recall@8"] == 0.0]
    partial_failures = [r for r in retrieval_details if 0 < r["recall@8"] < 1.0]
    successes = [r for r in retrieval_details if r["recall@8"] == 1.0]
    
    print(f"Total Queries: {len(retrieval_details)}")
    print(f"  ✓ Success (Recall@8 = 1.0): {len(successes)}")
    print(f"  ~ Partial (0 < Recall@8 < 1.0): {len(partial_failures)}")
    print(f"  ✗ Failed (Recall@8 = 0.0): {len(failed_queries)}")
    print()
    
    if failed_queries:
        print("FAILED QUERIES:")
        print()
        for i, fail in enumerate(failed_queries[:10], 1):  # Show top 10
            print(f"{i}. {fail['question']}")
            print(f"   Expected: {fail['expected_chunks']}")
            print(f"   Top retrieved: {fail['retrieved_ids'][:3]}")
            print(f"   Top score: {fail['top_score']:.4f}")
            print()
    
    # Score distribution
    scores = [r["top_score"] for r in retrieval_details]
    avg_score = sum(scores) / len(scores)
    min_score = min(scores)
    max_score = max(scores)
    
    print("RETRIEVAL SCORE DISTRIBUTION:")
    print(f"  Average: {avg_score:.4f}")
    print(f"  Min: {min_score:.4f}")
    print(f"  Max: {max_score:.4f}")
    print()
    
    # Score bins
    score_bins = defaultdict(int)
    for score in scores:
        bin_label = f"{int(score * 10) / 10:.1f}-{int(score * 10 + 1) / 10:.1f}"
        score_bins[bin_label] += 1
    
    print("Score Distribution:")
    for bin_label in sorted(score_bins.keys()):
        count = score_bins[bin_label]
        bar = "█" * (count * 3)
        print(f"  {bin_label}: {bar} ({count})")
    print()
    
    analysis = {
        "total_queries": len(retrieval_details),
        "success_count": len(successes),
        "partial_count": len(partial_failures),
        "failed_count": len(failed_queries),
        "avg_score": avg_score,
        "failed_queries": [f["question"] for f in failed_queries],
        "score_distribution": dict(score_bins)
    }
    
    return analysis


def analyze_faithfulness_issues(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze answer faithfulness and grounding issues."""
    
    print("=" * 70)
    print("FAITHFULNESS ANALYSIS")
    print("=" * 70)
    print()
    
    faithfulness_details = results["faithfulness_details"]
    
    # Filter successful evaluations
    successful = [
        f for f in faithfulness_details
        if "error" not in f["evaluation"]
    ]
    
    if not successful:
        print("⚠ No successful faithfulness evaluations found")
        print()
        return {"error": "No data"}
    
    # Faithfulness scores
    scores = [f["evaluation"]["faithfulness_score"] for f in successful]
    avg_score = sum(scores) / len(scores)
    
    print(f"Evaluated Answers: {len(successful)}")
    print(f"Average Faithfulness Score: {avg_score:.2f}")
    print()
    
    # Categorize by verdict
    verdicts = Counter(f["evaluation"]["verdict"] for f in successful)
    
    print("VERDICT DISTRIBUTION:")
    for verdict, count in verdicts.items():
        print(f"  {verdict}: {count}")
    print()
    
    # Identify low-scoring answers
    low_scores = [f for f in successful if f["evaluation"]["faithfulness_score"] < 0.7]
    
    if low_scores:
        print(f"LOW FAITHFULNESS SCORES (< 0.7): {len(low_scores)}")
        print()
        for i, item in enumerate(low_scores[:5], 1):
            print(f"{i}. {item['question']}")
            print(f"   Score: {item['evaluation']['faithfulness_score']:.2f}")
            print(f"   Verdict: {item['evaluation']['verdict']}")
            if "explanation" in item["evaluation"]:
                print(f"   Note: {item['evaluation']['explanation'][:100]}")
            print()
    
    analysis = {
        "total_evaluated": len(successful),
        "average_score": avg_score,
        "verdicts": dict(verdicts),
        "low_score_count": len(low_scores),
        "low_score_questions": [f["question"] for f in low_scores]
    }
    
    return analysis


def analyze_citation_quality(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze citation completeness and hallucination patterns."""
    
    print("=" * 70)
    print("CITATION QUALITY ANALYSIS")
    print("=" * 70)
    print()
    
    citation_details = results["citation_details"]
    
    # Hallucination analysis
    hallucinations = [c for c in citation_details if c["hallucinated_count"] > 0]
    
    print(f"Total Answers: {len(citation_details)}")
    print(f"  ✓ No hallucinations: {len(citation_details) - len(hallucinations)}")
    print(f"  ✗ With hallucinations: {len(hallucinations)}")
    print()
    
    if hallucinations:
        print("HALLUCINATED CITATIONS:")
        print()
        all_hallucinations = []
        for item in hallucinations[:10]:
            all_hallucinations.extend(item["hallucinated_citations"])
        
        halluc_counts = Counter(all_hallucinations)
        for cid, count in halluc_counts.most_common(10):
            print(f"  {cid}: {count} times")
        print()
    
    # Citation coverage analysis
    coverages = [c["citation_coverage"] for c in citation_details]
    avg_coverage = sum(coverages) / len(coverages)
    
    print(f"CITATION COVERAGE:")
    print(f"  Average: {avg_coverage:.1%}")
    print(f"  100% coverage: {sum(1 for c in coverages if c == 1.0)} answers")
    print()
    
    # Citations per answer
    avg_citations = sum(c["total_citations"] for c in citation_details) / len(citation_details)
    print(f"Average Citations per Answer: {avg_citations:.1f}")
    print()
    
    analysis = {
        "total_answers": len(citation_details),
        "hallucination_count": len(hallucinations),
        "hallucination_rate": len(hallucinations) / len(citation_details),
        "average_coverage": avg_coverage,
        "average_citations": avg_citations,
        "hallucinated_ids": dict(Counter(
            cid for c in hallucinations for cid in c["hallucinated_citations"]
        ).most_common(10))
    }
    
    return analysis


def analyze_safety_compliance(results: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze safety filter performance."""
    
    print("=" * 70)
    print("SAFETY COMPLIANCE ANALYSIS")
    print("=" * 70)
    print()
    
    safety_details = results["safety_details"]
    
    # Compliance by category
    by_category = defaultdict(list)
    for item in safety_details:
        category = item["expected_behavior"]
        by_category[category].append(item["compliant"])
    
    print("COMPLIANCE BY CATEGORY:")
    for category, compliances in sorted(by_category.items()):
        rate = sum(compliances) / len(compliances)
        print(f"  {category}: {rate:.0%} ({sum(compliances)}/{len(compliances)})")
    print()
    
    # Failed safety checks
    failures = [s for s in safety_details if not s["compliant"]]
    
    if failures:
        print(f"SAFETY FAILURES: {len(failures)}")
        print()
        for i, fail in enumerate(failures, 1):
            print(f"{i}. {fail['question']}")
            print(f"   Expected: {fail['expected_behavior']}")
            print(f"   Blocked by filter: {fail['blocked_by_filter']}")
            if 'generated_answer' in fail:
                print(f"   Generated: {fail['generated_answer'][:80]}...")
            print()
    else:
        print("✓ No safety failures detected")
        print()
    
    analysis = {
        "total_unsafe_queries": len(safety_details),
        "compliant_count": sum(s["compliant"] for s in safety_details),
        "failure_count": len(failures),
        "compliance_by_category": {
            cat: sum(comps) / len(comps)
            for cat, comps in by_category.items()
        },
        "failed_queries": [f["question"] for f in failures]
    }
    
    return analysis


def identify_error_patterns(analyses: Dict[str, Dict[str, Any]]) -> Dict[str, List[str]]:
    """Identify common error patterns and root causes."""
    
    print("=" * 70)
    print("ERROR PATTERN IDENTIFICATION")
    print("=" * 70)
    print()
    
    patterns = {
        "retrieval_issues": [],
        "faithfulness_issues": [],
        "citation_issues": [],
        "safety_issues": [],
        "systemic_issues": []
    }
    
    # Retrieval patterns
    retrieval = analyses["retrieval"]
    if retrieval["failed_count"] > retrieval["total_queries"] * 0.3:
        patterns["retrieval_issues"].append(
            f"High retrieval failure rate: {retrieval['failed_count']}/{retrieval['total_queries']}"
        )
    
    if retrieval["avg_score"] < 0.7:
        patterns["retrieval_issues"].append(
            f"Low average retrieval score: {retrieval['avg_score']:.3f}"
        )
    
    # Faithfulness patterns
    faithfulness = analyses["faithfulness"]
    if "error" not in faithfulness:
        if faithfulness["low_score_count"] > 0:
            patterns["faithfulness_issues"].append(
                f"{faithfulness['low_score_count']} answers with low faithfulness (<0.7)"
            )
    
    # Citation patterns
    citations = analyses["citations"]
    if citations["hallucination_count"] > 0:
        patterns["citation_issues"].append(
            f"{citations['hallucination_count']} answers with hallucinated citations"
        )
    
    if citations["average_coverage"] < 0.9:
        patterns["citation_issues"].append(
            f"Low citation coverage: {citations['average_coverage']:.1%}"
        )
    
    # Safety patterns
    safety = analyses["safety"]
    if safety["failure_count"] > 0:
        patterns["safety_issues"].append(
            f"{safety['failure_count']} unsafe queries not blocked"
        )
    
    # Print patterns
    for category, issues in patterns.items():
        if issues:
            print(f"{category.upper().replace('_', ' ')}:")
            for issue in issues:
                print(f"  ⚠ {issue}")
            print()
    
    if not any(patterns.values()):
        print("✓ No significant error patterns detected")
        print()
    
    return patterns


def generate_improvement_recommendations(
    analyses: Dict[str, Dict[str, Any]],
    patterns: Dict[str, List[str]]
) -> List[Dict[str, str]]:
    """Generate actionable improvement recommendations."""
    
    print("=" * 70)
    print("IMPROVEMENT RECOMMENDATIONS")
    print("=" * 70)
    print()
    
    recommendations = []
    
    # Retrieval improvements
    if patterns["retrieval_issues"]:
        recommendations.append({
            "area": "Retrieval Quality",
            "issue": "Low recall scores or retrieval failures",
            "recommendations": [
                "Consider hybrid retrieval (dense + sparse/BM25)",
                "Tune embedding model or try different models",
                "Implement query expansion or reformulation",
                "Adjust top-K value (current: 8)",
                "Add reranking step after initial retrieval"
            ]
        })
    
    # Faithfulness improvements
    if patterns["faithfulness_issues"]:
        recommendations.append({
            "area": "Answer Faithfulness",
            "issue": "Answers not fully grounded in context",
            "recommendations": [
                "Strengthen prompt instructions for grounding",
                "Reduce LLM temperature further (current: 0.1)",
                "Add explicit instruction to refuse if context insufficient",
                "Implement sentence-level attribution",
                "Use smaller context windows to reduce hallucination"
            ]
        })
    
    # Citation improvements
    if patterns["citation_issues"]:
        recommendations.append({
            "area": "Citation Quality",
            "issue": "Incomplete or hallucinated citations",
            "recommendations": [
                "Add stricter citation format validation",
                "Include chunk IDs in prompt more prominently",
                "Implement post-processing citation validator",
                "Increase max retry attempts (current: 2)",
                "Add citation examples in few-shot prompting"
            ]
        })
    
    # Safety improvements
    if patterns["safety_issues"]:
        recommendations.append({
            "area": "Safety Compliance",
            "issue": "Some unsafe queries not blocked",
            "recommendations": [
                "Expand keyword patterns in safety filter",
                "Add ML-based safety classifier",
                "Implement multi-stage safety checks",
                "Add user intent classification",
                "Create more comprehensive unsafe query database"
            ]
        })
    
    # General improvements
    recommendations.append({
        "area": "General System",
        "issue": "Overall system optimization",
        "recommendations": [
            "Implement response caching for common queries",
            "Add query preprocessing and normalization",
            "Monitor and log all edge cases",
            "Build feedback loop for continuous improvement",
            "Create A/B testing framework for prompt variations"
        ]
    })
    
    # Print recommendations
    for rec in recommendations:
        print(f"📋 {rec['area'].upper()}")
        print(f"   Issue: {rec['issue']}")
        print("   Recommendations:")
        for i, suggestion in enumerate(rec['recommendations'], 1):
            print(f"     {i}. {suggestion}")
        print()
    
    return recommendations


def generate_final_report(
    results: Dict[str, Any],
    analyses: Dict[str, Dict[str, Any]],
    patterns: Dict[str, List[str]],
    recommendations: List[Dict[str, str]]
):
    """Generate comprehensive final evaluation report."""
    
    print("=" * 70)
    print("FINAL EVALUATION REPORT")
    print("=" * 70)
    print()
    
    summary = results["summary"]
    
    print("EXECUTIVE SUMMARY:")
    print()
    
    # Key metrics
    print("Key Performance Indicators:")
    print(f"  Retrieval Recall@8: {summary['retrieval']['recall@8']:.1%}")
    print(f"  Retrieval Recall@5: {summary['retrieval']['recall@5']:.1%}")
    print(f"  Safety Compliance: {summary['safety']['compliance_rate']:.1%}")
    print(f"  Citation Quality: {summary['citations']['complete_citations_rate']:.1%}")
    
    if "error" not in summary['faithfulness']:
        print(f"  Answer Faithfulness: {summary['faithfulness']['faithful_rate']:.1%}")
    
    print()
    
    # Overall assessment
    print("SYSTEM ASSESSMENT:")
    print()
    
    # Score system based on metrics
    scores = {
        "Retrieval": summary['retrieval']['recall@8'],
        "Safety": summary['safety']['compliance_rate'],
        "Citations": summary['citations']['complete_citations_rate']
    }
    
    avg_score = sum(scores.values()) / len(scores)
    
    if avg_score >= 0.9:
        grade = "A (Excellent)"
        status = "Production-ready"
    elif avg_score >= 0.8:
        grade = "B (Good)"
        status = "Near production-ready with minor improvements"
    elif avg_score >= 0.7:
        grade = "C (Adequate)"
        status = "Requires improvements before production"
    else:
        grade = "D (Needs Work)"
        status = "Significant improvements required"
    
    print(f"Overall Grade: {grade}")
    print(f"Status: {status}")
    print()
    
    # Strengths
    print("STRENGTHS:")
    strengths = []
    if scores["Safety"] >= 0.95:
        strengths.append("Excellent safety compliance")
    if scores["Citations"] >= 0.95:
        strengths.append("High-quality citations")
    if scores["Retrieval"] >= 0.8:
        strengths.append("Strong retrieval performance")
    
    for strength in strengths:
        print(f"  ✓ {strength}")
    
    if not strengths:
        print("  • System meets basic requirements")
    
    print()
    
    # Weaknesses
    print("AREAS FOR IMPROVEMENT:")
    weaknesses = []
    if scores["Retrieval"] < 0.7:
        weaknesses.append("Retrieval recall needs improvement")
    if scores["Safety"] < 0.95:
        weaknesses.append("Safety filter has gaps")
    if scores["Citations"] < 0.9:
        weaknesses.append("Citation quality inconsistent")
    
    for weakness in weaknesses:
        print(f"  ⚠ {weakness}")
    
    if not weaknesses:
        print("  ✓ No major weaknesses identified")
    
    print()
    
    print(f"Report generated: {results['timestamp']}")
    print()
    
    # Save comprehensive report
    report_path = Path(__file__).parent / "evaluation_report.json"
    full_report = {
        "timestamp": results["timestamp"],
        "summary": summary,
        "analyses": analyses,
        "error_patterns": patterns,
        "recommendations": recommendations,
        "overall_grade": grade,
        "system_status": status
    }
    
    with open(report_path, 'w') as f:
        json.dump(full_report, f, indent=2)
    
    print(f"✓ Comprehensive report saved to: {report_path}")


def main():
    """Run complete error analysis and reporting."""
    
    print("\n")
    print("=" * 70)
    print("STEP 11: RESULTS ANALYSIS & ERROR DIAGNOSIS")
    print("=" * 70)
    print("\n")
    
    # Load evaluation results
    results = load_evaluation_results()
    
    # Run analyses
    analyses = {
        "retrieval": analyze_retrieval_failures(results),
        "faithfulness": analyze_faithfulness_issues(results),
        "citations": analyze_citation_quality(results),
        "safety": analyze_safety_compliance(results)
    }
    
    # Identify patterns
    patterns = identify_error_patterns(analyses)
    
    # Generate recommendations
    recommendations = generate_improvement_recommendations(analyses, patterns)
    
    # Final report
    generate_final_report(results, analyses, patterns, recommendations)
    
    print("\n✓ Error analysis complete")
    print()


if __name__ == "__main__":
    main()
