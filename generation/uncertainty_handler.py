"""
Uncertainty Handler - Low-Confidence Retrieval Detection

Detects weak retrieval evidence and provides safe fallback behavior.

Logic:
1. Check max similarity score after retrieval
2. If below threshold → Return safe fallback (no LLM call)
3. Threshold: 0.2-0.3 (configurable)

Prevents confident answers when retrieval evidence is weak.
"""

from typing import List, Dict, Any, Optional, Tuple


# Confidence thresholds
DEFAULT_LOW_CONFIDENCE_THRESHOLD = 0.25
DEFAULT_MEDIUM_CONFIDENCE_THRESHOLD = 0.40


# Safe fallback responses
CLARIFYING_QUESTION = (
    "I'm not finding strong matches in my knowledge base for your question. "
    "Could you clarify which specific condition, symptom, or topic you're asking about? "
    "Providing more details will help me find more relevant information."
)

SAFE_GENERAL_GUIDANCE = (
    "I don't have strong evidence from the provided sources to answer this question confidently. "
    "For accurate medical information, please consult official medical guidance or a healthcare professional."
)

LOW_CONFIDENCE_INFO = (
    "The available sources have limited information on this specific topic. "
    "I can share what I found, but the confidence is low. "
    "For comprehensive information, please consult a healthcare professional or trusted medical resource."
)


def get_max_similarity_score(retrieved_docs: List[Dict[str, Any]]) -> float:
    """
    Get maximum similarity score from retrieved documents.
    
    Args:
        retrieved_docs: List of retrieved documents with scores
    
    Returns:
        Maximum score (0.0 if no docs)
    """
    if not retrieved_docs:
        return 0.0
    
    scores = [doc.get("score", 0.0) for doc in retrieved_docs]
    return max(scores)


def check_retrieval_confidence(
    retrieved_docs: List[Dict[str, Any]],
    low_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    medium_threshold: float = DEFAULT_MEDIUM_CONFIDENCE_THRESHOLD
) -> Tuple[str, float]:
    """
    Check retrieval confidence level.
    
    Args:
        retrieved_docs: List of retrieved documents
        low_threshold: Threshold for low confidence
        medium_threshold: Threshold for medium confidence
    
    Returns:
        Tuple of (confidence_level, max_score)
        confidence_level: "low", "medium", or "high"
    """
    max_score = get_max_similarity_score(retrieved_docs)
    
    if max_score < low_threshold:
        return "low", max_score
    elif max_score < medium_threshold:
        return "medium", max_score
    else:
        return "high", max_score


def handle_low_confidence(
    query: str,
    retrieved_docs: List[Dict[str, Any]],
    low_threshold: float = DEFAULT_LOW_CONFIDENCE_THRESHOLD,
    fallback_type: str = "clarifying",
    verbose: bool = False
) -> Optional[str]:
    """
    Handle low-confidence retrieval.
    
    If confidence is below threshold, return safe fallback response.
    Otherwise, return None (proceed to generation).
    
    Args:
        query: User query
        retrieved_docs: Retrieved documents
        low_threshold: Confidence threshold
        fallback_type: Type of fallback ("clarifying", "safe_general", or "low_conf_warning")
        verbose: Print debug info
    
    Returns:
        Fallback response if low confidence, None otherwise
    """
    confidence_level, max_score = check_retrieval_confidence(
        retrieved_docs,
        low_threshold=low_threshold
    )
    
    if verbose:
        print(f"[Uncertainty Handler] Confidence: {confidence_level} (max_score={max_score:.3f})")
    
    if confidence_level == "low":
        if verbose:
            print(f"  → Below threshold ({low_threshold:.2f})")
            print(f"  → Returning safe fallback ({fallback_type})")
        
        # Choose fallback type
        if fallback_type == "clarifying":
            return CLARIFYING_QUESTION
        elif fallback_type == "safe_general":
            return SAFE_GENERAL_GUIDANCE
        elif fallback_type == "low_conf_warning":
            return LOW_CONFIDENCE_INFO
        else:
            return SAFE_GENERAL_GUIDANCE
    
    if verbose:
        print(f"  → Above threshold, proceeding to generation")
    
    return None


def format_uncertainty_response(
    fallback_message: str,
    retrieved_docs: List[Dict[str, Any]],
    include_disclaimer: bool = True
) -> Dict[str, Any]:
    """
    Format uncertainty response in standard answer format.
    
    Args:
        fallback_message: Fallback message text
        retrieved_docs: Retrieved documents (for metadata)
        include_disclaimer: Include educational disclaimer
    
    Returns:
        Formatted response dictionary
    """
    if include_disclaimer:
        disclaimer = "\n\nThis information is for educational purposes only and is not medical advice. Always consult a qualified healthcare professional for medical concerns."
        answer_text = fallback_message + disclaimer
    else:
        answer_text = fallback_message
    
    return {
        "success": True,
        "query": "",
        "answer": answer_text,
        "error": None,
        "retrieved_docs": retrieved_docs,
        "citations_used": [],
        "validation_passed": True,
        "low_confidence": True  # Flag to indicate this was a fallback
    }


def get_confidence_warning(max_score: float, threshold: float) -> str:
    """
    Generate a confidence warning message.
    
    Args:
        max_score: Maximum retrieval score
        threshold: Confidence threshold
    
    Returns:
        Warning message
    """
    return (
        f"⚠️ Low confidence retrieval (score: {max_score:.2f}, threshold: {threshold:.2f}). "
        "The available sources may not fully address your question."
    )


# Test/demo code
if __name__ == "__main__":
    print("=" * 70)
    print("Uncertainty Handler Demo")
    print("=" * 70)
    
    # Test cases
    test_cases = [
        {
            "name": "High Confidence",
            "docs": [
                {"id": "DOC_001", "score": 0.85, "text": "High relevance match"}
            ]
        },
        {
            "name": "Medium Confidence",
            "docs": [
                {"id": "DOC_002", "score": 0.35, "text": "Moderate relevance"}
            ]
        },
        {
            "name": "Low Confidence",
            "docs": [
                {"id": "DOC_003", "score": 0.15, "text": "Poor relevance"}
            ]
        },
        {
            "name": "No Results",
            "docs": []
        }
    ]
    
    for test in test_cases:
        print(f"\nTest: {test['name']}")
        print("-" * 70)
        
        fallback = handle_low_confidence(
            query="What are symptoms of diabetes?",
            retrieved_docs=test["docs"],
            low_threshold=0.25,
            fallback_type="clarifying",
            verbose=True
        )
        
        if fallback:
            print(f"\nFallback Response:\n{fallback}")
        else:
            print("\n→ Proceed to generation")
    
    print("\n" + "=" * 70)
    print("✓ Uncertainty Handler Demo Complete")
    print("=" * 70)
