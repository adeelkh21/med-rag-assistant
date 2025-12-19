"""
STEP 8: Response Post-Processing & Validation

Ensures generated answers are safe, grounded, and evaluable.

Mandatory Checks:
1. Citations present (at least one chunk ID per answer)
2. Citation validity (all IDs exist in retrieved context)
3. Disclaimer present (exact match required)

Failure → reject response and regenerate (or return error)
"""

from typing import List, Dict, Any, Tuple
import re


# Mandatory disclaimer (must match exactly)
REQUIRED_DISCLAIMER = (
    "This information is for educational purposes only and is not medical advice."
)


def extract_citations(answer: str) -> List[str]:
    """
    Extract all chunk ID citations from answer text.
    
    Citations are in format: (CHUNK_ID) or [CHUNK ID: CHUNK_ID]
    
    Args:
        answer: Generated answer text
    
    Returns:
        List of cited chunk IDs
    """
    # Pattern matches: (ID), [CHUNK ID: ID], or similar
    pattern = r'\(([A-Z0-9_]+)\)'
    citations = re.findall(pattern, answer)
    return list(set(citations))  # Remove duplicates


def check_citations_present(answer: str) -> Tuple[bool, str]:
    """
    Check if answer contains at least one citation.
    
    Args:
        answer: Generated answer text
    
    Returns:
        Tuple of (valid, error_message)
    """
    citations = extract_citations(answer)
    
    if not citations:
        return False, "No citations found in answer"
    
    return True, ""


def check_citation_validity(
    answer: str,
    retrieved_docs: List[Dict[str, Any]]
) -> Tuple[bool, str]:
    """
    Check that all cited chunk IDs exist in retrieved context.
    
    Prevents hallucinated citations.
    
    Args:
        answer: Generated answer text
        retrieved_docs: List of retrieved documents with 'id' field
    
    Returns:
        Tuple of (valid, error_message)
    """
    citations = extract_citations(answer)
    valid_ids = {doc["id"] for doc in retrieved_docs}
    
    # Check for hallucinated citations
    hallucinated = [cid for cid in citations if cid not in valid_ids]
    
    if hallucinated:
        return False, f"Hallucinated citations: {hallucinated}"
    
    return True, ""


def check_disclaimer_present(answer: str) -> Tuple[bool, str]:
    """
    Check that mandatory disclaimer is present.
    
    Args:
        answer: Generated answer text
    
    Returns:
        Tuple of (valid, error_message)
    """
    # Check for exact match (case-insensitive, ignoring extra whitespace)
    normalized_answer = " ".join(answer.split()).lower()
    normalized_disclaimer = " ".join(REQUIRED_DISCLAIMER.split()).lower()
    
    if normalized_disclaimer not in normalized_answer:
        return False, "Required disclaimer missing"
    
    return True, ""


def validate_response(
    answer: str,
    retrieved_docs: List[Dict[str, Any]]
) -> Tuple[bool, List[str]]:
    """
    Perform all validation checks on generated answer.
    
    This is the main validation entry point.
    
    Args:
        answer: Generated answer text
        retrieved_docs: Retrieved documents used for context
    
    Returns:
        Tuple of (is_valid, error_messages)
        - is_valid: True if all checks pass
        - error_messages: List of validation errors (empty if valid)
    """
    errors = []
    
    # Check 1: Citations present
    valid, error = check_citations_present(answer)
    if not valid:
        errors.append(error)
    
    # Check 2: Citation validity
    valid, error = check_citation_validity(answer, retrieved_docs)
    if not valid:
        errors.append(error)
    
    # Check 3: Disclaimer present
    valid, error = check_disclaimer_present(answer)
    if not valid:
        errors.append(error)
    
    is_valid = len(errors) == 0
    return is_valid, errors


def get_citations_summary(answer: str, retrieved_docs: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Get summary of citations used in answer.
    
    Useful for evaluation and debugging.
    
    Args:
        answer: Generated answer text
        retrieved_docs: Retrieved documents
    
    Returns:
        Dictionary with citation statistics
    """
    citations = extract_citations(answer)
    valid_ids = {doc["id"] for doc in retrieved_docs}
    
    return {
        "total_citations": len(citations),
        "unique_citations": len(set(citations)),
        "cited_doc_ids": citations,
        "uncited_docs": [doc["id"] for doc in retrieved_docs if doc["id"] not in citations],
        "hallucinated_citations": [cid for cid in citations if cid not in valid_ids]
    }


# Example test cases
def test_validator():
    """Test the validator with example answers."""
    print("=" * 70)
    print("Testing Response Validator")
    print("=" * 70)
    
    # Sample retrieved docs
    docs = [
        {"id": "NCI_DIABETES_01", "text": "...", "score": 0.9},
        {"id": "CDC_DIABETES_02", "text": "...", "score": 0.8}
    ]
    
    # Test cases
    test_cases = [
        # Valid answer
        (
            "Type 2 diabetes symptoms include increased thirst (NCI_DIABETES_01). "
            "Risk factors include being overweight (CDC_DIABETES_02). "
            "This information is for educational purposes only and is not medical advice.",
            True,
            "Valid answer with citations and disclaimer"
        ),
        
        # Missing citations
        (
            "Type 2 diabetes has various symptoms. "
            "This information is for educational purposes only and is not medical advice.",
            False,
            "Missing citations"
        ),
        
        # Hallucinated citation
        (
            "Diabetes symptoms include thirst (FAKE_ID_123). "
            "This information is for educational purposes only and is not medical advice.",
            False,
            "Hallucinated citation"
        ),
        
        # Missing disclaimer
        (
            "Type 2 diabetes symptoms include increased thirst (NCI_DIABETES_01).",
            False,
            "Missing disclaimer"
        ),
    ]
    
    passed = 0
    failed = 0
    
    for answer, expected_valid, description in test_cases:
        is_valid, errors = validate_response(answer, docs)
        
        status = "✓" if is_valid == expected_valid else "✗"
        
        if is_valid == expected_valid:
            passed += 1
        else:
            failed += 1
        
        print(f"\n{status} {description}")
        if not is_valid:
            print(f"  Errors: {errors}")
        
        # Show citation summary
        summary = get_citations_summary(answer, docs)
        print(f"  Citations: {summary['unique_citations']} unique")
    
    print("\n" + "=" * 70)
    print(f"Passed: {passed}/{len(test_cases)}")
    print(f"Failed: {failed}/{len(test_cases)}")
    
    return failed == 0


if __name__ == "__main__":
    success = test_validator()
    
    if success:
        print("\n✓ All validator tests passed")
    else:
        print("\n✗ Some tests failed")
