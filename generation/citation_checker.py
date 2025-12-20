"""
Citation Checker - Post-Generation Validation

Verifies that generated answers are faithfully supported by retrieved evidence.

Validation Rules:
1. Every factual sentence must have a citation
2. Every citation must refer to a retrieved chunk
3. Answer content must appear in cited chunks

Failure → Reject response → Trigger regeneration
"""

import re
from typing import List, Dict, Any, Tuple, Set


# Disclaimer patterns to ignore
DISCLAIMER_PATTERNS = [
    r"this information is for educational purposes",
    r"always consult.*healthcare professional",
    r"not medical advice",
    r"seek medical attention",
    r"get medical help",
    r"talk with your doctor",
]


def split_into_sentences(text: str) -> List[str]:
    """
    Split text into sentences.
    
    Args:
        text: Input text
    
    Returns:
        List of sentences
    """
    # Simple sentence splitting (handles most cases)
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]


def is_disclaimer_sentence(sentence: str) -> bool:
    """
    Check if sentence is a disclaimer (should be ignored).
    
    Args:
        sentence: Input sentence
    
    Returns:
        True if disclaimer
    """
    sentence_lower = sentence.lower()
    for pattern in DISCLAIMER_PATTERNS:
        if re.search(pattern, sentence_lower):
            return True
    return False


def extract_citations(sentence: str) -> List[str]:
    """
    Extract citation chunk IDs from a sentence.
    
    Matches both formats:
    - (CHUNK_ID)
    - [CHUNK_ID]
    
    Args:
        sentence: Input sentence
    
    Returns:
        List of chunk IDs
    """
    # Match both (CHUNK_ID) and [CHUNK_ID] formats
    citations = re.findall(r'[\(\[]([A-Z_0-9]+)[\)\]]', sentence)
    return citations


def get_keywords(text: str, min_length: int = 4) -> Set[str]:
    """
    Extract meaningful keywords from text.
    
    Args:
        text: Input text
        min_length: Minimum word length to consider
    
    Returns:
        Set of lowercase keywords
    """
    # Remove punctuation and split
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Filter out short words and common stopwords
    stopwords = {
        'the', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
        'could', 'should', 'may', 'might', 'can', 'this', 'that',
        'these', 'those', 'a', 'an', 'and', 'or', 'but', 'if', 'for',
        'from', 'to', 'in', 'on', 'at', 'by', 'with', 'about'
    }
    
    keywords = {w for w in words if len(w) >= min_length and w not in stopwords}
    return keywords


def verify_content_support(
    sentence: str,
    cited_chunks: List[Dict[str, Any]],
    min_keyword_overlap: float = 0.3
) -> Tuple[bool, str]:
    """
    Verify that sentence content is supported by cited chunks.
    
    Uses keyword overlap as a proxy for content support.
    
    Args:
        sentence: Sentence to verify
        cited_chunks: List of cited chunk documents
        min_keyword_overlap: Minimum overlap ratio (0.0-1.0)
    
    Returns:
        Tuple of (is_supported, reason)
    """
    if not cited_chunks:
        return False, "No chunks provided for verification"
    
    # Extract keywords from sentence (without citations)
    sentence_clean = re.sub(r'[\(\[][A-Z_0-9]+[\)\]]', '', sentence)
    sentence_keywords = get_keywords(sentence_clean)
    
    if not sentence_keywords:
        # Sentence has no meaningful keywords (might be connector/transition)
        return True, "No verifiable keywords"
    
    # Check overlap with any cited chunk
    for chunk in cited_chunks:
        chunk_text = chunk.get("text", "")
        chunk_keywords = get_keywords(chunk_text)
        
        # Calculate overlap
        overlap = sentence_keywords & chunk_keywords
        overlap_ratio = len(overlap) / len(sentence_keywords)
        
        if overlap_ratio >= min_keyword_overlap:
            return True, f"Supported by chunk {chunk.get('id')} ({overlap_ratio:.1%} overlap)"
    
    # No sufficient overlap found
    return False, f"Insufficient keyword overlap with cited chunks"


def validate_citations(
    answer: str,
    retrieved_docs: List[Dict[str, Any]],
    min_keyword_overlap: float = 0.3,
    verbose: bool = False
) -> Tuple[bool, List[Dict[str, Any]]]:
    """
    Validate that answer citations are properly supported.
    
    Args:
        answer: Generated answer text
        retrieved_docs: List of retrieved documents
        min_keyword_overlap: Minimum keyword overlap for support
        verbose: Print detailed validation info
    
    Returns:
        Tuple of (is_valid, validation_errors)
        validation_errors: List of error dictionaries
    """
    if verbose:
        print("\n[Citation Checker] Starting validation...")
    
    # Build chunk lookup
    chunk_lookup = {doc["id"]: doc for doc in retrieved_docs}
    
    # Split into sentences
    sentences = split_into_sentences(answer)
    
    if verbose:
        print(f"  Total sentences: {len(sentences)}")
    
    # Validation errors
    errors = []
    
    for idx, sentence in enumerate(sentences, 1):
        # Skip disclaimer sentences
        if is_disclaimer_sentence(sentence):
            if verbose:
                print(f"  [{idx}] Skipping disclaimer sentence")
            continue
        
        # Extract citations
        citations = extract_citations(sentence)
        
        if verbose:
            print(f"  [{idx}] Citations: {citations if citations else 'NONE'}")
        
        # Check: Does sentence have citations?
        if not citations:
            errors.append({
                "type": "missing_citation",
                "sentence_num": idx,
                "sentence": sentence[:100] + "..." if len(sentence) > 100 else sentence,
                "reason": "Factual sentence without citation"
            })
            continue
        
        # Check: Do citations refer to valid chunks?
        invalid_citations = [c for c in citations if c not in chunk_lookup]
        if invalid_citations:
            errors.append({
                "type": "invalid_citation",
                "sentence_num": idx,
                "sentence": sentence[:100] + "..." if len(sentence) > 100 else sentence,
                "citations": invalid_citations,
                "reason": f"Citations not in retrieved chunks: {invalid_citations}"
            })
            continue
        
        # Check: Is content supported by cited chunks?
        cited_chunks = [chunk_lookup[c] for c in citations]
        is_supported, support_reason = verify_content_support(
            sentence, cited_chunks, min_keyword_overlap
        )
        
        if not is_supported:
            errors.append({
                "type": "unsupported_content",
                "sentence_num": idx,
                "sentence": sentence[:100] + "..." if len(sentence) > 100 else sentence,
                "citations": citations,
                "reason": support_reason
            })
        elif verbose:
            print(f"      ✓ {support_reason}")
    
    # Summary
    is_valid = len(errors) == 0
    
    if verbose:
        if is_valid:
            print("  ✓ All citations valid\n")
        else:
            print(f"  ✗ Found {len(errors)} validation errors\n")
    
    return is_valid, errors


def get_validation_summary(errors: List[Dict[str, Any]]) -> str:
    """
    Generate human-readable summary of validation errors.
    
    Args:
        errors: List of validation errors
    
    Returns:
        Summary string
    """
    if not errors:
        return "All citations valid"
    
    summary_lines = [f"Found {len(errors)} citation error(s):"]
    
    for error in errors[:3]:  # Show first 3 errors
        error_type = error["type"].replace("_", " ").title()
        sentence_num = error["sentence_num"]
        reason = error["reason"]
        summary_lines.append(f"  - Sentence {sentence_num}: {error_type} - {reason}")
    
    if len(errors) > 3:
        summary_lines.append(f"  ... and {len(errors) - 3} more")
    
    return "\n".join(summary_lines)


# Test/demo code
if __name__ == "__main__":
    print("=" * 70)
    print("Citation Checker Demo")
    print("=" * 70)
    
    # Sample answer with citations
    sample_answer = """
    Type 2 diabetes is a chronic condition (DOC_001). The main symptoms include increased thirst and frequent urination (DOC_002). 
    Risk factors include being overweight and physical inactivity (DOC_003).
    This information is for educational purposes only and is not medical advice.
    """
    
    # Sample retrieved docs
    sample_docs = [
        {
            "id": "DOC_001",
            "text": "Type 2 diabetes is a chronic metabolic condition affecting blood sugar regulation."
        },
        {
            "id": "DOC_002",
            "text": "Common symptoms of diabetes include increased thirst, frequent urination, increased hunger, and fatigue."
        },
        {
            "id": "DOC_003",
            "text": "Risk factors for type 2 diabetes include being overweight, physical inactivity, family history, and age over 45."
        }
    ]
    
    # Validate
    is_valid, errors = validate_citations(
        sample_answer,
        sample_docs,
        verbose=True
    )
    
    print("=" * 70)
    print("RESULT:")
    print("=" * 70)
    print(f"Valid: {is_valid}")
    
    if errors:
        print("\n" + get_validation_summary(errors))
    
    print("\n" + "=" * 70)
