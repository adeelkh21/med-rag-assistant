"""
STEP 5: Safety Gate (Pre-LLM Filter)

Intercepts unsafe medical queries before retrieval or LLM calls.
Uses keyword/phrase matching to detect disallowed requests.

Disallowed Query Types:
- Diagnosis ("Do I have diabetes?")
- Medication dosage ("How much insulin should I take?")
- Treatment recommendations ("What treatment should I follow?")

Detection: Simple keyword matching (no ML required)
Response: Polite refusal, no retrieval, no LLM call
"""

from typing import Tuple, List
import re


# Refusal response template (STRICT)
REFUSAL_RESPONSE = (
    "I can't help with diagnosis or treatment decisions. "
    "Please consult a qualified healthcare professional for personalized medical advice."
)


# Keyword patterns for unsafe query detection
DIAGNOSIS_KEYWORDS = [
    # Direct diagnosis requests
    r'\bdo i have\b',
    r'\bhave i got\b',
    r'\bam i (?:suffering from|experiencing)\b',
    r'\bis this\b.*\b(?:cancer|disease|condition)\b',
    r'\bwhat disease\b',
    r'\bwhat condition\b',
    r'\bdiagnose me\b',
    r'\bdiagnosis\b.*\bme\b',
    r'\bthese symptoms (?:mean|indicate)\b',
    r'\bwhat (?:do|does) (?:these|my) symptoms\b',
    r'\bcould (?:i|this be)\b.*\b(?:cancer|diabetes|disease)\b',
]

MEDICATION_KEYWORDS = [
    # Dosage and prescription
    r'\bhow much\b.*\b(?:take|dose|dosage)\b',
    r'\bdose(?:age)?\b.*\b(?:should|recommend|correct|right)\b',
    r'\bhow many\b.*\b(?:mg|pill|tablet|capsule)\b',
    r'\bcan i (?:increase|decrease|change)\b.*\b(?:dose|dosage|medicine|medication)\b',
    r'\bwhat (?:is|should be) (?:the|my) (?:dose|dosage)\b',
    r'\bmg\b.*\b(?:take|should|recommend)\b',
    r'\bprescribe(?:d)?\b.*\b(?:me|for me)\b',
]

TREATMENT_KEYWORDS = [
    # Treatment and prescription recommendations
    r'\bshould i take\b',
    r'\bshould i start\b.*\b(?:treatment|therapy|medication|medicine)\b',
    r'\bbest (?:treatment|medicine|medication|drug)\b',
    r'\bwhat (?:treatment|medicine|medication) should\b',
    r'\brecommend(?:ed)? (?:treatment|medication|medicine)\b',
    r'\bwhich (?:treatment|medicine|medication|drug) is best\b',
    r'\bcan i start\b.*\b(?:treatment|therapy|chemotherapy)\b',
    r'\bshould i (?:undergo|get|have)\b.*\b(?:surgery|operation|treatment)\b',
    r'\bhow (?:should|do) i treat\b',
    r'\btreat my\b',
    r'\btreat (?:my|this|the)\b.*\b(?:condition|injury|wound|fracture|broken)\b',
]


def compile_patterns(keywords: List[str]) -> List[re.Pattern]:
    """Compile regex patterns for efficient matching."""
    return [re.compile(pattern, re.IGNORECASE) for pattern in keywords]


# Pre-compile all patterns for efficiency
DIAGNOSIS_PATTERNS = compile_patterns(DIAGNOSIS_KEYWORDS)
MEDICATION_PATTERNS = compile_patterns(MEDICATION_KEYWORDS)
TREATMENT_PATTERNS = compile_patterns(TREATMENT_KEYWORDS)


def check_diagnosis_request(query: str) -> bool:
    """
    Check if query requests a diagnosis.
    
    Args:
        query: User query string
    
    Returns:
        True if diagnosis request detected
    """
    return any(pattern.search(query) for pattern in DIAGNOSIS_PATTERNS)


def check_medication_request(query: str) -> bool:
    """
    Check if query requests medication dosage or prescription.
    
    Args:
        query: User query string
    
    Returns:
        True if medication request detected
    """
    return any(pattern.search(query) for pattern in MEDICATION_PATTERNS)


def check_treatment_request(query: str) -> bool:
    """
    Check if query requests treatment recommendations.
    
    Args:
        query: User query string
    
    Returns:
        True if treatment recommendation request detected
    """
    return any(pattern.search(query) for pattern in TREATMENT_PATTERNS)


def is_safe_query(query: str) -> Tuple[bool, str]:
    """
    Main safety check: Determine if query is safe to process.
    
    This is the primary entry point for safety filtering.
    
    Args:
        query: User query string
    
    Returns:
        Tuple of (is_safe, reason)
        - is_safe: True if query is safe, False if should be refused
        - reason: Empty string if safe, otherwise category of violation
    """
    if not query or not query.strip():
        return False, "empty_query"
    
    query = query.strip()
    
    # Check each category
    if check_diagnosis_request(query):
        return False, "diagnosis_request"
    
    if check_medication_request(query):
        return False, "medication_request"
    
    if check_treatment_request(query):
        return False, "treatment_request"
    
    # Query is safe
    return True, ""


def get_refusal_response(reason: str = "") -> str:
    """
    Get the refusal response for unsafe queries.
    
    Args:
        reason: Category of violation (optional, for logging)
    
    Returns:
        Refusal response string
    """
    return REFUSAL_RESPONSE


def filter_query(query: str) -> Tuple[bool, str]:
    """
    Filter query through safety gate.
    
    Convenience function that combines safety check and refusal response.
    
    Args:
        query: User query string
    
    Returns:
        Tuple of (should_proceed, response)
        - should_proceed: True if safe, False if refused
        - response: Empty string if safe, refusal message if unsafe
    """
    is_safe, reason = is_safe_query(query)
    
    if not is_safe:
        return False, get_refusal_response(reason)
    
    return True, ""


# Example test cases for validation
def test_safety_filter():
    """Test the safety filter with example queries."""
    test_cases = [
        # Safe queries
        ("What are the symptoms of diabetes?", True),
        ("How is lung cancer treated?", True),
        ("What causes high blood pressure?", True),
        ("Explain what heart disease is", True),
        
        # Diagnosis requests (unsafe)
        ("Do I have diabetes?", False),
        ("Is this cancer?", False),
        ("What disease do these symptoms indicate?", False),
        ("Could I have heart disease?", False),
        
        # Medication requests (unsafe)
        ("How much insulin should I take?", False),
        ("What is the correct dose of aspirin?", False),
        ("Can I increase my medicine dose?", False),
        ("How many mg of metformin should I take?", False),
        
        # Treatment requests (unsafe)
        ("What treatment should I follow?", False),
        ("Which medicine is best for hypertension?", False),
        ("Should I start chemotherapy?", False),
        ("What medication should I take for diabetes?", False),
    ]
    
    print("Testing Safety Filter...")
    print("=" * 70)
    
    passed = 0
    failed = 0
    
    for query, expected_safe in test_cases:
        is_safe, reason = is_safe_query(query)
        status = "✓" if is_safe == expected_safe else "✗"
        
        if is_safe == expected_safe:
            passed += 1
        else:
            failed += 1
        
        print(f"{status} {query}")
        if not is_safe:
            print(f"  → Blocked: {reason}")
    
    print("=" * 70)
    print(f"Passed: {passed}/{len(test_cases)}")
    print(f"Failed: {failed}/{len(test_cases)}")
    
    return failed == 0


if __name__ == "__main__":
    # Run tests
    success = test_safety_filter()
    
    if success:
        print("\n✓ All safety filter tests passed")
    else:
        print("\n✗ Some tests failed")
