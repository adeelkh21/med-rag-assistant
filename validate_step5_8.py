"""
Validation script for STEP 5-8: Complete Answer Generation Pipeline

Tests:
1. Safety gate functionality
2. Prompt construction
3. LLM client (with API key check)
4. Response validator
5. End-to-end answer generation
"""

import os
import sys
from pathlib import Path

# Test queries
SAFE_QUERY = "What are the symptoms of type 2 diabetes?"
UNSAFE_DIAGNOSIS = "Do I have cancer?"
UNSAFE_MEDICATION = "What dose of insulin should I take?"
UNSAFE_TREATMENT = "How should I treat my broken arm?"


def test_safety_filter():
    """Test STEP 5: Safety gate."""
    print("\n" + "=" * 70)
    print("TEST 1: Safety Filter")
    print("=" * 70)
    
    from generation.safety_filter import filter_query, is_safe_query, get_refusal_response
    
    test_cases = [
        (SAFE_QUERY, True, "Safe query"),
        (UNSAFE_DIAGNOSIS, False, "Diagnosis request"),
        (UNSAFE_MEDICATION, False, "Medication dosage"),
        (UNSAFE_TREATMENT, False, "Treatment recommendation"),
        ("What causes diabetes?", True, "General information"),
    ]
    
    passed = 0
    for query, expected_safe, description in test_cases:
        should_proceed, refusal = filter_query(query)
        
        if should_proceed == expected_safe:
            print(f"  ✓ {description}: {'SAFE' if should_proceed else 'BLOCKED'}")
            passed += 1
        else:
            print(f"  ✗ {description}: Expected {'SAFE' if expected_safe else 'BLOCKED'}, got {'SAFE' if should_proceed else 'BLOCKED'}")
    
    print(f"\nPassed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_prompts():
    """Test STEP 6: Prompt construction."""
    print("\n" + "=" * 70)
    print("TEST 2: Prompt Construction")
    print("=" * 70)
    
    from generation.prompts import (
        get_system_prompt,
        build_user_prompt,
        format_context_chunks,
        get_mandatory_disclaimer
    )
    
    # Mock retrieved documents
    mock_docs = [
        {"id": "DOC_001", "text": "Diabetes is a chronic condition...", "score": 0.85},
        {"id": "DOC_002", "text": "Type 2 diabetes symptoms include...", "score": 0.80},
    ]
    
    checks = []
    
    # Check system prompt
    system_prompt = get_system_prompt()
    checks.append(("System prompt exists", len(system_prompt) > 0))
    checks.append(("System prompt is locked", "medical information assistant" in system_prompt.lower()))
    
    # Check user prompt
    user_prompt = build_user_prompt(SAFE_QUERY, mock_docs)
    checks.append(("User prompt exists", len(user_prompt) > 0))
    checks.append(("Contains context", "Context:" in user_prompt or "[CHUNK" in user_prompt))
    checks.append(("Contains question", SAFE_QUERY in user_prompt))
    checks.append(("Contains citation instruction", "citation" in user_prompt.lower() or "cite" in user_prompt.lower()))
    
    # Check mandatory disclaimer
    disclaimer = get_mandatory_disclaimer()
    checks.append(("Disclaimer exists", len(disclaimer) > 0))
    checks.append(("Disclaimer warns about professional", "healthcare professional" in disclaimer.lower()))
    
    passed = 0
    for description, result in checks:
        if result:
            print(f"  ✓ {description}")
            passed += 1
        else:
            print(f"  ✗ {description}")
    
    print(f"\nPassed: {passed}/{len(checks)}")
    return passed == len(checks)


def test_llm_client():
    """Test STEP 7: Groq API client."""
    print("\n" + "=" * 70)
    print("TEST 3: LLM Client")
    print("=" * 70)
    
    from generation.llm_client import GroqClient
    
    # Check API key
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        print("  ✗ GROQ_API_KEY not set in environment")
        print("    Set with: export GROQ_API_KEY='your-key-here'")
        print("    Skipping API call test")
        return False
    
    print(f"  ✓ GROQ_API_KEY found ({api_key[:8]}...)")
    
    # Initialize client
    try:
        client = GroqClient()
        print("  ✓ GroqClient initialized")
    except Exception as e:
        print(f"  ✗ Failed to initialize client: {e}")
        return False
    
    # Test generation
    print("\n  Testing API call (this may take 5-10 seconds)...")
    try:
        response = client.generate(
            system_prompt="You are a helpful assistant.",
            user_prompt="Say 'Hello, I am working!' and nothing else.",
            temperature=0.1,
            max_tokens=50
        )
        
        print(f"  ✓ API call successful")
        print(f"    Response: {response[:100]}...")
        
        # Check response
        if len(response) > 0:
            print("  ✓ Non-empty response")
            return True
        else:
            print("  ✗ Empty response")
            return False
    
    except Exception as e:
        print(f"  ✗ API call failed: {e}")
        return False


def test_validator():
    """Test STEP 8: Response validator."""
    print("\n" + "=" * 70)
    print("TEST 4: Response Validator")
    print("=" * 70)
    
    from generation.validator import (
        validate_response,
        extract_citations,
        check_citation_validity,
        check_disclaimer_present
    )
    from generation.prompts import get_mandatory_disclaimer
    
    # Mock retrieved documents
    mock_docs = [
        {"id": "DOC_001", "text": "...", "score": 0.9},
        {"id": "DOC_002", "text": "...", "score": 0.85},
    ]
    
    disclaimer = get_mandatory_disclaimer()
    
    test_cases = [
        (
            f"Type 2 diabetes is a chronic condition (DOC_001). Symptoms include increased thirst (DOC_002). {disclaimer}",
            True,
            "Valid response with citations and disclaimer"
        ),
        (
            "Type 2 diabetes is a chronic condition. No citations here.",
            False,
            "No citations"
        ),
        (
            "Type 2 diabetes is a chronic condition (DOC_999). This is important information.",
            False,
            "Hallucinated citation"
        ),
        (
            "Type 2 diabetes is a chronic condition (DOC_001). Missing disclaimer.",
            False,
            "Missing disclaimer"
        ),
    ]
    
    passed = 0
    for response, expected_valid, description in test_cases:
        is_valid, errors = validate_response(response, mock_docs)
        
        if is_valid == expected_valid:
            print(f"  ✓ {description}: {'VALID' if is_valid else 'INVALID'}")
            passed += 1
        else:
            print(f"  ✗ {description}: Expected {'VALID' if expected_valid else 'INVALID'}, got {'VALID' if is_valid else 'INVALID'}")
            if errors:
                print(f"    Errors: {errors}")
    
    print(f"\nPassed: {passed}/{len(test_cases)}")
    return passed == len(test_cases)


def test_end_to_end():
    """Test STEP 5-8: Complete answer generation pipeline."""
    print("\n" + "=" * 70)
    print("TEST 5: End-to-End Answer Generation")
    print("=" * 70)
    
    # Check API key
    if not os.getenv("GROQ_API_KEY"):
        print("  ✗ GROQ_API_KEY not set - skipping end-to-end test")
        return False
    
    # Check if retrieval components exist
    if not Path("retrieval/index.faiss").exists():
        print("  ✗ FAISS index not found - run build_faiss_index.py first")
        return False
    
    from generation.answer_generator import MedicalAnswerGenerator
    
    # Initialize generator
    print("\n  Initializing answer generator...")
    try:
        generator = MedicalAnswerGenerator(top_k=6)
        print("  ✓ Generator initialized")
    except Exception as e:
        print(f"  ✗ Failed to initialize: {e}")
        return False
    
    # Test 1: Safe query
    print(f"\n  Test 1: Safe query")
    print(f"  Query: {SAFE_QUERY}")
    
    try:
        result = generator.generate_answer(SAFE_QUERY, verbose=False)
        
        if result["success"]:
            print(f"  ✓ Answer generated successfully")
            print(f"    Length: {len(result['answer'])} chars")
            print(f"    Citations: {len(result.get('citations_used', []))} used")
            print(f"    Retrieved: {len(result['retrieved_docs'])} docs")
            print(f"    Validation: {'PASSED' if result['validation_passed'] else 'FAILED'}")
            
            # Show snippet
            print(f"\n    Answer snippet:")
            print(f"    {result['answer'][:200]}...")
        else:
            print(f"  ✗ Generation failed: {result.get('error')}")
            return False
    
    except Exception as e:
        print(f"  ✗ Exception during generation: {e}")
        return False
    
    # Test 2: Unsafe query (should be blocked)
    print(f"\n  Test 2: Unsafe query (should block)")
    print(f"  Query: {UNSAFE_DIAGNOSIS}")
    
    try:
        result = generator.generate_answer(UNSAFE_DIAGNOSIS, verbose=False)
        
        if not result["success"] and result.get("error") == "unsafe_query":
            print(f"  ✓ Query blocked by safety filter")
            print(f"    Refusal: {result['answer'][:100]}...")
        else:
            print(f"  ✗ Unsafe query was not blocked!")
            return False
    
    except Exception as e:
        print(f"  ✗ Exception during safety check: {e}")
        return False
    
    print("\n  ✓ End-to-end tests passed")
    return True


def main():
    """Run all validation tests."""
    print("=" * 70)
    print("VALIDATION: STEP 5-8 - Answer Generation Pipeline")
    print("=" * 70)
    
    results = {}
    
    # Run tests
    results["Safety Filter"] = test_safety_filter()
    results["Prompts"] = test_prompts()
    results["LLM Client"] = test_llm_client()
    results["Validator"] = test_validator()
    results["End-to-End"] = test_end_to_end()
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    for test_name, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {status} - {test_name}")
    
    total_passed = sum(results.values())
    total_tests = len(results)
    
    print(f"\nTotal: {total_passed}/{total_tests} tests passed")
    
    if total_passed == total_tests:
        print("\n🎉 All tests passed! Pipeline is ready.")
    else:
        print("\n⚠ Some tests failed. Check output above.")
    
    print("=" * 70)
    
    return total_passed == total_tests


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
