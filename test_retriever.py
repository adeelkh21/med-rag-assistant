"""
Test script for STEP 4: Semantic Retriever
Tests retrieval functionality with sample medical queries.
"""

from retrieval.retriever import MedicalRetriever


def test_retriever():
    """Test retriever with various medical queries."""
    print("=" * 70)
    print("Testing Medical Retriever (STEP 4)")
    print("=" * 70)
    
    # Initialize retriever
    print("\nInitializing retriever...")
    retriever = MedicalRetriever()
    
    # Test queries
    test_queries = [
        "What are the symptoms of type 2 diabetes?",
        "How is lung cancer treated?",
        "What causes high blood pressure?",
        "Tell me about heart disease risk factors"
    ]
    
    print("\n" + "=" * 70)
    print("Running Test Queries")
    print("=" * 70)
    
    for query_num, query in enumerate(test_queries, 1):
        print(f"\n{'='*70}")
        print(f"Query {query_num}: {query}")
        print('='*70)
        
        # Retrieve top-3 results
        results = retriever.retrieve(query, k=3)
        
        print(f"\nTop-{len(results)} Results:\n")
        for i, doc in enumerate(results, 1):
            print(f"{i}. Score: {doc['score']:.4f}")
            print(f"   ID: {doc['id']}")
            print(f"   Topic: {doc['metadata']['topic'][:70]}")
            print(f"   Source: {doc['metadata']['source']}")
            print(f"   Text preview: {doc['text'][:120]}...")
            print()
    
    # Test determinism
    print("\n" + "=" * 70)
    print("Testing Determinism")
    print("=" * 70)
    
    query = "What are the symptoms of diabetes?"
    print(f"\nQuery: {query}")
    
    results1 = retriever.retrieve(query, k=5)
    results2 = retriever.retrieve(query, k=5)
    
    # Check if results are identical
    ids1 = [r['id'] for r in results1]
    ids2 = [r['id'] for r in results2]
    scores1 = [r['score'] for r in results1]
    scores2 = [r['score'] for r in results2]
    
    if ids1 == ids2 and scores1 == scores2:
        print("✓ PASS: Results are deterministic (identical on re-run)")
    else:
        print("✗ FAIL: Results differ between runs")
        print(f"  Run 1 IDs: {ids1[:3]}")
        print(f"  Run 2 IDs: {ids2[:3]}")
    
    # Test output format
    print("\n" + "=" * 70)
    print("Validating Output Format")
    print("=" * 70)
    
    result = results1[0]
    required_keys = {"id", "text", "score", "metadata"}
    metadata_keys = {"topic", "source", "source_type"}
    
    format_valid = True
    
    if not all(k in result for k in required_keys):
        print(f"✗ FAIL: Missing top-level keys. Expected {required_keys}, got {result.keys()}")
        format_valid = False
    
    if not all(k in result['metadata'] for k in metadata_keys):
        print(f"✗ FAIL: Missing metadata keys. Expected {metadata_keys}, got {result['metadata'].keys()}")
        format_valid = False
    
    if not isinstance(result['score'], (int, float)):
        print(f"✗ FAIL: Score must be numeric, got {type(result['score'])}")
        format_valid = False
    
    if format_valid:
        print("✓ PASS: Output format matches specification")
        print(f"\nSample output structure:")
        print(f"  id: {result['id']}")
        print(f"  score: {result['score']}")
        print(f"  text length: {len(result['text'])} chars")
        print(f"  metadata keys: {list(result['metadata'].keys())}")
    
    print("\n" + "=" * 70)
    print("✓ All Tests Complete")
    print("=" * 70)


if __name__ == "__main__":
    test_retriever()
