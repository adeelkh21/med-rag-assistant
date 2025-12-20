"""
STEP 5-8: Complete Answer Generation Pipeline

Integrates:
- STEP 5: Safety gate (pre-LLM filtering)
- STEP 6: Prompt engineering
- STEP 7: Groq API (LLaMA-3 70B)
- STEP 8: Response validation

End-to-end pipeline: Query → Safety Check → Retrieval → Generation → Validation → Answer
"""

from typing import Optional, Dict, Any, List, Tuple
from pathlib import Path
import os
import re

# Load environment variables from .env file
try:
    from dotenv import load_dotenv
    project_root = Path(__file__).parent.parent
    env_path = project_root / ".env"
    load_dotenv(dotenv_path=env_path)
except ImportError:
    pass

from retrieval.retriever import MedicalRetriever
from generation.safety_filter import filter_query, get_refusal_response
from generation.prompts import build_user_prompt, get_system_prompt, get_mandatory_disclaimer
from generation.llm_client import GroqClient
from generation.validator import validate_response, get_citations_summary
from generation.citation_checker import validate_citations, get_validation_summary
from generation.uncertainty_handler import handle_low_confidence


CASUAL_PATTERNS = [
    r"^(hi|hey|hello|yo)\b",
    r"^good\s+(morning|afternoon|evening)\b",
    r"\bhow are you\b",
    r"\bwhat's up\b",
    r"\bhow's it going\b",
    r"\bthank(s| you)\b",
]


def _is_casual_greeting(text: str) -> bool:
    """Detect short, non-medical greetings to reply warmly without RAG."""
    if not text:
        return False
    normalized = text.strip().lower()
    return any(re.search(pattern, normalized) for pattern in CASUAL_PATTERNS)


def _friendly_greeting_response() -> str:
    return (
        "Hey there! I'm here to help with health questions whenever you need. "
        "What would you like to explore today?"
    )


def _convert_citations_to_numbers(answer: str, retrieved_docs: List[Dict[str, Any]]) -> str:
    """
    Convert chunk ID citations to numbered citations [1], [2], etc.
    
    Transforms:
    "Text (CHUNK_ID: WHO_01) more text (CDC_02)"
    To:
    "Text [1] more text [2]"
    
    Args:
        answer: Generated answer with chunk IDs
        retrieved_docs: Retrieved documents with IDs
    
    Returns:
        Answer with numbered citations
    """
    # Build mapping of chunk IDs to their position in retrieved_docs
    chunk_id_to_number = {}
    for i, doc in enumerate(retrieved_docs, 1):
        chunk_id_to_number[doc["id"]] = i
    
    # Replace chunk ID citations with numbered citations
    def replace_citation(match):
        chunk_id = match.group(1)
        if chunk_id in chunk_id_to_number:
            return f"[{chunk_id_to_number[chunk_id]}]"
        return ""  # Remove citation if not found in retrieved docs
    
    # Pattern 1: (CHUNK_ID: XXX) format - the main format from LLM
    result = re.sub(r'\(CHUNK_ID:\s*([A-Za-z0-9_]+)\)', replace_citation, answer)
    
    # Pattern 2: (XXX_YYY_ZZ) format - chunk IDs always have underscores and are UPPERCASE with numbers
    # Must have at least one underscore to distinguish from words like (UV) or (CHD)
    result = re.sub(r'\(([A-Z]+_[A-Za-z0-9_]+)\)', replace_citation, result)
    
    return result


class MedicalAnswerGenerator:
    """
    Complete medical RAG system with safety, retrieval, generation, and validation.
    
    Pipeline:
    1. Safety gate: Block unsafe queries
    2. Retrieval: Get top-k relevant documents
    3. Prompt construction: Build context + query prompt
    4. Generation: Call Groq LLaMA-3 70B
    5. Validation: Check citations and disclaimer
    6. Return: Final answer or error
    """
    
    def __init__(
        self,
        retriever: Optional[MedicalRetriever] = None,
        groq_client: Optional[GroqClient] = None,
        top_k: int = 6,
        max_retries: int = 2,
        uncertainty_threshold: float = 0.25,
        enable_citation_checking: bool = True,
        enable_uncertainty_handling: bool = True
    ):
        """
        Initialize answer generator.
        
        Args:
            retriever: MedicalRetriever instance (or creates new one)
            groq_client: GroqClient instance (or creates new one)
            top_k: Number of documents to retrieve (default: 6)
            max_retries: Max regeneration attempts if validation fails
            uncertainty_threshold: Confidence threshold for low-confidence detection (default: 0.25)
            enable_citation_checking: Enable citation validation (default: True)
            enable_uncertainty_handling: Enable low-confidence fallbacks (default: True)
        """
        print("Initializing MedicalAnswerGenerator...")
        
        # Initialize retriever
        if retriever is None:
            print("  Loading retriever...")
            self.retriever = MedicalRetriever()
        else:
            self.retriever = retriever
        
        # Initialize LLM client
        if groq_client is None:
            print("  Initializing Groq client...")
            self.groq_client = GroqClient()
        else:
            self.groq_client = groq_client
        
        self.top_k = top_k
        self.max_retries = max_retries
        self.uncertainty_threshold = uncertainty_threshold
        self.enable_citation_checking = enable_citation_checking
        self.enable_uncertainty_handling = enable_uncertainty_handling
        
        print("[OK] MedicalAnswerGenerator ready\n")
    
    def generate_answer(
        self,
        query: str,
        temperature: float = 0.1,
        verbose: bool = False
    ) -> Dict[str, Any]:
        """
        Generate citation-grounded answer for medical query.
        
        Full pipeline with safety, retrieval, generation, and validation.
        
        Args:
            query: User medical question
            temperature: LLM temperature (0.0-0.2 recommended for determinism)
            verbose: Print progress information
        
        Returns:
            Dictionary with:
            {
                "success": bool,
                "answer": str (if success),
                "error": str (if failure),
                "query": str,
                "retrieved_docs": List[Dict],
                "citations_used": List[str],
                "validation_passed": bool
            }
        """
        result = {
            "success": False,
            "query": query,
            "answer": None,
            "error": None,
            "retrieved_docs": None,
            "citations_used": None,
            "validation_passed": False
        }
        
        if verbose:
            print(f"Query: {query}\n")

        # Handle casual greetings with a direct, friendly response
        if _is_casual_greeting(query):
            friendly_answer = _friendly_greeting_response()
            if verbose:
                print("[0/5] Casual greeting detected - returning friendly reply\n")
            return {
                "success": True,
                "query": query,
                "answer": friendly_answer,
                "error": None,
                "retrieved_docs": [],
                "citations_used": [],
                "validation_passed": True,
            }
        
        # STEP 5: Safety Gate
        if verbose:
            print("[1/5] Safety check...")
        
        should_proceed, refusal = filter_query(query)
        if not should_proceed:
            if verbose:
                print("  [BLOCKED] Query blocked by safety filter\n")
            
            result["success"] = False
            result["answer"] = refusal
            result["error"] = "unsafe_query"
            return result
        
        if verbose:
            print("  [OK] Query is safe\n")
        
        # STEP 2 (from earlier): Retrieval
        if verbose:
            print(f"[2/5] Retrieving top-{self.top_k} documents...")
        
        try:
            retrieved_docs = self.retriever.retrieve(query, k=self.top_k)
            result["retrieved_docs"] = retrieved_docs
            
            if verbose:
                print(f"  [OK] Retrieved {len(retrieved_docs)} documents")
                print(f"  Top score: {retrieved_docs[0]['score']:.4f}\n")
        
        except Exception as e:
            if verbose:
                print(f"  [ERROR] Retrieval failed: {e}\n")
            
            result["error"] = f"retrieval_error: {e}"
            return result
        
        # Check for low-confidence retrieval (uncertainty handling)
        if self.enable_uncertainty_handling:
            if verbose:
                print("[2.5/5] Checking retrieval confidence...")
            
            fallback_response = handle_low_confidence(
                query=query,
                retrieved_docs=retrieved_docs,
                low_threshold=self.uncertainty_threshold,
                fallback_type="clarifying"
            )
            
            if fallback_response:
                if verbose:
                    print(f"  [WARNING] Low confidence detected (threshold={self.uncertainty_threshold})")
                    print(f"  -> Returning safe fallback instead of generating\n")
                
                result["success"] = True
                result["answer"] = fallback_response
                result["validation_passed"] = True
                result["low_confidence"] = True
                return result
            
            if verbose:
                print("  [OK] Confidence sufficient for generation\n")
        
        # STEP 6: Prompt Construction
        if verbose:
            print("[3/5] Building prompt...")
        
        system_prompt = get_system_prompt()
        user_prompt = build_user_prompt(query, retrieved_docs)
        
        if verbose:
            print(f"  [OK] Prompt ready (context: {len(retrieved_docs)} chunks)\n")
        
        # STEP 7: Generation (with retries)
        answer = None
        validation_errors = []
        
        for attempt in range(self.max_retries + 1):
            if verbose and attempt > 0:
                print(f"  Retry {attempt}/{self.max_retries}...")
            
            if verbose:
                print(f"[4/5] Generating answer...")
            
            try:
                answer = self.groq_client.generate(
                    system_prompt=system_prompt,
                    user_prompt=user_prompt,
                    temperature=temperature
                )
                
                if verbose:
                    print(f"  [OK] Answer generated ({len(answer)} chars)\n")
            
            except Exception as e:
                if verbose:
                    print(f"  [ERROR] Generation failed: {e}\n")
                
                result["error"] = f"generation_error: {e}"
                return result
            
            # STEP 8: Validation
            if verbose:
                print("[5/5] Validating answer...")
            
            # Run standard validation (format check)
            is_valid, validation_errors = validate_response(answer, retrieved_docs)
            
            # Run citation checker if enabled
            if self.enable_citation_checking and is_valid:
                citation_valid, citation_errors = validate_citations(
                    answer=answer,
                    retrieved_docs=retrieved_docs,
                    min_keyword_overlap=0.3
                )
                
                if not citation_valid:
                    is_valid = False
                    validation_errors.extend([
                        err["reason"] for err in citation_errors
                    ])
                    
                    if verbose:
                        print(f"  [ERROR] Citation validation failed:")
                        for err in citation_errors:
                            print(f"    - {err['reason']}")
            
            if is_valid:
                if verbose:
                    print("  [OK] Validation passed\n")
                break
            else:
                if verbose:
                    print(f"  [ERROR] Validation failed: {validation_errors}")
                    if attempt < self.max_retries:
                        print(f"  -> Regenerating...\n")
                    else:
                        print(f"  -> Max retries reached\n")
        
        # Final result
        if is_valid:
            result["success"] = True
            # Convert chunk IDs to numbered citations
            result["answer"] = _convert_citations_to_numbers(answer, retrieved_docs)
            result["validation_passed"] = True
            
            # Add citation summary
            citation_summary = get_citations_summary(answer, retrieved_docs)
            result["citations_used"] = citation_summary["cited_doc_ids"]
        
        else:
            result["success"] = False
            result["error"] = f"validation_failed: {validation_errors}"
            result["answer"] = answer  # Include invalid answer for debugging
            result["validation_passed"] = False
        
        return result
    
    def answer(self, query: str, verbose: bool = True) -> str:
        """
        Simplified interface: Return answer text directly.
        
        Args:
            query: User question
            verbose: Print progress
        
        Returns:
            Answer text (or error message)
        """
        result = self.generate_answer(query, verbose=verbose)
        
        if result["success"]:
            return result["answer"]
        else:
            # Return error or refusal
            if result.get("answer"):
                return result["answer"]  # Safety refusal
            else:
                return f"Error: {result.get('error', 'Unknown error')}"


def create_generator(
    index_path: str = "retrieval/index.faiss",
    metadata_path: str = "retrieval/metadata_lookup.pkl",
    top_k: int = 6
) -> MedicalAnswerGenerator:
    """
    Convenience function to create answer generator.
    
    Args:
        index_path: Path to FAISS index
        metadata_path: Path to metadata lookup
        top_k: Number of documents to retrieve
    
    Returns:
        MedicalAnswerGenerator instance
    """
    return MedicalAnswerGenerator(top_k=top_k)


# Interactive demo
def main():
    """Interactive demo of the answer generator."""
    import sys
    import os
    
    print("=" * 70)
    print("Medical RAG System - Interactive Demo")
    print("=" * 70)
    
    # Check for API key
    if not os.getenv("GROQ_API_KEY"):
        print("\n✗ GROQ_API_KEY environment variable not set")
        print("  Set it with: export GROQ_API_KEY='your-key-here'")
        sys.exit(1)
    
    # Initialize generator
    try:
        generator = create_generator()
    except Exception as e:
        print(f"\n✗ Failed to initialize generator: {e}")
        sys.exit(1)
    
    # Example queries
    examples = [
        "What are the symptoms of type 2 diabetes?",
        "How is lung cancer treated?",
        "Do I have diabetes?",  # Should be blocked
    ]
    
    print("\nExample queries:")
    for i, q in enumerate(examples, 1):
        print(f"  {i}. {q}")
    
    print("\n" + "=" * 70)
    
    # Interactive loop
    while True:
        print("\nEnter a medical question (or 'quit', 'example N'):")
        user_input = input("> ").strip()
        
        if not user_input:
            continue
        
        if user_input.lower() in ["quit", "exit", "q"]:
            break
        
        # Handle examples
        if user_input.lower().startswith("example"):
            try:
                idx = int(user_input.split()[1]) - 1
                query = examples[idx]
            except (IndexError, ValueError):
                print("Invalid example number")
                continue
        else:
            query = user_input
        
        print("\n" + "=" * 70)
        
        # Generate answer
        result = generator.generate_answer(query, verbose=True)
        
        print("=" * 70)
        print("RESULT:")
        print("=" * 70)
        
        if result["success"]:
            print(f"\n{result['answer']}\n")
            
            if result.get("citations_used"):
                print(f"Citations used: {len(result['citations_used'])}")
                print(f"  {', '.join(result['citations_used'][:5])}...")
        else:
            if result.get("answer"):
                # Safety refusal
                print(f"\n{result['answer']}\n")
            else:
                print(f"\n✗ Error: {result.get('error')}\n")
        
        print("=" * 70)


if __name__ == "__main__":
    main()
