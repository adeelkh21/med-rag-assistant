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
        max_retries: int = 2
    ):
        """
        Initialize answer generator.
        
        Args:
            retriever: MedicalRetriever instance (or creates new one)
            groq_client: GroqClient instance (or creates new one)
            top_k: Number of documents to retrieve (default: 6)
            max_retries: Max regeneration attempts if validation fails
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
        
        print("✓ MedicalAnswerGenerator ready\n")
    
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
        
        # STEP 5: Safety Gate
        if verbose:
            print("[1/5] Safety check...")
        
        should_proceed, refusal = filter_query(query)
        if not should_proceed:
            if verbose:
                print("  ✗ Query blocked by safety filter\n")
            
            result["success"] = False
            result["answer"] = refusal
            result["error"] = "unsafe_query"
            return result
        
        if verbose:
            print("  ✓ Query is safe\n")
        
        # STEP 2 (from earlier): Retrieval
        if verbose:
            print(f"[2/5] Retrieving top-{self.top_k} documents...")
        
        try:
            retrieved_docs = self.retriever.retrieve(query, k=self.top_k)
            result["retrieved_docs"] = retrieved_docs
            
            if verbose:
                print(f"  ✓ Retrieved {len(retrieved_docs)} documents")
                print(f"  Top score: {retrieved_docs[0]['score']:.4f}\n")
        
        except Exception as e:
            if verbose:
                print(f"  ✗ Retrieval failed: {e}\n")
            
            result["error"] = f"retrieval_error: {e}"
            return result
        
        # STEP 6: Prompt Construction
        if verbose:
            print("[3/5] Building prompt...")
        
        system_prompt = get_system_prompt()
        user_prompt = build_user_prompt(query, retrieved_docs)
        
        if verbose:
            print(f"  ✓ Prompt ready (context: {len(retrieved_docs)} chunks)\n")
        
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
                    print(f"  ✓ Answer generated ({len(answer)} chars)\n")
            
            except Exception as e:
                if verbose:
                    print(f"  ✗ Generation failed: {e}\n")
                
                result["error"] = f"generation_error: {e}"
                return result
            
            # STEP 8: Validation
            if verbose:
                print("[5/5] Validating answer...")
            
            is_valid, validation_errors = validate_response(answer, retrieved_docs)
            
            if is_valid:
                if verbose:
                    print("  ✓ Validation passed\n")
                break
            else:
                if verbose:
                    print(f"  ✗ Validation failed: {validation_errors}")
                    if attempt < self.max_retries:
                        print(f"  → Regenerating...\n")
                    else:
                        print(f"  → Max retries reached\n")
        
        # Final result
        if is_valid:
            result["success"] = True
            result["answer"] = answer
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
