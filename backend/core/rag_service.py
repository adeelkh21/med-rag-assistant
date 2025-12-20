"""
RAG Service - Orchestrates the full pipeline

Uses the existing MedicalAnswerGenerator which already has:
- Safety filtering
- Retrieval
- LLM generation
- Validation
"""

from generation.answer_generator import MedicalAnswerGenerator
from retrieval.bm25_retriever import BM25Retriever
from retrieval.hybrid_retriever import HybridRetriever
from retrieval.retriever import MedicalRetriever
from backend.schemas.query import AnswerResponse, Citation, SafetyInfo


class RAGService:
    """
    Production RAG Service
    
    Wraps MedicalAnswerGenerator and converts output to API format.
    Supports multiple retrieval methods: dense, bm25, hybrid.
    """
    
    def __init__(self):
        print("Initializing RAG Service...")
        self.generator_dense = None
        self.generator_bm25 = None
        self.generator_hybrid = None
        print("[OK] RAG Service ready (lazy loading)")
    
    def _initialize_dense(self):
        """Lazy initialization of dense generator"""
        if self.generator_dense is None:
            print("Loading Dense MedicalAnswerGenerator...")
            self.generator_dense = MedicalAnswerGenerator()
    
    def _initialize_bm25(self):
        """Lazy initialization of BM25 generator"""
        if self.generator_bm25 is None:
            print("Loading BM25 MedicalAnswerGenerator...")
            bm25_retriever = BM25Retriever()
            self.generator_bm25 = MedicalAnswerGenerator(retriever=bm25_retriever)
    
    def _initialize_hybrid(self):
        """Lazy initialization of hybrid generator"""
        if self.generator_hybrid is None:
            print("Loading Hybrid MedicalAnswerGenerator...")
            hybrid_retriever = HybridRetriever(alpha=0.5)
            self.generator_hybrid = MedicalAnswerGenerator(retriever=hybrid_retriever)

    def answer_question(self, question: str, retrieval_method: str = 'dense') -> dict:
        """
        Backward-compatible interface used by the API layer.
        Maps the generator output into the legacy dict expected by the router.
        
        Args:
            question: User question
            retrieval_method: One of 'dense', 'bm25', 'hybrid'
        """
        print(f"[RAGService] Using retrieval method: {retrieval_method}")
        
        # Initialize appropriate generator
        if retrieval_method == 'bm25':
            self._initialize_bm25()
            generator = self.generator_bm25
            print("[RAGService] Using BM25 retriever")
        elif retrieval_method == 'hybrid':
            self._initialize_hybrid()
            generator = self.generator_hybrid
            print("[RAGService] Using Hybrid retriever")
        else:  # default to dense
            self._initialize_dense()
            generator = self.generator_dense
            print("[RAGService] Using Dense retriever")

        result = generator.generate_answer(query=question, verbose=False)
        error = result.get("error")
        is_refused = error == "unsafe_query" or error is not None

        error_msg = result.get("error")
        # Always return a string for Pydantic validation. Bubble up the actual error if present
        answer_text = result.get("answer")
        if not answer_text:
            if error_msg:
                answer_text = f"Could not generate answer: {error_msg}"
            else:
                answer_text = "Sorry, I couldn't generate an answer right now. Please try again."

        retrieved_docs = result.get("retrieved_docs", []) or []

        # Build citations from retrieved docs (filter to cited ones when available)
        cited_ids = set(result.get("citations_used") or [])
        use_filter = len(cited_ids) > 0

        citations = []
        for doc in retrieved_docs:
            doc_id = doc.get("id", "unknown")
            if use_filter and doc_id not in cited_ids:
                continue
            meta = doc.get("metadata", {}) or {}
            citations.append(
                {
                    "doc_id": doc_id,
                    "source": meta.get("source", "unknown"),
                    "topic": meta.get("topic", "unknown"),
                    "text": doc.get("text", ""),
                    "similarity_score": doc.get("score"),
                }
            )

        return {
            "answer": answer_text,
            "citations": citations,
            "retrieved_chunks": [doc.get("id", "unknown") for doc in retrieved_docs],
            "safety": {
                "is_refused": is_refused,
                "reason": error_msg if is_refused else None,
            },
        }
    
    async def process_question(self, question: str) -> AnswerResponse:
        """
        Process a medical question through the complete RAG pipeline.
        
        Args:
            question: User's medical question
        
        Returns:
            AnswerResponse with answer, citations, safety info, and disclaimer
        """
        # Initialize on first use
        self._initialize()
        
        # Generate answer using existing pipeline
        result = self.generator.generate_answer(query=question, verbose=False)
        
        # Check if query was refused by safety filter
        is_refused = result.get("safety", {}).get("is_refused", False)
        
        # Convert citations to API format
        citations = [
            Citation(
                chunk_id=cit.get("doc_id", "unknown"),
                text=cit.get("text", ""),
                similarity_score=cit.get("score", 0.0)
            )
            for cit in result.get("citations", [])
        ]
        
        # Build response
        return AnswerResponse(
            answer=result["answer"],
            citations=citations,
            safety=SafetyInfo(
                is_safe=not is_refused,
                reason=result.get("safety", {}).get("reason") if is_refused else None
            ),
            disclaimer="This information is for educational purposes only and should not be used as a substitute for professional medical advice."
        )

