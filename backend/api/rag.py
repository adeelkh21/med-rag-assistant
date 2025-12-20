"""RAG API Endpoints"""
from fastapi import APIRouter, HTTPException
from backend.schemas.query import QuestionRequest, AnswerResponse, Citation, SafetyInfo
from backend.core.rag_service import RAGService

router = APIRouter(tags=["RAG"])

# Initialize RAG service (singleton)
rag_service = None

def get_rag_service():
    """Lazy initialize RAG service"""
    global rag_service
    if rag_service is None:
        rag_service = RAGService()
    return rag_service

@router.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Ask a medical guidance question
    
    Execution order:
    1. Safety filter (FIRST)
    2. Retriever (if safe) - Uses selected retrieval method
    3. LLM generation (if safe)
    4. Response assembly
    """
    try:
        print(f"[API] Received request with retrieval_method: {request.retrieval_method}")
        service = get_rag_service()
        result = service.answer_question(request.question, request.retrieval_method)
        
        # Convert to response schema
        return AnswerResponse(
            answer=result["answer"],
            citations=[
                Citation(
                    chunk_id=cit["doc_id"],
                    source=cit["source"],
                    topic=cit["topic"],
                    text=cit.get("text"),
                    similarity_score=cit.get("similarity_score")
                )
                for cit in result["citations"]
            ],
            retrieved_chunks=result["retrieved_chunks"],
            safety=SafetyInfo(
                is_refused=result["safety"]["is_refused"],
                reason=result["safety"].get("reason")
            )
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
