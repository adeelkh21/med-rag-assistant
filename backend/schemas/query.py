"""Request/Response Schemas"""
from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class QuestionRequest(BaseModel):
    """User question request"""
    question: str = Field(..., min_length=1, max_length=500)
    retrieval_method: Literal['dense', 'bm25', 'hybrid'] = Field(default='dense', description="Retrieval method to use")

class Citation(BaseModel):
    """Citation information"""
    chunk_id: str
    source: str
    topic: str
    text: Optional[str] = None
    similarity_score: Optional[float] = None

class SafetyInfo(BaseModel):
    """Safety check result"""
    is_refused: bool
    reason: Optional[str] = None

class AnswerResponse(BaseModel):
    """RAG answer response"""
    answer: str
    citations: List[Citation]
    retrieved_chunks: List[str]
    safety: SafetyInfo
    disclaimer: str = "For educational purposes only. This is not medical advice."
