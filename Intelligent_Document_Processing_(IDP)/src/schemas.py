from pydantic import BaseModel, Field
from typing import Optional, List


class QueryResponse(BaseModel):
    """
    Clean response model — no more rule-based intent fields.
    Everything goes through Llama now.
    """
    query: str = Field(..., description="Original user question")
    answer: str = Field(..., description="Answer from Llama")
    detailed: bool = Field(False, description="Whether this was a detailed response")
    sources: List[str] = Field(default=[], description="PDF filenames used to generate the answer")
    context_snippet: Optional[str] = Field(None, description="Relevant context used (first 300 chars)")


class UploadResponse(BaseModel):
    status: str
    chunks_indexed: int
    message: str


class HealthResponse(BaseModel):
    status: str
    ollama_connected: bool
    chroma_chunks: int
    model: str