from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from typing import List
import tempfile
import os

from pdf_reader import PDFReader
from text_cleaner import TextProcessor
from embedder import Embedder
from vector_store import VectorStore
from llm import LLM
from schemas import QueryResponse, UploadResponse, HealthResponse

# ─── App setup ────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Intelligent Document Processing (IDP)",
    description="Phase 2 RAG: Llama understands your documents",
    version="2.2.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Initialize components ────────────────────────────────────────────────────

processor = TextProcessor()
embedder = Embedder()
vector_db = VectorStore(dim=384)
llm = LLM(model="llama3.2:3b")

# ─── Trigger words that switch to detailed mode ───────────────────────────────

DETAIL_TRIGGERS = [
    "explain", "describe", "summarize", "summary", "detail",
    "tell me about", "what is the", "how does", "why", "elaborate",
    "full", "complete", "all", "everything about"
]

def is_detailed_request(question: str) -> bool:
    q = question.lower()
    return any(trigger in q for trigger in DETAIL_TRIGGERS)

# ─── Routes ───────────────────────────────────────────────────────────────────

@app.get("/health", response_model=HealthResponse)
def health():
    return HealthResponse(
        status="ok",
        ollama_connected=llm.health_check(),
        chroma_chunks=vector_db.count(),
        model=llm.model
    )


@app.post("/upload", response_model=UploadResponse)
async def upload_pdfs(files: List[UploadFile] = File(...)):
    """
    Upload one or more PDFs. Each chunk is stored with its source filename.
    Clears previous index on each upload batch.
    """
    total_chunks = 0
    file_names = []

    vector_db.clear()

    for file in files:
        if not file.filename.endswith(".pdf"):
            raise HTTPException(
                status_code=400,
                detail=f"'{file.filename}' is not a PDF."
            )

        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            reader = PDFReader(tmp_path)
            raw_text = reader.extract_text()

            if not raw_text:
                raise HTTPException(
                    status_code=422,
                    detail=f"Could not extract text from '{file.filename}'."
                )

            clean_text = processor.clean_text(raw_text)
            chunks = processor.chunk_text(clean_text, chunk_size=300, overlap=50)
            embeddings = embedder.embed(chunks)

            vector_db.add(
                embeddings=embeddings,
                chunks=chunks,
                metadata=[
                    {"source": file.filename, "chunk_index": i}
                    for i in range(len(chunks))
                ]
            )

            total_chunks += len(chunks)
            file_names.append(file.filename)

        finally:
            os.unlink(tmp_path)

    return UploadResponse(
        status="success",
        chunks_indexed=total_chunks,
        message=f"Indexed {total_chunks} chunks from {len(files)} file(s): {', '.join(file_names)}"
    )


@app.post("/query", response_model=QueryResponse)
def query(question: str):
    """
    Ask any question about the indexed PDFs.

    Auto-detects if the user wants a short or detailed answer based
    on trigger words like 'explain', 'summarize', 'describe' etc.
    """
    if vector_db.count() == 0:
        raise HTTPException(
            status_code=400,
            detail="No PDFs indexed yet. POST to /upload first."
        )

    # Embed + retrieve top relevant chunks
    query_embedding = embedder.embed([question])[0]
    chunks = vector_db.search(query_embedding, k=5)
    sources = vector_db.get_sources(query_embedding, k=5)

    # Join chunks into context — give Llama as much relevant text as possible
    context = "\n\n".join(chunks)

    # Detect if user wants a detailed answer
    detailed = is_detailed_request(question)

    # Ask Llama
    answer = llm.generate_answer(question, context, detailed=detailed)

    return QueryResponse(
        query=question,
        answer=answer,
        detailed=detailed,
        sources=sources,
        context_snippet=context[:300]
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)