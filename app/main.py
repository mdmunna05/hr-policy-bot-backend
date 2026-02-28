"""
main.py — FastAPI application for the HR Policy Bot backend

Endpoints:
  POST /ask          — RAG query (company + question → answer)
  POST /index        — Trigger one-time S3 → FAISS indexing
  GET  /companies    — List available indexed companies
  GET  /health       — Health check
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from app.rag import answer_query, list_companies
from app.indexing import build_indexes

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="HR Policy Bot API",
    description="Retrieval-Augmented Generation backend for company HR policies",
    version="2.0.0",
)

# Allow frontend origins (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Schemas ──────────────────────────────────────────────────────────────────
class QueryRequest(BaseModel):
    company: str
    query: str


class QueryResponse(BaseModel):
    company: str
    query: str
    answer: str
    sources: list[str]


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/companies")
def get_companies():
    """Return the list of companies that have a FAISS index available."""
    return {"companies": list_companies()}


@app.post("/index")
def run_indexing():
    """Trigger the S3 → FAISS indexing pipeline (one-time setup)."""
    try:
        build_indexes()
        return {"status": "Indexing completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=QueryResponse)
def ask(request: QueryRequest):
    """
    Validate company, retrieve context from FAISS,
    send to Groq LLM, and return the answer.
    """
    company = request.company.lower().strip()

    if company not in list_companies():
        raise HTTPException(
            status_code=404,
            detail=f"Company '{company}' not found. Available: {list_companies()}",
        )

    try:
        result = answer_query(company, request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
