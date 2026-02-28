"""
main.py — FastAPI application for the HR Policy Bot backend

Endpoints:
  POST /ask          — RAG query (company + question → answer)
  POST /index        — Trigger one-time S3 → FAISS indexing
  GET  /companies    — List available indexed companies
  GET  /health       — Health check
"""

import os
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

from app.rag import ask
from app.indexing import build_indexes

# ── App ──────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="HR Policy Bot API",
    description="Retrieval-Augmented Generation backend for company HR policies",
    version="1.0.0",
)

# Allow frontend origins (adjust in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

INDEX_DIR = os.path.join(os.path.dirname(__file__), "..", "faiss_indexes")


# ── Schemas ──────────────────────────────────────────────────────────────────
class AskRequest(BaseModel):
    company: str
    query: str


class AskResponse(BaseModel):
    company: str
    query: str
    answer: str
    sources: list[str]


# ── Routes ───────────────────────────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/companies")
def list_companies():
    """Return the list of companies that have a FAISS index available."""
    if not os.path.isdir(INDEX_DIR):
        return {"companies": []}

    companies = sorted(
        f.replace(".index", "")
        for f in os.listdir(INDEX_DIR)
        if f.endswith(".index")
    )
    return {"companies": companies}


@app.post("/index")
def run_indexing():
    """Trigger the S3 → FAISS indexing pipeline (one-time setup)."""
    try:
        build_indexes()
        return {"status": "Indexing completed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/ask", response_model=AskResponse)
def ask_question(req: AskRequest):
    """
    Accept a company name + question, retrieve context from FAISS,
    and return the Groq-generated answer.
    """
    try:
        result = ask(company=req.company, query=req.query)
        return result
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
