"""
rag.py — Retrieval-Augmented Generation query engine

Loads a company-specific FAISS index, retrieves the top-k most relevant
chunks for a query, then sends them to Groq LLM for answer generation.
"""

import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from dotenv import load_dotenv

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────
INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "faiss_indexes")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
TOP_K = int(os.getenv("TOP_K", "3"))

# ── Singletons (loaded once, reused across requests) ────────────────────────
_embed_model: SentenceTransformer | None = None
_groq_client: Groq | None = None


def _get_embed_model() -> SentenceTransformer:
    global _embed_model
    if _embed_model is None:
        _embed_model = SentenceTransformer(EMBED_MODEL)
    return _embed_model


def _get_groq_client() -> Groq:
    global _groq_client
    if _groq_client is None:
        _groq_client = Groq(api_key=GROQ_API_KEY)
    return _groq_client


# ── Core functions ───────────────────────────────────────────────────────────
def _load_index(company: str):
    """Load the FAISS index and chunk metadata for the given company."""
    company = company.lower().strip()
    index_path = os.path.join(INDEX_DIR, f"{company}.index")
    meta_path = os.path.join(INDEX_DIR, f"{company}_meta.pkl")

    if not os.path.exists(index_path):
        raise FileNotFoundError(
            f"No FAISS index found for company '{company}'. "
            f"Expected file: {index_path}"
        )

    index = faiss.read_index(index_path)

    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    return index, meta["chunks"]


def _retrieve(query: str, company: str, top_k: int = TOP_K) -> list[str]:
    """Return the top-k most relevant text chunks for the query."""
    model = _get_embed_model()
    index, chunks = _load_index(company)

    query_vec = model.encode([query])
    query_vec = np.array(query_vec, dtype="float32")
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, top_k)

    results = []
    for idx in indices[0]:
        if 0 <= idx < len(chunks):
            results.append(chunks[idx])
    return results


def _build_prompt(query: str, context_chunks: list[str], company: str) -> str:
    """Construct the system + user prompt for the LLM."""
    context = "\n\n---\n\n".join(context_chunks)
    return (
        f"You are an expert HR assistant for {company.title()}.\n"
        f"Use ONLY the following HR policy excerpts to answer the question.\n"
        f"If the answer is not in the excerpts, say 'I don't have enough information "
        f"from the {company.title()} HR policy to answer that.'\n\n"
        f"### HR Policy Excerpts\n{context}\n\n"
        f"### Question\n{query}"
    )


def ask(company: str, query: str) -> dict:
    """
    End-to-end RAG pipeline:
    1. Retrieve relevant chunks from the company's FAISS index
    2. Send chunks + query to Groq LLM
    3. Return the answer
    """
    # Retrieve
    chunks = _retrieve(query, company)

    if not chunks:
        return {
            "company": company,
            "query": query,
            "answer": f"No relevant information found in {company.title()}'s HR policy.",
            "sources": [],
        }

    # Generate
    prompt = _build_prompt(query, chunks, company)
    client = _get_groq_client()

    chat = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": "You are a helpful HR policy assistant."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.3,
        max_tokens=1024,
    )

    answer = chat.choices[0].message.content

    return {
        "company": company,
        "query": query,
        "answer": answer,
        "sources": chunks,
    }
