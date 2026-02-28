"""
rag.py — Production-grade Retrieval-Augmented Generation engine

Loads a company-specific FAISS index, retrieves the top-k most relevant
chunks for a query, then sends them to Groq LLM for answer generation.

Features:
  - Company isolation (separate FAISS indexes)
  - No cross-company contamination
  - Strict informational mode (no approvals / payroll / exceptions)
  - Cosine similarity with threshold to prevent irrelevant retrieval
  - Multilingual support (10 languages)
  - Structured system + user message design
"""

import os
import pickle
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
from groq import Groq
from deep_translator import GoogleTranslator
from dotenv import load_dotenv

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────
INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "faiss_indexes")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")
TOP_K = int(os.getenv("TOP_K", "3"))
SIMILARITY_THRESHOLD = float(os.getenv("SIMILARITY_THRESHOLD", "0.35"))

# ── Supported languages ─────────────────────────────────────────────────────
LANGUAGES = {
    "English": "en",
    "Hindi": "hi",
    "Spanish": "es",
    "French": "fr",
    "German": "de",
    "Chinese": "zh-CN",
    "Japanese": "ja",
    "Korean": "ko",
    "Tamil": "ta",
    "Telugu": "te",
}

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


# ── Translation helpers ──────────────────────────────────────────────────────
def _translate(text: str, target_lang: str) -> str:
    """Translate text to target language. Returns original on failure."""
    if target_lang == "en":
        return text
    try:
        return GoogleTranslator(source="auto", target=target_lang).translate(text)
    except Exception:
        return text


# ── Helpers ──────────────────────────────────────────────────────────────────
def list_companies() -> list[str]:
    """Return sorted list of companies that have a FAISS index on disk."""
    if not os.path.isdir(INDEX_DIR):
        return []
    return sorted(
        f.replace(".index", "")
        for f in os.listdir(INDEX_DIR)
        if f.endswith(".index")
    )


def list_languages() -> list[str]:
    """Return list of supported language names."""
    return list(LANGUAGES.keys())


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
    """
    Return the top-k most relevant text chunks for the query.
    Uses cosine similarity (IndexFlatIP on normalized vectors).
    Filters out results below SIMILARITY_THRESHOLD.
    """
    model = _get_embed_model()
    index, chunks = _load_index(company)

    query_vec = model.encode([query])
    query_vec = np.array(query_vec, dtype="float32")
    faiss.normalize_L2(query_vec)

    scores, indices = index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if 0 <= idx < len(chunks) and score >= SIMILARITY_THRESHOLD:
            results.append(chunks[idx])
    return results


def answer_query(company: str, query: str, language: str = "English") -> dict:
    """
    Hybrid RAG pipeline with multilingual support:
    1. Translate query to English (if needed)
    2. Try to retrieve relevant chunks from the company's FAISS index
    3. If policy chunks found → answer from context (RAG mode)
    4. If no relevant chunks → answer using LLM general knowledge (open mode)
    5. Translate answer back to requested language
    6. Return answer with sources
    """
    company = company.lower().strip()
    lang_code = LANGUAGES.get(language, "en")

    # ── Translate query to English for embedding ─────────────────────────
    query_en = _translate(query, "en") if lang_code != "en" else query

    # ── Retrieve ─────────────────────────────────────────────────────────
    chunks = _retrieve(query_en, company)

    client = _get_groq_client()

    if chunks:
        # ── RAG MODE: Answer from policy context ────────────────────────
        context = "\n\n".join(chunks)

        system_prompt = f"""You are the official HR Policy Assistant for {company.title()}.

Your role is to explain HR policies, onboarding processes, leave policies,
attendance rules, benefits, compliance, and workplace conduct.

STRICT RULES:
- You provide information only.
- You cannot approve leave.
- You cannot process payroll.
- You cannot modify employee records.
- You cannot invent policies.
- You must answer ONLY using the context below.
- If information is not available in the context, say:
  "This information is not available in the provided company policy."

STYLE:
- Clear, friendly, professional tone.
- Start with a direct answer.
- Use bullet points where applicable.
- Keep responses concise and structured.
- No assumptions beyond the provided context.

Company: {company.title()}

Context:
{context}
"""
        user_message = f"Question: {query_en}"

    else:
        # ── OPEN MODE: Answer using general knowledge ───────────────────
        system_prompt = f"""You are a smart, helpful assistant working inside an HR Policy Bot for {company.title()}.

You can answer general questions on any topic — coding, math, science, history,
technology, career advice, interview tips, general HR concepts, etc.

RULES:
- Be helpful and accurate.
- If the question is about a specific {company.title()} internal policy that you
  don't have data for, say: "I don't have specific {company.title()} policy data
  for this, but here's some general information."
- Keep a professional, friendly tone.
- Use bullet points and clear structure.
"""
        user_message = query_en
        chunks = []

    # ── Generate ─────────────────────────────────────────────────────────
    chat = client.chat.completions.create(
        model=GROQ_MODEL,
        temperature=0.2,
        top_p=0.9,
        max_tokens=500,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
    )

    answer = chat.choices[0].message.content

    # ── Translate answer if needed ───────────────────────────────────────
    if lang_code != "en":
        answer = _translate(answer, lang_code)

    return {
        "company": company,
        "query": query,
        "answer": answer,
        "sources": chunks,
        "language": language,
    }
