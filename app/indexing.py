"""
indexing.py — One-time S3 → FAISS indexing process

Connects to S3, fetches all .txt HR policy files,
chunks the text, creates embeddings, and saves one FAISS index per company.
"""

import os
import re
import boto3
import faiss
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

load_dotenv()

# ── Config ───────────────────────────────────────────────────────────────────
S3_BUCKET = os.getenv("S3_BUCKET", "hr-policy-bot-documents")
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
INDEX_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "faiss_indexes")
EMBED_MODEL = os.getenv("EMBED_MODEL", "all-MiniLM-L6-v2")
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "50"))


# ── Helpers ──────────────────────────────────────────────────────────────────
def _get_s3_client():
    return boto3.client("s3", region_name=AWS_REGION)


def _company_from_filename(filename: str) -> str:
    """
    Extract company name from filename.
    e.g. 'google_hr_knowledge_base.txt' → 'google'
    """
    return filename.split("_")[0].lower()


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """
    Split text into overlapping chunks of approximately `chunk_size` characters.
    """
    text = re.sub(r"\n{3,}", "\n\n", text.strip())
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        start += chunk_size - overlap
    return chunks


# ── Main indexing logic ──────────────────────────────────────────────────────
def build_indexes():
    """
    Fetch all .txt files from S3, build a FAISS index per company, and persist
    each index + its metadata to disk.
    """
    s3 = _get_s3_client()
    model = SentenceTransformer(EMBED_MODEL)
    os.makedirs(INDEX_DIR, exist_ok=True)

    # List all .txt objects in the bucket
    response = s3.list_objects_v2(Bucket=S3_BUCKET)
    txt_files = [
        obj["Key"]
        for obj in response.get("Contents", [])
        if obj["Key"].endswith(".txt")
    ]

    if not txt_files:
        print("⚠️  No .txt files found in S3 bucket:", S3_BUCKET)
        return

    print(f"📄  Found {len(txt_files)} file(s) in s3://{S3_BUCKET}")

    for key in txt_files:
        filename = os.path.basename(key)
        company = _company_from_filename(filename)
        print(f"\n🔧  Processing: {filename}  →  company = {company}")

        # Download file content
        obj = s3.get_object(Bucket=S3_BUCKET, Key=key)
        text = obj["Body"].read().decode("utf-8")

        # Chunk
        chunks = _chunk_text(text)
        print(f"   📦  {len(chunks)} chunks created")

        if not chunks:
            print("   ⚠️  Skipping — no usable text.")
            continue

        # Embed
        embeddings = model.encode(chunks, show_progress_bar=True)
        embeddings = np.array(embeddings, dtype="float32")

        # Build FAISS index (Inner Product after L2-normalising = cosine similarity)
        faiss.normalize_L2(embeddings)
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)

        # Save index
        index_path = os.path.join(INDEX_DIR, f"{company}.index")
        faiss.write_index(index, index_path)

        # Save chunk metadata alongside the index
        meta_path = os.path.join(INDEX_DIR, f"{company}_meta.pkl")
        with open(meta_path, "wb") as f:
            pickle.dump({"company": company, "chunks": chunks}, f)

        print(f"   ✅  Saved {index_path}  ({index.ntotal} vectors, dim={dim})")

    print("\n🎉  Indexing complete!")


# ── CLI entry-point ──────────────────────────────────────────────────────────
if __name__ == "__main__":
    build_indexes()
