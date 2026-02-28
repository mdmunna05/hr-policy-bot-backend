# HR Policy Bot — Backend

RAG-powered HR policy Q&A system using **FastAPI**, **FAISS**, **SentenceTransformers**, **AWS S3**, and **Groq LLM**.

## Architecture

```
S3 (hr-policy-bot-documents)
        │
        ▼
   indexing.py ──► FAISS indexes (one per company)
        │
        ▼
    rag.py ──► Groq LLM ──► Answer
        │
        ▼
    main.py (FastAPI)
```

## Project Structure

```
hr-policy-bot-backend/
├── app/
│   ├── main.py        # FastAPI routes
│   ├── rag.py         # Retrieval + Groq generation
│   └── indexing.py    # S3 → FAISS index builder
├── faiss_indexes/     # Generated FAISS indexes (gitignored)
├── requirements.txt
├── .gitignore
└── README.md
```

## Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file:

```env
S3_BUCKET=hr-policy-bot-documents
AWS_REGION=us-east-1
GROQ_API_KEY=your_groq_api_key
GROQ_MODEL=llama-3.3-70b-versatile
EMBED_MODEL=all-MiniLM-L6-v2
```

### 3. Run indexing (one-time)

```bash
python -m app.indexing
```

Or via the API:

```bash
curl -X POST http://localhost:8000/index
```

### 4. Start the server

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## API Endpoints

| Method | Path         | Description                        |
| ------ | ------------ | ---------------------------------- |
| GET    | `/health`    | Health check                       |
| GET    | `/companies` | List indexed companies             |
| POST   | `/index`     | Trigger S3 → FAISS indexing        |
| POST   | `/ask`       | Ask a question about an HR policy  |

### POST `/ask` — Example

**Request:**
```json
{
  "company": "google",
  "query": "What is the leave policy?"
}
```

**Response:**
```json
{
  "company": "google",
  "query": "What is the leave policy?",
  "answer": "Google provides ...",
  "sources": ["chunk 1 ...", "chunk 2 ...", "chunk 3 ..."]
}
```

## EC2 Deployment

```bash
# SSH into EC2
sudo apt update && sudo apt install python3-pip python3-venv -y
git clone https://github.com/mdmunna05/hr-policy-bot-backend.git
cd hr-policy-bot-backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
# Create .env with your keys
python -m app.indexing          # one-time
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

## Companies Indexed

| Company    | S3 File                            |
| ---------- | ---------------------------------- |
| Accenture  | accenture_hr_knowledge_base.txt    |
| Cisco      | cisco_hr_knowledge_base.txt        |
| Google     | google_hr_knowledge_base.txt       |
| Infosys    | infosys_hr_knowledge_base.txt      |
| Microsoft  | microsoft_hr_knowledge_base.txt    |
| TCS        | tcs_hr_knowledge_base.txt          |
