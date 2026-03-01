# Context-Aware RAG Engine — Project Plan

> From zero to deployed chatbot over PDF collections.

---

## Phase 0 · Project Setup & Tooling (Day 1)

| Task | Detail |
|------|--------|
| Init repo | `git init`, `.gitignore`, README |
| Python env | `python -m venv .venv` — Python 3.11+ |
| Project structure | See folder layout below |
| Dev tools | `ruff`, `mypy`, `pytest`, `pre-commit` |
| Config management | `.env` + `pydantic-settings` for secrets/keys |

### Folder Layout

```
CntextAware/
├── app/                    # FastAPI application
│   ├── api/                # Route handlers
│   │   ├── chat.py         # /chat endpoint (streaming + sync)
│   │   ├── upload.py       # /upload PDF endpoint
│   │   └── health.py       # /health
│   ├── core/               # App config, dependencies
│   │   ├── config.py       # Settings (pydantic-settings)
│   │   └── dependencies.py # FastAPI dependency injection
│   ├── models/             # Pydantic request/response schemas
│   ├── services/           # Business logic
│   │   ├── pdf_parser.py
│   │   ├── chunker.py
│   │   ├── embedder.py
│   │   ├── vector_store.py
│   │   ├── retriever.py
│   │   ├── reranker.py
│   │   └── llm_chain.py
│   └── main.py             # FastAPI app entrypoint
├── chatbot/                # Frontend (Streamlit / Chainlit)
│   └── app.py
├── scripts/                # One-off scripts (bulk ingest, eval)
├── tests/
├── data/                   # Local PDF storage (gitignored)
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
├── requirements.txt
├── pyproject.toml
├── .env.example
└── PROJECT_PLAN.md         # ← You are here
```

---

## Phase 1 · PDF Ingestion Pipeline (Days 2–4)

### 1.1 PDF Parsing

| Option | Pros | Use when |
|--------|------|----------|
| **PyMuPDF (fitz)** | Fast, layout-aware, table support | Default choice |
| **pdfplumber** | Better table extraction | Table-heavy PDFs |
| **Unstructured.io** | Handles images, OCR, mixed formats | Scanned / messy PDFs |

**Tasks:**
- [ ] Build `pdf_parser.py` — extract text + metadata (page number, headers, filename)
- [ ] Handle edge cases: scanned PDFs (OCR via `pytesseract`), multi-column layouts
- [ ] Store raw extracted text with source metadata

### 1.2 Chunking Strategy (Critical for accuracy)

**Approach — Hierarchical / Semantic Chunking:**

```
Level 1: Document → Sections  (by headings / structural cues)
Level 2: Sections → Paragraphs
Level 3: Paragraphs → Overlapping windows (chunk_size=512 tokens, overlap=64)
```

| Strategy | Implementation |
|----------|---------------|
| **Recursive Character Splitting** | `langchain.text_splitter.RecursiveCharacterTextSplitter` |
| **Semantic Chunking** | Split on sentence embeddings similarity drops |
| **Token-aware splitting** | Use `tiktoken` to respect model token limits |

**Tasks:**
- [ ] Build `chunker.py` with configurable `chunk_size`, `overlap`, `strategy`
- [ ] Attach metadata to every chunk: `{doc_id, page_num, chunk_index, source_file, section_heading}`
- [ ] Deduplicate near-identical chunks (MinHash / jaccard)

### 1.3 Embedding Generation

| Model | Dims | Speed | Quality |
|-------|------|-------|---------|
| **`text-embedding-3-small`** (OpenAI) | 1536 | Fast (API) | Great |
| **`all-MiniLM-L6-v2`** (local) | 384 | Very fast | Good |
| **`BAAI/bge-large-en-v1.5`** (local) | 1024 | Medium | Excellent |
| **`nomic-embed-text`** (Ollama) | 768 | Fast (local) | Very good |

**Tasks:**
- [ ] Build `embedder.py` — abstract interface, swap models easily
- [ ] Batch embed chunks (not one-by-one)
- [ ] Cache embeddings to avoid re-computation

---

## Phase 2 · Vector Store & Retrieval (Days 5–7)

### 2.1 Vector Store Setup

**Primary: FAISS** (fast, local, no infra)
**Alternative: ChromaDB** (persistent, metadata filtering, simpler API)

| Feature | FAISS | ChromaDB |
|---------|-------|----------|
| Speed | ★★★★★ | ★★★★ |
| Metadata filtering | Manual | Built-in |
| Persistence | Manual save/load | Built-in |
| Scalability | Millions of vectors | Medium scale |
| Setup complexity | Low | Low |

**Decision:** Start with **ChromaDB** for faster iteration (metadata filtering is huge for grounding). Switch to **FAISS** if latency becomes an issue at scale.

**Tasks:**
- [ ] Build `vector_store.py` — abstract interface over ChromaDB/FAISS
- [ ] Implement `add_documents()`, `search()`, `delete()`
- [ ] Persist index to disk; support incremental updates (add new PDFs without re-indexing all)

### 2.2 Retrieval Pipeline

```
User Query
    │
    ▼
┌─────────────────┐
│ Query Embedding  │
└────────┬────────┘
         ▼
┌─────────────────┐     ┌──────────────────┐
│ Vector Search   │────▶│ Top-K Candidates │ (k=20)
│ (ANN / cosine)  │     │ + metadata       │
└─────────────────┘     └────────┬─────────┘
                                 ▼
                    ┌────────────────────────┐
                    │ Re-Ranker              │ (Cross-encoder or Cohere Rerank)
                    │ Scores query×chunk     │
                    └────────────┬───────────┘
                                 ▼
                    ┌────────────────────────┐
                    │ Top-N Final Chunks     │ (n=5)
                    │ + source citations     │
                    └────────────────────────┘
```

**Tasks:**
- [ ] Build `retriever.py` — query embedding → vector search → top-K
- [ ] Build `reranker.py` — cross-encoder reranking (`cross-encoder/ms-marco-MiniLM-L-6-v2` or Cohere Rerank API)
- [ ] Implement **Hybrid Search**: combine dense (vector) + sparse (BM25 via `rank_bm25`) retrieval
- [ ] Add metadata filters (by document, date range, section)

---

## Phase 3 · LLM Integration & Grounding (Days 8–11)

### 3.1 LLM Chain with Source Grounding

**Goal:** Reduce hallucinations by ~70% via strict grounding.

**System Prompt Template:**
```
You are a precise research assistant. Answer the user's question using
ONLY the provided context. Follow these rules strictly:

1. If the context contains the answer, respond and cite sources as [Source: filename, page X].
2. If the context is insufficient, say "I don't have enough information to answer this."
3. NEVER fabricate information not present in the context.
4. Quote directly from sources when possible.

Context:
{retrieved_chunks_with_metadata}

Question: {user_query}
```

**Anti-hallucination techniques:**
| Technique | How |
|-----------|-----|
| **Strict system prompt** | "Only use provided context" |
| **Citation enforcement** | Require `[Source: ...]` in every claim |
| **Confidence scoring** | LLM self-rates confidence; flag low-confidence answers |
| **Chunk attribution** | Map each sentence in response back to a source chunk |
| **Temperature = 0** | Deterministic outputs |
| **Max-token guardrails** | Prevent runaway generation |

**Tasks:**
- [ ] Build `llm_chain.py` — prompt template + LLM call + response parsing
- [ ] Support multiple LLM backends: OpenAI GPT-4o, local Ollama (llama3, mistral)
- [ ] Implement streaming responses (SSE via FastAPI)
- [ ] Add citation extraction & validation (parse `[Source: ...]` from response)
- [ ] Implement fallback: if retrieval returns low-similarity chunks, say "not enough info"

### 3.2 Conversation Memory

- [ ] Add conversation history (last N turns) to prompt context
- [ ] Implement query reformulation: use LLM to rewrite follow-up queries as standalone questions
- [ ] Store chat sessions (SQLite or Redis)

---

## Phase 4 · FastAPI Backend (Days 12–14)

### 4.1 API Endpoints

```
POST   /api/upload          # Upload PDF(s), trigger ingestion pipeline
POST   /api/chat            # Send message, get RAG response
GET    /api/chat/stream     # SSE streaming variant
GET    /api/documents       # List indexed documents
DELETE /api/documents/{id}  # Remove document from index
GET    /api/health          # Health check
```

**Tasks:**
- [ ] Wire up all endpoints in FastAPI
- [ ] Add request validation (Pydantic models)
- [ ] Implement async PDF processing (background tasks or Celery)
- [ ] Add rate limiting (`slowapi`)
- [ ] CORS configuration for frontend
- [ ] Error handling middleware (structured JSON errors)
- [ ] Request logging with correlation IDs

### 4.2 Performance Targets

| Metric | Target |
|--------|--------|
| Query-to-first-token | < 1.5s |
| Full response | < 5s |
| PDF ingestion (100-page) | < 60s |
| Concurrent users | 50+ |

---

## Phase 5 · Chatbot Frontend (Days 15–17)

### Option A: Streamlit (Fastest to build)
```python
# chatbot/app.py
import streamlit as st
import requests

st.title("📄 Context-Aware RAG Chatbot")
# Chat UI with st.chat_message, st.chat_input
# File uploader with st.file_uploader
# Source citations displayed in expandable sections
```

### Option B: Chainlit (Better chat UX, built for RAG)
- Native streaming, file upload, source display
- Minimal code, production-ready chat UI

### Option C: React + Tailwind (Full control)
- Use `@shadcn/ui` chat components
- WebSocket or SSE for streaming
- More effort, most customizable

**Recommended: Chainlit** — best balance of speed and features for a RAG chatbot.

**Tasks:**
- [ ] Build chat interface with message history
- [ ] File upload UI (drag-and-drop PDFs)
- [ ] Display source citations with page numbers (clickable/expandable)
- [ ] Show processing status (indexing progress bar)
- [ ] Dark/light mode

---

## Phase 6 · Evaluation & Optimization (Days 18–20)

### 6.1 RAG Evaluation Metrics

| Metric | Tool | What it measures |
|--------|------|-----------------|
| **Faithfulness** | RAGAS | Does the answer stick to retrieved context? |
| **Answer Relevancy** | RAGAS | Is the answer relevant to the question? |
| **Context Precision** | RAGAS | Are retrieved chunks relevant? |
| **Context Recall** | RAGAS | Did we retrieve all needed info? |
| **Hallucination Rate** | Custom | % of claims not grounded in sources |

**Tasks:**
- [ ] Create evaluation dataset: 50+ question-answer-source triples from your PDFs
- [ ] Run RAGAS evaluation pipeline
- [ ] A/B test chunking strategies (size, overlap, semantic vs. fixed)
- [ ] A/B test retrieval configs (top-K, with/without reranker, hybrid vs. dense-only)
- [ ] Benchmark latency under load (`locust` or `k6`)

### 6.2 Optimization Levers

```
Low accuracy?
├── Chunking too large/small → tune chunk_size (256–1024 tokens)
├── Missing context → increase top-K, add hybrid search
├── Wrong chunks ranked high → add reranker
└── LLM ignoring context → strengthen system prompt, lower temperature

High latency?
├── Embedding slow → batch, cache, use smaller model
├── Vector search slow → use FAISS IVF index, reduce dimensions
├── LLM slow → use smaller model, enable streaming
└── Reranker slow → limit candidates, use lighter model
```

---

## Phase 7 · Deployment (Days 21–25)

### 7.1 Containerization

```dockerfile
# docker/Dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

```yaml
# docker/docker-compose.yml
services:
  api:
    build: .
    ports: ["8000:8000"]
    env_file: .env
    volumes: ["./data:/app/data"]
  chatbot:
    build:
      context: .
      dockerfile: docker/Dockerfile.chatbot
    ports: ["8501:8501"]
```

### 7.2 Deployment Options

| Platform | Cost | Complexity | Best for |
|----------|------|-----------|----------|
| **Railway / Render** | ~$7/mo | Low | Quick deploy, demos |
| **AWS EC2 + S3** | ~$20/mo | Medium | Production |
| **Azure Container Apps** | ~$15/mo | Medium | Enterprise |
| **GCP Cloud Run** | Pay-per-use | Medium | Variable traffic |
| **Self-hosted (VPS)** | $5–20/mo | Medium | Full control |

### 7.3 Production Checklist

- [ ] Dockerize API + Frontend
- [ ] Set up CI/CD (GitHub Actions → build → test → deploy)
- [ ] Environment variables via secrets manager
- [ ] Add monitoring: structured logging (`loguru`), health checks
- [ ] Set up error tracking (Sentry)
- [ ] Add authentication (API keys or OAuth)
- [ ] HTTPS via reverse proxy (Caddy/Nginx)
- [ ] Persistent vector store volume (don't lose index on redeploy)
- [ ] Auto-scaling config (if cloud)
- [ ] Backup strategy for vector index + chat history

---

## Tech Stack Summary

| Layer | Technology |
|-------|-----------|
| **Language** | Python 3.11+ |
| **API Framework** | FastAPI + Uvicorn |
| **PDF Parsing** | PyMuPDF + Unstructured (fallback) |
| **Chunking** | LangChain splitters + custom semantic |
| **Embeddings** | OpenAI `text-embedding-3-small` or `bge-large` |
| **Vector Store** | ChromaDB (dev) → FAISS (prod) |
| **Reranking** | `cross-encoder/ms-marco-MiniLM-L-6-v2` |
| **LLM** | OpenAI GPT-4o / Ollama (local) |
| **Frontend** | Chainlit or Streamlit |
| **Deployment** | Docker + Railway/AWS |
| **Eval** | RAGAS |

---

## Dependencies (`requirements.txt`)

```
# Core
fastapi>=0.115.0
uvicorn[standard]>=0.30.0
pydantic>=2.0
pydantic-settings>=2.0
python-multipart>=0.0.9

# PDF Processing
PyMuPDF>=1.24.0
pdfplumber>=0.11.0
pytesseract>=0.3.10

# RAG Pipeline
langchain>=0.3.0
langchain-community>=0.3.0
langchain-openai>=0.2.0
tiktoken>=0.7.0
chromadb>=0.5.0
faiss-cpu>=1.8.0
rank-bm25>=0.2.2
sentence-transformers>=3.0.0

# LLM
openai>=1.50.0

# Frontend
chainlit>=1.2.0

# Eval
ragas>=0.2.0

# Dev
pytest>=8.0
ruff>=0.6.0
loguru>=0.7.0
python-dotenv>=1.0.0
```

---

## Milestone Timeline

| Week | Phase | Deliverable |
|------|-------|-------------|
| **1** | 0 + 1 | Project scaffolded, PDFs parsed & chunked, embeddings stored |
| **2** | 2 + 3 | Vector search working, LLM generates grounded answers with citations |
| **3** | 4 + 5 | FastAPI endpoints live, chatbot UI functional end-to-end |
| **4** | 6 + 7 | Evaluated, optimized, Dockerized, deployed |

---

## Next Step

Run this to scaffold the project:

```bash
mkdir -p app/api app/core app/models app/services chatbot scripts tests data docker
touch app/__init__.py app/main.py app/api/__init__.py app/core/__init__.py
touch app/models/__init__.py app/services/__init__.py
```

Then start with **Phase 1.1** — PDF parsing.
