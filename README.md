---
title: Context-Aware RAG Engine
emoji: 🧠
colorFrom: purple
colorTo: blue
sdk: docker
pinned: false
---

<div align="center">

# 🧠 Context-Aware RAG Engine

### Next-Generation Semantic Search & Question Answering via Submodular Multi-Objective Context Optimization

[![Python Version](https://img.shields.io/badge/python-3.11%2B-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.115%2B-009688.svg?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-HNSW_Vector_Store-orange.svg?style=for-the-badge)](https://www.trychroma.com/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![Hugging Face](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-yellow.svg?style=for-the-badge)](https://huggingface.co/spaces/Vizz17/context-aware-rag)

<p align="center">
  <b>Eliminating chunk redundancy and parametric hallucinations in Retrieval-Augmented Generation through submodular optimization, dynamic neural weight prediction, hybrid BM25 + dense search, cross-encoder reranking, and multi-tenant session isolation.</b>
</p>

[Key Features](#-key-features) •
[Algorithmic Paradigm Shift](#-algorithmic-paradigm-shift) •
[System Architecture](#-system-architecture) •
[Empirical Benchmarks](#-empirical-benchmarks) •
[Quick Start](#-quick-start) •
[Configuration](#-configuration-reference) •
[API Reference](#-api-reference)

</div>

---

## 🌟 Overview

Conventional Retrieval-Augmented Generation (RAG) pipelines rely on **naïve Top-K retrieval**, which ranks candidate document chunks independently using scalar cosine similarity. This causes critical failure modes in production:

1. **Semantic Redundancy:** Top-ranked results are often near-duplicate passages containing repetitive sentences, consuming scarce LLM context window tokens without introducing new facts.
2. **Missing Corroboration:** Highly relevant individual points might lack mutual support across documents, increasing the risk of hallucination when synthesizing nuanced answers.
3. **No Diversity Awareness:** Marginal similarity ranking cannot evaluate how a new chunk complements the *already-selected* set.

**Context-Aware RAG** replaces naive Top-K retrieval with a **Multi-Objective Submodular Greedy Optimizer** and an **Adaptive Neural Weight Predictor**. The retrieval engine dynamically balances **Semantic Relevance**, **Information Coverage (Diversity)**, and **Cross-Document Support (Agreement)** to curate an optimal, hallucination-resistant evidence set.

---

## ✨ Key Features

- 🧬 **Multi-Objective Submodular Context Optimizer**:
  - Dynamically evaluates chunks against the already-selected set $S$ at every iteration.
  - Balances **Relevance** ($\alpha$), **Coverage / Marginal Diversity** ($\beta$), and **Cross-Document Support** ($\gamma$).
  - Eliminates up to **77.5%** of near-duplicate chunks while boosting grounding confidence by **+24.2%**.

- 🧠 **Adaptive Neural Weight Predictor**:
  - Lightweight PyTorch MLP (`weight_predictor.py`) that inspects the query embedding representation and dynamically infers query-dependent weights $(\alpha, \beta, \gamma)$.
  - Prior-informed initialization ensures stable balanced defaults ($\alpha=0.5, \beta=0.3, \gamma=0.2$) with seamless query adaptation.

- 🔍 **Hybrid Multi-Stage Retrieval Pipeline**:
  - **Dense ANN Retrieval**: High-recall candidate search from ChromaDB HNSW index ($N=30$ pool).
  - **Sparse BM25 Fusion**: Token-level lexical matching combined via Reciprocal Rank Fusion (RRF).
  - **Greedy Submodular Selection**: Optimal context subset reduction.
  - **Cross-Encoder Reranking**: High-precision rescoring with `cross-encoder/ms-marco-MiniLM-L-6-v2`.

- 🤖 **Universal Multi-Provider LLM & Embedding Support**:
  - **LLM Backends**: Groq (ultra-low latency LLaMA 3.3 70B), Google Gemini (Gemini 2.0 Flash / 1.5 Pro via `google-genai`), OpenAI (GPT-4o / GPT-4o-mini), Hugging Face Serverless Inference, and local Ollama (`llama3.2`).
  - **Embedding Providers**: Google Gemini (`gemini-embedding-001`), OpenAI (`text-embedding-3-small`), and local HuggingFace `sentence-transformers` (`all-MiniLM-L6-v2`, `BAAI/bge-large-en-v1.5`).

- 🔒 **Multi-Tenant User Isolation & Guest Sessions**:
  - SQLite user management with salted PBKDF2-HMAC password hashing.
  - Fast Guest mode with zero sign-up friction.
  - Strict data isolation: documents, embeddings, and chat histories are isolated by `user_id` inside ChromaDB.
  - Automatic scheduled TTL cleanup purging expired guest session vectors every hour.

- 💬 **Conversational Memory & Document-Scoped Chats**:
  - History tracking per document or workspace collection.
  - Contextual query expansion: automatically merges conversational context with short follow-up inquiries.
  - Source attribution with exact filename, page number, chunk index, and similarity scores.

- 🎨 **Glassmorphism Single-Page Application (SPA)**:
  - Zero heavy frontend frameworks; pure HTML5, modern Tailwind CSS, and Vanilla JavaScript.
  - Cyberpunk dark theme with interactive HTML5 particle canvas background.
  - Dedicated interactive architecture dashboard (`/static/roadmap.html`) and live benchmark visualizer (`/static/change.html`) powered by Plotly.js.

---

## 📐 Algorithmic Paradigm Shift

### The Tri-Metric Objective Function

Rather than scoring items independently, the optimizer iteratively selects the chunk $d$ that maximizes the joint marginal objective:

$$\text{Score}(d) = \alpha \cdot \text{Rel}(q, d) + \beta \cdot \text{Cov}(d \mid S) + \gamma \cdot \text{Sup}(d, S)$$

$$\text{subject to } \alpha + \beta + \gamma = 1.0$$

```
                         Candidate Chunk d
                               │
       ┌───────────────────────┼───────────────────────┐
       ▼                       ▼                       ▼
  Relevance              Coverage (Diversity)     Support (Agreement)
  Rel(q, d) = cos(q, d)   Cov(d|S) = 1 - max cos  Sup(d,S) = mean cos
                                        s∈S                    s∈S
       │                       │                       │
       ▼ (× α)                 ▼ (× β)                 ▼ (× γ)
       └───────────────────────┬───────────────────────┘
                               ▼
                        Combined Score(d)
```

| Component | Formulation | Range | Purpose |
| :--- | :--- | :---: | :--- |
| **Relevance** $\text{Rel}(q, d)$ | $\cos(q, d)$ | $[-1, 1]$ | Quantifies direct semantic alignment between query vector $q$ and chunk vector $d$. |
| **Coverage** $\text{Cov}(d \mid S)$ | $1 - \max_{s \in S} \cos(d, s)$ | $[0, 1]$ | Penalizes semantic redundancy. Measures novel information introduced relative to selected set $S$. |
| **Support** $\text{Sup}(d, S)$ | $\frac{1}{\|S\|} \sum_{s \in S} \cos(d, s)$ | $[0, 1]$ | Rewards factual consensus. Prioritizes passages that corroborate previously selected evidence. |

### Greedy Submodular Selection Loop

1. **Candidate Retrieval**: Retrieve top candidate pool ($N = 30$) from ChromaDB using dense vector similarity.
2. **Initialize Selected Set**: Start with an empty selection set $S = \emptyset$.
3. **Iterative Evaluation**: For each remaining candidate $d \notin S$, recalculate $\text{Cov}(d \mid S)$ and $\text{Sup}(d, S)$ against current set $S$.
4. **Greedy Maximization**: Select $d^* = \arg\max_{d \notin S} \text{Score}(d)$ and update $S \leftarrow S \cup \{d^*\}$.
5. **Termination**: Repeat until $|S| = k$ (where $k = \text{retrieval\_top\_k}$). Pass $S$ directly to cross-encoder reranker and LLM.

---

## 🏗️ System Architecture

```text
┌────────────────────────────────────────────────────────────────────────┐
│                        Corpus Ingestion Pipeline                       │
│                                                                        │
│   PDF File ──► PyMuPDF Parser ──► Recursive Chunker (512 tokens / 64)  │
│                                           │                            │
│                                           ▼                            │
│                         Embedding Engine (Gemini / Local / OpenAI)     │
│                                           │                            │
│                                           ▼                            │
│                         ChromaDB Vector Store (HNSW Cosine)            │
│                         - user_id namespace partition                  │
│                         - chat_histories collection                    │
└────────────────────────────────────────────────────────────────────────┘

┌────────────────────────────────────────────────────────────────────────┐
│                    Grounded Retrieval & Inference Flow                 │
│                                                                        │
│   User Query ──► Adaptive Weight Predictor (Neural MLP: α, β, γ)       │
│        │                                                               │
│        ├──► Dense Vector Search (ChromaDB, N=30 candidate pool)        │
│        │                                                               │
│        ├──► (Optional) BM25 Sparse Search ──► Reciprocal Rank Fusion   │
│        │                                                               │
│        └──► ✨ Multi-Objective Greedy Optimizer [Rel + Cov + Sup] ✨   │
│                                           │                            │
│                                           ▼                            │
│             Cross-Encoder Reranker (ms-marco-MiniLM-L-6-v2)            │
│                                           │                            │
│                                           ▼                            │
│             Strict Grounding Context Prompt Formatter                  │
│                                           │                            │
│                                           ▼                            │
│             LLM Synthesizer (Groq / Gemini / OpenAI / Ollama)          │
│                                           │                            │
│                                           ▼                            │
│             Answer with Exact Source Citations + Session Log           │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 📊 Empirical Benchmarks

Evaluated over real-world document collections (283 indexed chunks across 8 multi-faceted benchmark queries) comparing standard Top-K vs Multi-Objective Optimization:

```
Redundancy (Lower is better)
  Top-K:        ████████████████████ 45.2%
  Optimized:    ████ 10.1%  (-77.5% redundancy reduction)

Grounding / Faithfulness (Higher is better)
  Top-K:        ████████████ 57.0%
  Optimized:    ████████████████ 81.2%  (+24.2% factual gain)

Evidence Diversity (Higher is better)
  Top-K:        ████████ 19.9%
  Optimized:    █████████ 22.5%  (+2.6% diversity expansion)
```

| Quality Metric | Standard Top-K Baseline | Multi-Objective Optimized | Absolute Delta |
| :--- | :---: | :---: | :---: |
| **Evidence Diversity** | 19.9% | **22.5%** | **+2.6%** |
| **Near-Duplicate Redundancy** | 45.2% | **10.1%** | **-77.5%** |
| **Grounded Faithfulness** | 57.0% | **81.2%** | **+24.2%** |
| **Answer Reliability & Focus** | 66.2% | **63.8%** | Controlled trade-off |
| **Key Concept Coverage** | 2 / 5 points | **5 / 5 points** | **+3 points** |

---

## 🚀 Quick Start

### Prerequisites

- Python 3.11+
- Git
- (Optional) Docker & Docker Compose
- API Key for your preferred LLM provider (Groq, Google Gemini, or OpenAI) or a local Ollama instance

### 1. Clone & Set Up Virtual Environment

```bash
# Clone the repository
git clone https://github.com/vzz95900/ContextRag.git
cd ContextRag

# Create virtual environment
python -m venv .venv

# Activate environment:
# Windows (PowerShell):
.venv\Scripts\Activate.ps1
# Windows (CMD):
.venv\Scripts\activate.bat
# Linux / macOS:
source .venv/bin/activate
```

### 2. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Configure Environment Variables

Copy `.env.example` to `.env` and configure your API keys and model preferences:

```bash
# On Linux/macOS
cp .env.example .env

# On Windows (PowerShell)
Copy-Item .env.example .env
```

Edit `.env` with your preferred credentials:

```dotenv
# Select your LLM Provider: groq | gemini | openai | ollama | huggingface
LLM_PROVIDER=groq
LLM_MODEL=llama-3.3-70b-versatile
GROQ_API_KEY=gsk_your_groq_api_key_here

# Embeddings: gemini | openai | local
EMBEDDING_PROVIDER=local
EMBEDDING_MODEL=all-MiniLM-L6-v2

# Optimizer & Reranking Settings
ENABLE_OPTIMIZER=true
ENABLE_ADAPTIVE_WEIGHTS=true
ENABLE_RERANKER=true
ENABLE_HYBRID_SEARCH=true
```

### 4. Run the Application

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

Open your browser at **[http://localhost:8000](http://localhost:8000)**.

- **Main Chat & Document Gateway**: `http://localhost:8000/`
- **Interactive Roadmap & Architecture**: `http://localhost:8000/static/roadmap.html`
- **Algorithmic Shift & Plotly Benchmarks**: `http://localhost:8000/static/change.html`
- **Interactive Swagger API Docs**: `http://localhost:8000/docs`

---

## 🐳 Docker Deployment

### Using Docker Compose (Recommended)

```bash
docker compose -f docker/docker-compose.yml up --build -d
```

### Using Standalone Docker (Hugging Face Spaces Compatible)

The repository includes a root `Dockerfile` pre-configured to expose port `7860`, matching Hugging Face Spaces requirements:

```bash
# Build Docker image
docker build -t context-aware-rag .

# Run container on port 7860
docker run -p 7860:7860 --env-file .env -v $(pwd)/data:/app/data context-aware-rag
```

Access the app at `http://localhost:7860`.

---

## 📚 Bulk PDF Ingestion

You can ingest an entire directory of PDF files in batch using the included CLI ingestion script:

```bash
# 1. Place your target PDFs into data/pdfs/
mkdir -p data/pdfs
# Copy your files into data/pdfs/

# 2. Run the ingestion runner
python scripts/ingest_pdfs.py
```

The script skips duplicate documents, chunks pages with contextual overlap, computes vector embeddings, and registers chunks into ChromaDB.

---

## ⚙️ Configuration Reference

All settings are managed via Pydantic BaseSettings in `app/core/config.py` and overridden through environment variables:

| Variable | Default Value | Description |
| :--- | :--- | :--- |
| **`LLM_PROVIDER`** | `groq` | Primary LLM: `groq`, `gemini`, `openai`, `ollama`, `huggingface`. |
| **`LLM_MODEL`** | `llama-3.3-70b-versatile` | LLM identifier for inference. |
| **`LLM_TEMPERATURE`** | `0.0` | Sampling temperature (`0.0` for maximum factual determinism). |
| **`LLM_MAX_TOKENS`** | `1024` | Maximum output generation token budget. |
| **`EMBEDDING_PROVIDER`** | `gemini` | Embeddings backend: `gemini`, `openai`, `local`. |
| **`EMBEDDING_MODEL`** | `gemini-embedding-001` | Model name (e.g. `all-MiniLM-L6-v2`, `text-embedding-3-small`). |
| **`VECTOR_STORE_PROVIDER`**| `chromadb` | Vector engine: `chromadb` or `faiss`. |
| **`CHROMA_PERSIST_DIR`** | `./data/chromadb` | Local persistent directory for ChromaDB embeddings and chat state. |
| **`CHUNK_SIZE`** | `512` | Token length per document passage chunk. |
| **`CHUNK_OVERLAP`** | `64` | Token overlap between consecutive chunks for boundary continuity. |
| **`RETRIEVAL_TOP_K`** | `10` | Final document count selected for the LLM context prompt. |
| **`RERANK_TOP_N`** | `3` | Final top chunks selected after cross-encoder scoring. |
| **`ENABLE_OPTIMIZER`** | `true` | Toggle multi-objective submodular greedy selection vs top-k. |
| **`ENABLE_ADAPTIVE_WEIGHTS`**| `true` | Dynamically predict $(\alpha, \beta, \gamma)$ using query neural MLP. |
| **`OPTIMIZER_CANDIDATE_N`**| `30` | Initial high-recall pool size pulled before multi-objective filtering. |
| **`OPTIMIZER_ALPHA`** | `0.5` | Fallback weight for Semantic Relevance. |
| **`OPTIMIZER_BETA`** | `0.3` | Fallback weight for Coverage / Diversity. |
| **`OPTIMIZER_GAMMA`** | `0.2` | Fallback weight for Cross-Document Support. |
| **`ENABLE_HYBRID_SEARCH`** | `true` | Combine dense cosine vectors with BM25 sparse keyword scores via RRF. |
| **`ENABLE_RERANKER`** | `true` | Enable Cross-Encoder rescoring (`ms-marco-MiniLM-L-6-v2`). |
| **`ADMIN_SECRET`** | `""` | Secret passkey for administrative user oversight endpoints. |

---

## 📡 API Reference

All protected endpoints accept the HTTP session header:
`Authorization: Bearer <session_token>`

### 🔐 Authentication

| Method | Endpoint | Description | Request Body / Query |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/auth/register` | Register new user | `{"username": "...", "password": "..."}` |
| `POST` | `/api/auth/login` | Authenticate user & get bearer token | `{"username": "...", "password": "..."}` |
| `POST` | `/api/auth/guest` | Instant guest session (cleared on logout / 4hr TTL) | *None* |
| `POST` | `/api/auth/logout` | Revoke session & clean up guest vector data | *Bearer Header* |
| `GET` | `/api/admin/users` | List registered accounts & active sessions | `?secret=<ADMIN_SECRET>` |

### 📄 Document Ingestion & Management

| Method | Endpoint | Description | Request / Response |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/upload` | Upload and index PDF document | Multipart `file=@document.pdf` |
| `GET` | `/api/documents` | List indexed documents for current user | Returns `DocumentListResponse` |
| `DELETE`| `/api/documents/{doc_id}` | Remove document and vector index | Path parameter `doc_id` |

### 💬 Grounded Question Answering & Chat

| Method | Endpoint | Description | Request Body |
| :--- | :--- | :--- | :--- |
| `POST` | `/api/chat` | Query RAG engine with conversational history | `{"query": "...", "session_id": "...", "filters": {"doc_id": "..."}, "history": [...]}` |
| `GET` | `/api/chats` | Retrieve conversation sessions | Optional query `?doc_id=<doc_id>` |
| `GET` | `/api/chats/{session_id}`| Fetch complete message thread | Path parameter `session_id` |
| `DELETE`| `/api/chats/{session_id}`| Delete conversation session | Path parameter `session_id` |

### 💓 System Health

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `GET` | `/health` | System status, vector store engine, and active LLM backend. |

---

## 📁 Repository Structure

```text
ContextRag/
├── .env.example                 # Template for environment variables
├── Dockerfile                   # Production Dockerfile (Hugging Face Spaces, Port 7860)
├── README.md                    # Core documentation & project specification
├── pyproject.toml               # Python project configuration & linter rules
├── requirements.txt             # Direct Python dependencies
├── app/
│   ├── main.py                  # FastAPI entry point, lifecycle hooks & static routes
│   ├── api/
│   │   ├── auth.py              # User registration, guest login, session endpoints
│   │   ├── chat.py              # RAG inference, query expansion & history management
│   │   ├── health.py            # Health check & status endpoint
│   │   └── upload.py            # PDF upload, extraction & vector registration
│   ├── core/
│   │   ├── config.py            # Centralized settings via Pydantic BaseSettings
│   │   └── dependencies.py      # Common FastAPI dependencies
│   ├── models/
│   │   └── schemas.py           # Pydantic schemas for requests, responses & citations
│   └── services/
│       ├── auth.py              # SQLite user DB, salted PBKDF2 hashing, TTL cleanup
│       ├── chunker.py           # Recursive token-based text chunker
│       ├── embedder.py          # Unified embeddings interface (Gemini, Local, OpenAI)
│       ├── llm_chain.py         # Multi-provider LLM caller with grounding system prompt
│       ├── optimizer.py         # Submodular Greedy Optimizer (Rel, Cov, Sup)
│       ├── pdf_parser.py        # Optical text extraction via PyMuPDF
│       ├── reranker.py          # Cross-Encoder candidate rescoring model
│       ├── retriever.py         # Dense vector search + BM25 Reciprocal Rank Fusion
│       ├── vector_store.py      # ChromaDB multi-tenant wrapper with cosine index
│       └── weight_predictor.py  # Adaptive Neural MLP for query-specific weights
├── data/
│   ├── chromadb/                # Persistent vector database store
│   ├── pdfs/                    # Target directory for bulk ingestion
│   ├── uploads/                 # Upload directory for web UI PDF files
│   └── users.db                 # SQLite user & session database
├── docker/
│   ├── Dockerfile               # Backend Dockerfile for multi-container setup
│   ├── Dockerfile.chatbot       # Frontend service container
│   └── docker-compose.yml       # Docker Compose multi-service definition
├── frontend/
│   ├── app.js                   # Client SPA logic, API client & session manager
│   ├── change.html              # Algorithmic Paradigm Shift & Plotly benchmark page
│   ├── index.html               # Main Glassmorphism UI & document gateway
│   ├── roadmap.html             # System architecture & pipeline deep-dive
│   └── style.css                # Cyberpunk glassmorphism styling & canvas effects
└── scripts/
    └── ingest_pdfs.py           # CLI bulk PDF ingestion utility
```

---

## 🛠️ Testing & Quality Assurance

```bash
# Run test suite
pytest -v

# Run code linting & formatting checks
ruff check app/
```

---

## 🤝 Contributing

Contributions, feedback, and architectural discussions are welcome!
1. Fork the Project
2. Create your Feature Branch (`git checkout -b feature/AmazingFeature`)
3. Commit your Changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the Branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for details.
