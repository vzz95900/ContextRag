# Context-Aware RAG Engine

⚡ A high-performance, next-generation Semantic Search & QA engine utilizing **Multi-Objective Optimization** to eliminate redundancy and hallucination in RAG pipelines. 

Built with a premium Glassmorphism Single-Page Application (SPA) frontend.

## Features

- **Multi-Objective Optimizer** — Replaces standard "Top-K" retrieval with a greedy selection algorithm balancing:
  - 📐 **Relevance**: Semantic match to the user query.
  - 🌐 **Coverage**: Diversity of information (penalizing near-duplicate chunks).
  - 🤝 **Support**: Cross-document corroboration to naturally resist hallucinations.
- **Glassmorphism SPA Frontend** — Premium, fully responsive Single Page Application natively served by FastAPI. Zero frontend framework dependencies; pure HTML, TailwindCSS, and Vanilla JS.
- **Advanced Retrieval Pipeline** — Dense vector search from ChromaDB with a larger candidate pool (N=30) -> Multi-Objective optimization -> Optional BM25 Hybrid Fusion -> CrossEncoder Reranking.
- **Grounded Answers** — LLM responses cite exact sources; explicitly constrained generation using Gemini.
- **PDF Ingestion** — Parse, chunk, and embed PDFs automatically.

## Quick Start

```bash
# 1. Create a virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
# source .venv/bin/activate # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Gemini API key in .env
echo "GEMINI_API_KEY=your-key-here" > .env
# Optional: Tune optimizer weights in .env
# ENABLE_OPTIMIZER=true
# OPTIMIZER_ALPHA=0.5
# OPTIMIZER_BETA=0.3
# OPTIMIZER_GAMMA=0.2

# 4. Start the Application
uvicorn app.main:app --reload

# 5. Open your browser
# Visit: http://localhost:8000
```



## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload & index a PDF |
| `POST` | `/api/chat` | Ask a question (Returns ChatResponse with used `retrieval_mode`) |
| `GET` | `/api/documents` | List indexed documents |
| `DELETE` | `/api/documents/{id}` | Remove a document |
| `GET` | `/health` | Health check |

## Architecture

```text
PDF → Parse → Chunk → Embed → ChromaDB
                                  ↓
User Query → Embed → Vector Search (N=30 pool) 
                          ↓
      ✨ Multi-Objective Optimizer (Greedy Loop) ✨
                    [Rel + Cov + Sup]
                          ↓
              BM25 Hybrid Fusion (Optional)
                          ↓
                 CrossEncoder Rerank 
                          ↓ 
              Gemini LLM → Answer + Citations
```
