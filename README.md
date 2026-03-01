# Context-Aware RAG Engine

Semantic search & QA chatbot over PDF collections with source-grounded answers.

## Features

- **PDF Ingestion** — Parse, chunk, and embed PDFs automatically
- **Hybrid Retrieval** — Dense vector search + BM25 with cross-encoder reranking
- **Grounded Answers** — LLM responses cite exact sources; refuses to hallucinate
- **Gemini-Powered** — Uses Google Gemini for embeddings and generation
- **Low-Latency API** — FastAPI backend with ChromaDB vector store

## Quick Start

```bash
# 1. Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Gemini API key in .env
#    GEMINI_API_KEY=your-key-here

# 4. Start the API server
uvicorn app.main:app --reload

# 5. Start the chatbot (new terminal)
streamlit run chatbot/app.py
```

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `POST` | `/api/upload` | Upload & index a PDF |
| `POST` | `/api/chat` | Ask a question |
| `GET` | `/api/documents` | List indexed documents |
| `DELETE` | `/api/documents/{id}` | Remove a document |
| `GET` | `/health` | Health check |

## Architecture

```
PDF → Parse → Chunk → Embed → ChromaDB
                                  ↓
User Query → Embed → Vector Search → BM25 Hybrid → Rerank → LLM → Answer + Citations
```
