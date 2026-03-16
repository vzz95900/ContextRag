"""Debug script for the chat retrieval and generation pipeline."""
import sys, traceback
from pathlib import Path

ROOT = Path(r"d:\Pjts\CntextAware")
sys.path.insert(0, str(ROOT))
LOG = ROOT / "data" / "chat_debug.log"

lines = []
def log(msg):
    print(msg)
    lines.append(str(msg))

try:
    from app.core.config import settings
    log(f"Testing chat pipeline...")
    log(f"LLM Provider: {settings.llm_provider}")
    log(f"LLM Model: {settings.llm_model}")
except Exception as e:
    log(f"CONFIG ERROR: {e}")
    sys.exit(1)

# Step 1: Embed query
try:
    log("\n[Step 1] Embedding query...")
    from app.services.embedder import embed_query
    query = "What is this document about?"
    query_emb = embed_query(query)
    log(f"  OK (dim={len(query_emb)})")
except Exception as e:
    log(f"EMBED QUERY ERROR:\n{traceback.format_exc()}")
    LOG.write_text("\n".join(lines), encoding="utf-8")
    sys.exit(1)

# Step 2: Retrieve chunks
try:
    log("\n[Step 2] Retrieving chunks from ChromaDB...")
    from app.services.retriever import retrieve
    results = retrieve(query, top_k=5)
    log(f"  OK (found {len(results)} chunks)")
    if results:
        log(f"  Top result score: {results[0].get('score', 'N/A')}")
except Exception as e:
    log(f"RETRIEVE ERROR:\n{traceback.format_exc()}")
    LOG.write_text("\n".join(lines), encoding="utf-8")
    sys.exit(1)

# Step 3: Call Gemini LLM
try:
    log("\n[Step 3] Calling Gemini to generate answer...")
    from app.services.llm_chain import generate_answer
    if not results:
        log("  WARNING: No chunks found in DB. Did ingest succeed?")
        # Create fake chunk just to test LLM
        results = [{"text": "This is a test document.", "metadata": {"filename": "test.pdf", "page_num": 1}}]
    
    answer = generate_answer(query, results)
    log(f"  OK! Answer snippet: {answer[:100]}...")
except Exception as e:
    log(f"LLM ERROR:\n{traceback.format_exc()}")
    LOG.write_text("\n".join(lines), encoding="utf-8")
    sys.exit(1)

log("\nALL_STEPS_OK")
LOG.write_text("\n".join(lines), encoding="utf-8")
