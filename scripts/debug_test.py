import sys, traceback
from pathlib import Path

ROOT = Path(r"d:\Pjts\CntextAware")
sys.path.insert(0, str(ROOT))
LOG = ROOT / "data" / "ingest_debug.log"

lines = []

def log(msg):
    lines.append(str(msg))

try:
    from app.core.config import settings
    log(f"llm_provider={settings.llm_provider}")
    log(f"embedding_provider={settings.embedding_provider}")
    log(f"embedding_model={settings.embedding_model}")
    log(f"gemini_key_set={'yes' if settings.gemini_api_key else 'no'}")
except Exception:
    log(f"CONFIG_ERROR:\n{traceback.format_exc()}")
    LOG.write_text("\n".join(lines), encoding="utf-8")
    sys.exit(1)

pdf = ROOT / "data" / "pdfs" / "Greedy.pdf"
log(f"test_pdf={pdf.name} exists={pdf.exists()}")

try:
    from app.services.pdf_parser import parse_pdf
    pages = parse_pdf(str(pdf))
    log(f"parse_ok pages={len(pages)}")
except Exception:
    log(f"PARSE_ERROR:\n{traceback.format_exc()}")
    LOG.write_text("\n".join(lines), encoding="utf-8")
    sys.exit(1)

try:
    from app.services.chunker import chunk_pages
    chunks = chunk_pages(pages)
    log(f"chunk_ok count={len(chunks)}")
except Exception:
    log(f"CHUNK_ERROR:\n{traceback.format_exc()}")
    LOG.write_text("\n".join(lines), encoding="utf-8")
    sys.exit(1)

try:
    from app.services.embedder import embed_texts
    emb = embed_texts([chunks[0].text])
    log(f"embed_ok dim={len(emb[0])}")
except Exception:
    log(f"EMBED_ERROR:\n{traceback.format_exc()}")
    LOG.write_text("\n".join(lines), encoding="utf-8")
    sys.exit(1)

log("ALL_STEPS_OK")
LOG.write_text("\n".join(lines), encoding="utf-8")
