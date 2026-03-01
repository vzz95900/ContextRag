"""Debug the upload pipeline step by step."""
import sys, os
os.chdir(r"D:\Pjts\CntextAware")
sys.path.insert(0, ".")

log = open("debug_output.txt", "w")
def p(msg):
    log.write(msg + "\n")
    log.flush()

try:
    from app.core.config import settings
    p(f"embedding_model: {settings.embedding_model}")
    p(f"gemini_api_key set: {bool(settings.gemini_api_key)}")

    from app.services.pdf_parser import parse_pdf
    pdf_path = r"data/uploads/Resum.pdf"
    p(f"Parsing: {pdf_path}")
    pages = parse_pdf(pdf_path)
    p(f"Pages extracted: {len(pages)}")
    for pg in pages[:2]:
        p(f"  Page {pg.page_num}: {len(pg.text)} chars")

    from app.services.chunker import chunk_pages
    chunks = chunk_pages(pages)
    p(f"Chunks created: {len(chunks)}")

    texts = [c.text for c in chunks]
    p(f"Texts to embed: {len(texts)}")
    
    if texts:
        from app.services.embedder import embed_texts
        embeddings = embed_texts(texts[:2])
        p(f"Embeddings returned: {len(embeddings)}")
        if embeddings:
            p(f"Embedding dim: {len(embeddings[0])}")
        else:
            p("ERROR: embeddings is empty!")
    else:
        p("ERROR: No texts!")

    p("DONE")
except Exception as e:
    import traceback
    p(f"ERROR: {type(e).__name__}: {e}")
    p(traceback.format_exc())
finally:
    log.close()
