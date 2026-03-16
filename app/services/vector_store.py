"""Vector store abstraction — ChromaDB (default) with FAISS fallback."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from app.core.config import settings
from app.services.chunker import Chunk


# ── ChromaDB Backend ────────────────────────────────────────


class ChromaStore:
    """Persistent ChromaDB vector store."""

    def __init__(self, persist_dir: str | None = None, collection_name: str = "rag_docs"):
        import chromadb
        import shutil

        persist_dir = persist_dir or settings.chroma_persist_dir
        Path(persist_dir).mkdir(parents=True, exist_ok=True)

        try:
            self._client = chromadb.PersistentClient(path=persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._chat_collection = self._client.get_or_create_collection(
                name="chat_histories",
            )
        except (ValueError, AttributeError) as e:
            # Corrupted or incompatible DB — wipe and recreate
            import logging
            logging.getLogger(__name__).warning(
                "ChromaDB init failed (%s). Resetting database...", e
            )
            shutil.rmtree(persist_dir, ignore_errors=True)
            Path(persist_dir).mkdir(parents=True, exist_ok=True)
            self._client = chromadb.PersistentClient(path=persist_dir)
            self._collection = self._client.get_or_create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"},
            )
            self._chat_collection = self._client.get_or_create_collection(
                name="chat_histories",
            )

    @property
    def count(self) -> int:
        return self._collection.count()

    def add_chunks(self, chunks: List[Chunk], embeddings: List[List[float]]) -> None:
        """Add chunks with pre-computed embeddings."""
        self._collection.upsert(
            ids=[c.chunk_id for c in chunks],
            embeddings=embeddings,
            documents=[c.text for c in chunks],
            metadatas=[
                {
                    "doc_id": c.doc_id,
                    "filename": c.filename,
                    "page_num": c.page_num,
                    "chunk_index": c.chunk_index,
                    "token_count": c.token_count,
                }
                for c in chunks
            ],
        )

    def search(
        self,
        query_embedding: List[float],
        top_k: int = 20,
        where: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return top-k most similar chunks."""
        kwargs: Dict[str, Any] = {
            "query_embeddings": [query_embedding],
            "n_results": top_k,
            "include": ["documents", "metadatas", "distances"],
        }
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)

        hits: List[Dict[str, Any]] = []
        for i in range(len(results["ids"][0])):
            hits.append(
                {
                    "chunk_id": results["ids"][0][i],
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i],
                    "score": 1 - results["distances"][0][i],  # cosine → similarity
                }
            )
        return hits

    def delete_document(self, doc_id: str) -> int:
        """Delete all chunks belonging to a document. Returns count deleted."""
        existing = self._collection.get(where={"doc_id": doc_id})
        count = len(existing["ids"])
        if count:
            self._collection.delete(ids=existing["ids"])
        return count

    def list_documents(self) -> List[Dict[str, Any]]:
        """Return unique documents in the store."""
        all_meta = self._collection.get(include=["metadatas"])
        docs: Dict[str, Dict[str, Any]] = {}
        for meta in all_meta["metadatas"]:
            did = meta["doc_id"]
            if did not in docs:
                docs[did] = {
                    "doc_id": did,
                    "filename": meta["filename"],
                    "chunk_count": 0,
                    "pages": set(),
                }
            docs[did]["chunk_count"] += 1
            docs[did]["pages"].add(meta["page_num"])
        # Convert sets for serialization
        for d in docs.values():
            d["page_count"] = len(d.pop("pages"))
        return list(docs.values())

    # ── Chat Histories ────────────────────────────────────────

    def save_chat(self, session_id: str, title: str, messages: list) -> None:
        import json
        from datetime import datetime
        self._chat_collection.upsert(
            ids=[session_id],
            embeddings=[[0.0] * 384],  # Dummy embedding to bypass default model
            documents=[json.dumps(messages)],
            metadatas=[{
                "title": title,
                "updated_at": datetime.utcnow().isoformat()
            }]
        )

    def get_chat(self, session_id: str) -> dict | None:
        import json
        results = self._chat_collection.get(ids=[session_id])
        if not results["ids"]:
            return None
        return {
            "session_id": results["ids"][0],
            "title": results["metadatas"][0]["title"],
            "messages": json.loads(results["documents"][0])
        }

    def list_chats(self) -> list:
        results = self._chat_collection.get(include=["metadatas"])
        chats = []
        for i, sid in enumerate(results["ids"]):
            chats.append({
                "session_id": sid,
                "title": results["metadatas"][i]["title"],
                "updated_at": results["metadatas"][i]["updated_at"]
            })
        chats.sort(key=lambda x: x["updated_at"], reverse=True)
        return chats

    def delete_chat(self, session_id: str) -> None:
        self._chat_collection.delete(ids=[session_id])


# ── Factory ─────────────────────────────────────────────────

_store_instance: Optional[ChromaStore] = None


def get_vector_store() -> ChromaStore:
    """Return the singleton vector store instance."""
    global _store_instance
    if _store_instance is None:
        _store_instance = ChromaStore()
    return _store_instance
