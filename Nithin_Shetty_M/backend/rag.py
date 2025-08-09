import os
import json
from pathlib import Path
from sentence_transformers import SentenceTransformer
import faiss
import asyncio

FAISS_INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", "data/vectorstore"))
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

class VectorStore:
    def __init__(self):
        self.index_file = FAISS_INDEX_PATH / "faiss.index"
        self.meta_file = FAISS_INDEX_PATH / "metadata.json"
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.index = None
        self.metadata = None
        self._load()

    def _load(self):
        if not self.index_file.exists() or not self.meta_file.exists():
            raise FileNotFoundError("FAISS index or metadata not found. Build the index first.")
        self.index = faiss.read_index(str(self.index_file))
        with open(self.meta_file, "r", encoding="utf-8") as f:
            self.metadata = json.load(f)

    def search(self, query, top_k=5):
        query_vec = self.embedder.encode([query], convert_to_numpy=True)
        D, I = self.index.search(query_vec, top_k)
        results = []
        for idx in I[0]:
            if 0 <= idx < len(self.metadata):
                results.append(self.metadata[idx])
        return results

class RAGService:
    def __init__(self):
        self.vs = VectorStore()

    async def stream_answer(self, query, session_id):
        results = self.vs.search(query)
        answer_text = " ".join(results)
        for word in answer_text.split():
            chunk = json.dumps({"text": word})
            yield f"data: {chunk}\n\n"
            await asyncio.sleep(0.03)
