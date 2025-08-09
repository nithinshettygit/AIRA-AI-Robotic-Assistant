import os
import json
from pathlib import Path
from dotenv import load_dotenv

from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.docstore.document import Document
from langchain.vectorstores import FAISS

load_dotenv()

EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
FAISS_INDEX_PATH = Path(os.getenv("FAISS_INDEX_PATH", "data/vectorstore"))
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "500"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "100"))
INPUT_FILE = Path("data/knowledgebase.json")

def flatten_nested_json(obj, prefix=""):
    docs = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            new_prefix = prefix + k + " " if prefix else k + " "
            docs.extend(flatten_nested_json(v, new_prefix))
    elif isinstance(obj, str):
        docs.append(Document(page_content=obj))
    return docs

def main():
    print(f"[INFO] Loading {INPUT_FILE} ...")
    if not INPUT_FILE.exists():
        raise FileNotFoundError(f"Input JSON file not found: {INPUT_FILE}")

    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    docs = flatten_nested_json(data)
    print(f"[INFO] Total documents before chunking: {len(docs)}")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    chunks = splitter.split_documents(docs)
    print(f"[INFO] Total chunks after splitting: {len(chunks)}")

    embeddings = SentenceTransformerEmbeddings(model_name=EMBEDDING_MODEL)
    print(f"[INFO] Creating FAISS index with embedding model '{EMBEDDING_MODEL}'...")
    faiss_index = FAISS.from_documents(chunks, embeddings)

    FAISS_INDEX_PATH.mkdir(parents=True, exist_ok=True)
    faiss_index.save_local(str(FAISS_INDEX_PATH))

    print(f"[SUCCESS] FAISS index and metadata saved in {FAISS_INDEX_PATH}")

if __name__ == "__main__":
    main()
