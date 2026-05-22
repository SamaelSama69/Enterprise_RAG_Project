import os
import pickle

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

DATA_PATH   = "data"
VECTOR_PATH = "vector_store"
CHUNKS_PATH = "chunks.pkl"

# ── Load all PDFs ─────────────────────────────────────────────────
documents = []

for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DATA_PATH, file))
        documents.extend(loader.load())

print(f"Loaded pages: {len(documents)}")

if len(documents) == 0:
    raise ValueError("No PDFs found in the data/ folder. Add at least one PDF before ingesting.")

# ── Chunk ─────────────────────────────────────────────────────────
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
chunks = splitter.split_documents(documents)

print(f"Chunks: {len(chunks)}")

# ── Embed & save FAISS vector store ──────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vectorstore = FAISS.from_documents(chunks, embeddings)
vectorstore.save_local(VECTOR_PATH)

print(f"Vector store saved to: {VECTOR_PATH}/")

# ── Save raw chunks for BM25 (fix: was only using 10 docs before) ─
with open(CHUNKS_PATH, "wb") as f:
    pickle.dump(chunks, f)

print(f"Chunks saved to: {CHUNKS_PATH}")
print("Ingestion complete.")
