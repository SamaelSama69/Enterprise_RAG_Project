
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
import os

DATA_PATH = "data"
VECTOR_PATH = "vector_store"

documents = []

for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DATA_PATH, file))
        documents.extend(loader.load())

print("Loaded pages:", len(documents))

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
docs = splitter.split_documents(documents)

print("Chunks:", len(docs))

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.save_local(VECTOR_PATH)

print("Vector database created.")
