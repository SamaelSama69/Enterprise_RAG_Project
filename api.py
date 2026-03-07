
from fastapi import FastAPI
from rag_pipeline import ask

app = FastAPI()

@app.get("/")
def root():
    return {"message": "Production RAG API running"}

@app.get("/ask")
def ask_question(query: str):
    return ask(query)
