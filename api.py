from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from rag_pipeline import ask

app = FastAPI(
    title="RAG API",
    description="Retrieval-Augmented Generation over company documents",
    version="1.0.0",
)


class QueryRequest(BaseModel):
    query: str


class QueryResponse(BaseModel):
    answer:        str
    sources:       list[str]
    num_docs_used: int


@app.get("/")
def root():
    return {"message": "RAG API is running. POST to /ask to query your documents."}


# Fix: changed from GET to POST — queries belong in the request body,
# not the URL, especially when they can contain special characters
@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")
    try:
        result = ask(request.query)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
