import pickle

from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from transformers import pipeline
from rank_bm25 import BM25Okapi

VECTOR_PATH = "vector_store"
CHUNKS_PATH = "chunks.pkl"
MAX_HISTORY = 6  # keep last 3 user+assistant pairs (6 lines)

chat_history = []

# ── Load embeddings & FAISS ───────────────────────────────────────
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.load_local(
    VECTOR_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

# ── Load all chunks for BM25 (fix: was only 10 docs before) ───────
with open(CHUNKS_PATH, "rb") as f:
    all_chunks = pickle.load(f)

texts     = [chunk.page_content for chunk in all_chunks]
tokenized = [t.split() for t in texts]
bm25      = BM25Okapi(tokenized)

# ── Load FLAN-T5 ──────────────────────────────────────────────────
# Fix: use "text2text-generation" not "text-generation"
# FLAN-T5 is an encoder-decoder model — text-generation caused it
# to echo the entire prompt back as part of the output.
pipe = pipeline(
    "text2text-generation",         # ← fixed
    model="google/flan-t5-base",
    max_new_tokens=256,
    temperature=0.1,
    do_sample=False,
)

# ── Helpers ───────────────────────────────────────────────────────
def format_docs(docs):
    context = ""
    sources = []
    for d in docs:
        context += d.page_content + "\n\n"
        sources.append(d.metadata.get("source", "unknown"))
    return context.strip(), list(set(sources))


def ask(question: str) -> dict:
    # Dense retrieval (FAISS)
    vector_docs = retriever.invoke(question)

    # Sparse retrieval (BM25) over the full corpus
    tokenized_query = question.split()
    bm25_scores     = bm25.get_scores(tokenized_query)
    top_index       = int(bm25_scores.argmax())
    bm25_doc        = all_chunks[top_index]

    # Combine — deduplicate by page_content
    seen     = set()
    combined = []
    for doc in vector_docs + [bm25_doc]:
        if doc.page_content not in seen:
            seen.add(doc.page_content)
            combined.append(doc)

    context, sources = format_docs(combined)

    # Fix: cap history to avoid overflowing FLAN-T5's 512-token limit
    history_text = "\n".join(chat_history[-MAX_HISTORY:])

    prompt = f"""You are a helpful AI assistant answering questions using company documentation.
Only answer from the context provided. If the answer is not in the context, say "I don't know based on the available documents."

Conversation History:
{history_text}

Context:
{context}

Question: {question}

Answer:"""

    # Fix: call pipe() directly and extract generated_text
    # HuggingFacePipeline wrapper was masking the raw output structure
    raw    = pipe(prompt)
    answer = raw[0]["generated_text"].strip()

    # Update bounded history
    chat_history.append(f"User: {question}")
    chat_history.append(f"Assistant: {answer}")

    return {
        "answer":        answer,
        "sources":       sources,
        "num_docs_used": len(combined),
    }
