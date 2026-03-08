
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import pipeline
from rank_bm25 import BM25Okapi

VECTOR_PATH = "vector_store"

chat_history = []

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

vectorstore = FAISS.load_local(
    VECTOR_PATH,
    embeddings,
    allow_dangerous_deserialization=True
)

retriever = vectorstore.as_retriever(search_kwargs={"k":3})

docs = vectorstore.similarity_search("test", k=10)
texts = [d.page_content for d in docs]
tokenized = [t.split(" ") for t in texts]
bm25 = BM25Okapi(tokenized)

pipe = pipeline(
    "text-generation",
    model="google/flan-t5-base",
    max_new_tokens=256,
    temperature=0.1
)

llm = HuggingFacePipeline(pipeline=pipe)

def format_docs(docs):
    context = ""
    sources = []

    for d in docs:
        context += d.page_content + "\n\n"
        sources.append(d.metadata.get("source", "document"))

    return context, list(set(sources))

def ask(question):

    vector_docs = retriever.invoke(question)

    tokenized_query = question.split(" ")
    bm25_scores = bm25.get_scores(tokenized_query)
    top_index = bm25_scores.argmax()
    bm25_doc = docs[top_index]

    docs_combined = vector_docs + [bm25_doc]

    context, sources = format_docs(docs_combined)

    history_text = "\n".join(chat_history)

    prompt = f"""
You are a helpful AI assistant answering questions using company documentation.

Conversation History:
{history_text}

Context:
{context}

Question:
{question}

Provide a clear answer and reference the sources.
"""

    answer = llm.invoke(prompt)

    chat_history.append(f"User: {question}")
    chat_history.append(f"Assistant: {answer}")

    return {
        "answer": answer,
        "sources": sources,
        "num_docs_used": len(docs_combined)
    }
