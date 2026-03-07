
# 🧠 Production RAG Assistant
### Enterprise-Style Retrieval Augmented Generation System

![Python](https://img.shields.io/badge/Python-3.10+-blue)
![LangChain](https://img.shields.io/badge/LangChain-RAG-green)
![FAISS](https://img.shields.io/badge/VectorDB-FAISS-orange)
![HuggingFace](https://img.shields.io/badge/HuggingFace-Transformers-yellow)
![FastAPI](https://img.shields.io/badge/API-FastAPI-009688)
![Streamlit](https://img.shields.io/badge/UI-Streamlit-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

---

# 📌 Project Overview

Production RAG Assistant is a Retrieval-Augmented Generation (RAG) system designed to answer questions using custom documents such as company policies, support documents, or internal knowledge bases.

The system retrieves relevant document chunks and uses a Large Language Model (LLM) to generate accurate responses grounded in the retrieved context.

This project demonstrates a complete production-style GenAI pipeline including:

- document ingestion
- embedding generation
- hybrid retrieval
- LLM generation
- API deployment
- chat interface

---

# 🚀 Key Features

✔ Retrieval-Augmented Generation pipeline  
✔ FAISS vector database  
✔ Hybrid search (Vector + BM25)  
✔ HuggingFace embedding models  
✔ FLAN-T5 LLM for text generation  
✔ Context-aware conversation memory  
✔ FastAPI backend  
✔ Streamlit chatbot interface  
✔ Source document referencing  

---

# 🏗 System Architecture

```
                ┌─────────────────────┐
                │   PDF Documents     │
                └─────────┬───────────┘
                          │
                          ▼
                 Document Loader
                      (LangChain)
                          │
                          ▼
                     Text Chunking
                          │
                          ▼
                   Embedding Model
              (Sentence Transformers)
                          │
                          ▼
                 FAISS Vector Database
                          │
                          ▼
                 Hybrid Retrieval Layer
             (Vector Search + BM25 Search)
                          │
                          ▼
                 Context Assembly
                          │
                          ▼
                     FLAN-T5 LLM
                          │
                          ▼
                 Generated Answer
                          │
                          ▼
               FastAPI + Streamlit UI
```

---

# 📂 Project Structure

```
production-rag-assistant
│
├── api.py
├── chat_ui.py
├── ingest.py
├── rag_pipeline.py
├── requirements.txt
├── README.md
│
├── data/
│   ├── company_policy.pdf
│   └── faq.pdf
│
└── vector_store/
    ├── index.faiss
    └── index.pkl
```

---

# ⚙️ Technology Stack

### GenAI / NLP

- LangChain
- HuggingFace Transformers
- Sentence Transformers
- FLAN-T5 LLM

### Retrieval

- FAISS Vector Database
- BM25 Keyword Retrieval

### Backend

- FastAPI
- Uvicorn

### Frontend

- Streamlit Chat UI

### Document Processing

- PyPDF

---

# 📦 Requirements

Python >= 3.10

---

# 🛠 Installation

## Clone Repository

```
git clone https://github.com/yourusername/production-rag-assistant.git
cd production-rag-assistant
```

## Create Virtual Environment

Windows:

```
python -m venv venv
venv\Scripts\activate
```

Mac / Linux:

```
python3 -m venv venv
source venv/bin/activate
```

## Install Dependencies

```
pip install -r requirements.txt
```

---

# 📄 Adding Documents

Place PDF documents inside the data folder.

Example:

```
data/
 ├── return_policy.pdf
 ├── shipping_info.pdf
 └── faq.pdf
```

---

# 🧠 Building the Vector Database

Run:

```
python ingest.py
```

Example output:

```
Loaded pages: 15
Chunks: 64
Vector database created.
```

---

# 🚀 Running the System

## Start API

```
python -m uvicorn api:app --reload
```

API docs:

```
http://127.0.0.1:8000/docs
```

## Start Chat UI

```
streamlit run chat_ui.py
```

Open:

```
http://localhost:8501
```

---

# 🔄 End-to-End Workflow

```
PDF Documents
     │
     ▼
Document Loader (LangChain)
     │
     ▼
Text Chunking
     │
     ▼
Embedding Generation
(Sentence Transformers)
     │
     ▼
Vector Storage (FAISS)
     │
     ▼
Hybrid Retrieval (Vector + BM25)
     │
     ▼
Relevant Context
     │
     ▼
Prompt Construction
     │
     ▼
FLAN-T5 Generation
     │
     ▼
Answer + Sources
     │
     ▼
Streamlit Chat UI
```

---

# 📊 Skills Demonstrated

• Retrieval-Augmented Generation (RAG)  
• Vector databases and similarity search  
• Hybrid retrieval techniques  
• Large Language Model integration  
• FastAPI backend development  
• Streamlit UI development  
• Production-style AI system architecture  

---

# 🎯 Use Cases

- Customer support assistants
- Enterprise knowledge retrieval
- Internal documentation search
- Policy / compliance assistants
- Intelligent document Q&A systems

---

# 👨‍💻 Author

Sham Solanki  
Master's in Data Science – Deakin University

---

# 📜 License

MIT License
