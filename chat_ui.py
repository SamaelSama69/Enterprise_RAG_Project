
import streamlit as st
from rag_pipeline import ask

st.title("GenAI RAG Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

query = st.chat_input("Ask a question")

if query:
    response = ask(query)
    st.session_state.messages.append(("user", query))
    st.session_state.messages.append(("assistant", response["answer"]))

for role, msg in st.session_state.messages:
    with st.chat_message(role):
        st.write(msg)
