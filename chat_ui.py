import streamlit as st
from rag_pipeline import ask

st.set_page_config(page_title="RAG Assistant", page_icon="🤖")
st.title("GenAI RAG Assistant")

if "messages" not in st.session_state:
    st.session_state.messages = []

# Fix: render existing history FIRST so messages appear in correct order
# and the UI doesn't freeze while the model runs
for role, msg, *extra in st.session_state.messages:
    with st.chat_message(role):
        st.write(msg)
        if extra and role == "assistant":
            sources = extra[0]
            if sources:
                st.caption(f"📄 Sources: {', '.join(sources)}")

# Then handle new input
query = st.chat_input("Ask a question about your documents...")

if query:
    # Show user message immediately — don't wait for the model
    with st.chat_message("user"):
        st.write(query)
    st.session_state.messages.append(("user", query))

    # Run model with spinner so UI doesn't look frozen
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            response = ask(query)

        answer  = response["answer"]
        sources = response["sources"]

        st.write(answer)
        if sources:
            st.caption(f"📄 Sources: {', '.join(sources)}")

    st.session_state.messages.append(("assistant", answer, sources))
