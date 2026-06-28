import os
import streamlit as st

from utils.loader import load_pdf
from utils.splitter import split_documents
from utils.embedder import create_vector_store
from utils.chatbot import generate_answer

st.set_page_config(
    page_title="Enterprise Financial Intelligence Assistant",
    page_icon="💼",
)

st.title("💼 Enterprise Financial Intelligence Assistant")

# -----------------------------
# Session State
# -----------------------------
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "history" not in st.session_state:
    st.session_state.history = []

# -----------------------------
# Upload PDF
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload Financial PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    os.makedirs("data", exist_ok=True)

    file_path = os.path.join(
        "data",
        uploaded_file.name
    )

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ PDF Uploaded Successfully!")

    # Load PDF
    documents = load_pdf(file_path)

    st.write(f"Pages Loaded: {len(documents)}")

    if len(documents) == 0:
        st.error("No text found inside PDF.")
        st.stop()

    # Split PDF
    chunks = split_documents(documents)

    st.success(f"Chunks Created: {len(chunks)}")

    # Show first chunk
    st.subheader("First Chunk")

    st.write(chunks[0].page_content)

    # Create Vector Store
    vector_store = create_vector_store(chunks)

    st.session_state.vector_store = vector_store

    st.success("✅ Embeddings Created")

# -----------------------------
# Ask Questions
# -----------------------------
st.markdown("---")

st.subheader("Ask Questions")

question = st.text_input("Enter your question")

if question:

    if st.session_state.vector_store is None:

        st.warning("Please upload a PDF first.")

    else:

        results = st.session_state.vector_store.similarity_search(
            question,
            k=3
        )

        context = "\n\n".join(
            [doc.page_content for doc in results]
        )

        answer = generate_answer(
            question,
            context
        )

        st.subheader("Answer")

        st.write(answer)

        st.session_state.history.append(
            (question, answer)
        )

        st.subheader("Retrieved Chunks")

        for i, doc in enumerate(results):

            st.markdown(f"### Chunk {i+1}")

            st.write(doc.page_content)

            st.write(doc.metadata)

# -----------------------------
# History
# -----------------------------
if st.session_state.history:

    st.markdown("---")

    st.subheader("Conversation History")

    for q, a in st.session_state.history:

        st.write("**Question:**", q)

        st.write("**Answer:**", a)