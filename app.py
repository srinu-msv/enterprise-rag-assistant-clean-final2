import os
import streamlit as st
from utils.loader import load_pdf
from utils.splitter import split_documents
from utils.embedder import create_vector_store
from utils.retriever import load_vector_store
from utils.chatbot import generate_answer

# Page Configuration
st.set_page_config(page_title="Enterprise Financial Intelligence Assistant")

# App Title
st.title("📄 Enterprise Financial Intelligence Assistant")

# File Upload
uploaded_file = st.file_uploader(
    "Upload your Financial PDF",
    type=["pdf"]
)

if uploaded_file is not None:

    # Create data folder if it doesn't exist
    os.makedirs("data", exist_ok=True)

    # Save uploaded file
    file_path = os.path.join("data", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ PDF Uploaded Successfully!")

    # Load PDF
    documents = load_pdf(file_path)

    # Split PDF into chunks
    chunks = split_documents(documents)

    st.success(f"✅ Total Chunks Created: {len(chunks)}")

    # Create Embeddings and FAISS Vector Store
    vector_store = create_vector_store(chunks)

    st.success("✅ Embeddings Created Successfully!")

    # Save FAISS Index
    vector_store.save_local("faiss_index")

    st.success("✅ FAISS Index Saved Successfully!")

    # Display first chunk
    st.subheader("First Chunk")

    st.write(chunks[0].page_content)
    # Display first chunk

# -------------------------
# Add the new code BELOW
# -------------------------

st.markdown("---")

st.subheader("Ask a Question")

question = st.text_input("Enter your question")

if question:

    vector_store = load_vector_store()

    results = vector_store.similarity_search(question, k=3)

    # Combine retrieved chunks
    context = "\n\n".join([doc.page_content for doc in results])

    # Generate answer using ChatGPT
    answer = generate_answer(question, context)

    st.subheader("AI Answer")
    st.write(answer)

    # Show retrieved chunks
    st.subheader("Retrieved Context")

    for i, doc in enumerate(results, start=1):
        st.write(f"### Result {i}")
        st.write(doc.page_content)