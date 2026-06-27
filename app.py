import os
import streamlit as st
from utils.loader import load_pdf
from utils.splitter import split_documents
from utils.embedder import create_vector_store
from utils.retriever import load_vector_store
from utils.chatbot import generate_answer

# Page Configuration
st.set_page_config(page_title="Enterprise Financial Intelligence Assistant", page_icon="💼")

# App Title
st.title("💼 Enterprise Financial Intelligence Assistant")

# File Upload
uploaded_file = st.file_uploader(
    "Upload your Financial PDF",
    type=["pdf"]
)

if uploaded_file is not None:
    os.makedirs("data", exist_ok=True)
    file_path = os.path.join("data", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success("✅ PDF Uploaded Successfully!")

    documents = load_pdf(file_path)
    chunks = split_documents(documents)
    st.success(f"✅ Total Chunks Created: {len(chunks)}")

    vector_store = create_vector_store(chunks)
    st.success("✅ Embeddings Created Successfully!")

    vector_store.save_local("faiss_index")
    st.success("✅ FAISS Index Saved Successfully!")

    st.subheader("First Chunk")
    st.write(chunks[0].page_content)

# -------------------------
# Question Answering Section
# -------------------------

st.markdown("---")
st.subheader("Ask a Question")

if "history" not in st.session_state:
    st.session_state["history"] = []

question = st.text_input("Enter your question")

if question:
    vector_store = load_vector_store()
    results = vector_store.similarity_search(question, k=3)

    if not results:
        st.warning("No relevant information found in the uploaded documents.")
    else:
        context = "\n\n".join([doc.page_content for doc in results])
        answer = generate_answer(question, context)

        st.subheader("AI Answer")
        st.write(answer)

        # Save Q&A to history
        st.session_state["history"].append((question, answer))

        # Show retrieved chunks
        st.subheader("Retrieved Context")
        for i, doc in enumerate(results, start=1):
            st.write(f"### Result {i}")
            st.write(doc.page_content)

        # Show sources if available
        st.subheader("Sources")
        for doc in results:
            if "source" in doc.metadata:
                st.write(f"- {doc.metadata['source']}")

# Show conversation history
if st.session_state["history"]:
    st.markdown("---")
    st.subheader("Conversation History")
    for q, a in st.session_state["history"]:
        st.write(f"**Q:** {q}")
        st.write(f"**A:** {a}")
