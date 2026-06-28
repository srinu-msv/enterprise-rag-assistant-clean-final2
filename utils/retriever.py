from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import os


def load_vector_store():

    # Check if the FAISS index exists
    if not os.path.exists("faiss_index"):
        raise FileNotFoundError(
            "FAISS index not found. Please upload a PDF and create embeddings first."
        )

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = FAISS.load_local(
        folder_path="faiss_index",
        embeddings=embeddings,
        allow_dangerous_deserialization=True
    )

    return vector_store