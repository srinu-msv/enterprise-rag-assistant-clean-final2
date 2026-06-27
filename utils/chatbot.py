import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

load_dotenv()

llm = ChatGroq(
    api_key=os.getenv("GROQ_API_KEY"),
    model="llama-3.3-70b-versatile",
    temperature=0
)

def generate_answer(question, context):

    prompt = f"""
You are an Enterprise Financial Intelligence Assistant.

Answer the user's question using only the context provided.

If the answer is not available in the context, reply:
"I couldn't find the answer in the uploaded document."

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content