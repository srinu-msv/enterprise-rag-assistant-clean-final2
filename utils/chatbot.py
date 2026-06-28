print("========== GEMINI CHATBOT LOADED ==========")

import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI

# Load environment variables
load_dotenv()

# Initialize Gemini LLM
llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0
)

def generate_answer(question, context):

    prompt = f"""
You are an Enterprise Financial Intelligence Assistant.

Answer ONLY using the context provided.

If the answer is not available in the context,
reply exactly:

"I couldn't find the answer in the uploaded document."

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    return response.content