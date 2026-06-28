import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Initialize OpenAI LLM
llm = ChatOpenAI(
    api_key=os.getenv("OPENAI_API_KEY"),
    model="gpt-4o-mini",  # or "gpt-4o" / "gpt-4-turbo"
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
