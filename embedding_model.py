import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")
def get_embedding_function():
    embeddings = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

    return embeddings