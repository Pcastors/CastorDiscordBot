import os
from langchain_community.embeddings.ollama import OllamaEmbeddings
from dotenv import load_dotenv
load_dotenv()

LLM_MODEL = os.getenv('LLM_MODEL')

def get_embedding_function():
    embeddings = OllamaEmbeddings(model=LLM_MODEL)
    return embeddings