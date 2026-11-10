from dotenv import load_dotenv
from langchain_ollama import ChatOllama
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_ollama import ChatOllama
from rich import print

#1 creating a tool
@tool
def get_text_length(text: str) -> int:
    """Returns the number of character in a given text"""
    return len(text)

llm = ChatOllama(model="qwen3:8b")

