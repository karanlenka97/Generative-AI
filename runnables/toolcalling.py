from dotenv import load_dotenv
from langchain_ollama import ChatOllama
load_dotenv()
from langchain_mistralai import ChatMistralAI
from langchain.tools import tool
from langchain_ollama import ChatOllama
from rich import print
from langchain_core.messages import HumanMessage

#1 creating a tool
@tool
def get_text_length(text: str) -> int:
    """Returns the number of character in a given text"""
    return len(text)

llm = ChatOllama(model="qwen3:8b")

#2 Tool binding with LLM
llm_with_tool = llm.bind_tools([get_text_length])

message = []
query = HumanMessage("Please tell me the length of the following text: 'Hello, this is a sample text for testing the tool binding with LLM.'")
print(message)