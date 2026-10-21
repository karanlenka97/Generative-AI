from transformers import pipeline
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

CYAN = "\033[96m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RESET = "\033[0m"

chatbot = pipeline(
    "text-generation",
    model="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
)