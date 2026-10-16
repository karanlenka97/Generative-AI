import os
from pathlib import Path

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

project_root = Path(__file__).resolve().parent.parent
load_dotenv(project_root / ".env", override=True)

hf_token = os.getenv("HUGGINGFACEHUB_API_TOKEN")
if not hf_token:
    raise ValueError("Hugging Face API token not found. Check your .env file.")

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
    task="text-generation",
    huggingfacehub_api_token=hf_token,
    max_new_tokens=200,
)

model = ChatHuggingFace(llm=llm)

response = model.invoke("Who are you?")
print(response.content)