from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
import os

load_dotenv()

embeddings = OpenAIEmbeddings(model="text-embedding-3-small", 
                              chunk_size=1)

vector = embeddings.embed_query("Hello world")

print(vector)