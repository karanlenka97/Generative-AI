from langchain_huggingface import HuggingFaceEmbeddings

embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2")

texts = [
    "Hello this is karan lenka",
    "hello you are hugging face embedding model",
    "and you are using langchain library to generate embedding for your text data"
]

vectors = embedding.embed_documents(texts)

print(vectors)
