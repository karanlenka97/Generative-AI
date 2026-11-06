from dotenv import load_dotenv
load_dotenv()

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Create a prompt template with a variable for the subject to explain.
prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words"
)

# 2. Model
model = ChatMistralAI(model="mistral-small-2506")

# 3. Output Parser
parser = StrOutputParser()

# Connect the prompt, model, and parser into one runnable chain.
chain = prompt | model | parser

# Run the chain with the topic to explain.
result = chain.invoke("Machine Learning")
print(result)