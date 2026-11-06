from dotenv import load_dotenv
load_dotenv()
# Load API credentials and other configuration from the .env file.

from langchain_mistralai import ChatMistralAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

model = ChatMistralAI(model="mistral-small-2506")
parser = StrOutputParser()

# Ask the model to generate code for the requested topic.
code_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a code generator"),
    ("human", "{topic}")
])

# Ask the model to explain generated code in simple terms.
explain_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant who explains code in simple terms"),
    ("human", "Explain the following code in simple words:\n{code}")
])

# Generate code from the user's request.
seq = code_prompt | model | parser 

# Preserve the generated code and pass it to the explanation prompt.
# The lambda converts the code string into the mapping expected by that prompt.
seq2 = RunnableParallel(
    {"code" :  RunnablePassthrough(),
     "explanation" : explain_prompt | model | parser
    }
)

chain = seq | seq2

# Generate the code, then produce its explanation.
result = chain.invoke({"topic" : "please write a code of palindrome in python "})

print(result['code'])
print(result['explanation'])
