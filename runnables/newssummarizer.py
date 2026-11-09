from dotenv import load_dotenv
load_dotenv()
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

search_tool = TavilySearchResults(
    max_results=5,
)

llm = ChatOllama(model="qwen:3b",temperature = 0.1)

prompt = ChatPromptTemplate.from_template(
    """
You are a helpful assistant

summarize the following news into clear bullet points
{news}
"""
)

chain = prompt | llm | StrOutputParser()

news_result = search_tool.run("Latest news about AI") 

result = chain.invoke({"news":news_result})

print(result)
