from dotenv import load_dotenv
load_dotenv()
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatOllama

search_tool = TavilySearchResults(
    max_results=5,
)

llm = ChatOllama(model="ollama-news-summarizer", base_url="http://localhost:11434")
