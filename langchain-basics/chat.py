import os
import chromadb

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.globals import set_llm_cache
from langchain_community.cache import SQLiteCache
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_community.tools import DuckDuckGoSearchRun, DuckDuckGoSearchResults
from langchain_community.utilities import DuckDuckGoSearchAPIWrapper

load_dotenv()

# save llm query to sqlite cache
set_llm_cache(SQLiteCache(database_path="langchain.db"))

# load the llm model
model = ChatGroq(model="llama-3.2-1b-preview")


def basic_llm_action():
    messages = [
        SystemMessage("Translate the following from English to Hindi"),
        HumanMessage("hi!"),
    ]
    response = model.invoke(messages)
    print(response)

def llm_action_in_stream():
    for token in model.stream(["tell me a joke"]):
        print(token.content, end="!")


def add_embeddings():
    raw_documents = TextLoader("article.txt").load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    documents = text_splitter.split_documents(raw_documents)
    db = Chroma.from_documents(documents, OllamaEmbeddings(model="nomic-embed-text"))
    query = "who is the rival of openai?"
    query = "what are the shocking features of deepseek?"
    query = "when was deepseek released?"
    query = "CEO of Y Combinator"
    docs = db.similarity_search(query)
    print(docs[0].page_content)

def ddg():
    search = DuckDuckGoSearchRun()
    response = search.invoke("when was linux invented?")
    print(response)
    print()
    search_list = DuckDuckGoSearchResults(output_format="list")
    response = search_list.invoke("usr inr conversion rate")
    print(response)
    print()
    wrapper = DuckDuckGoSearchAPIWrapper(region="in-en", time="d", source="news")
    news_search = DuckDuckGoSearchResults(output_format="list", api_wrapper=wrapper)
    response = search_list.invoke("latest tech news")
    print(response)


if __name__ == "__main__":
    # basic_llm_action()
    # llm_action_in_stream()
    # add_embeddings()
    # ddg()
