import os
from uuid import uuid4

import chromadb
from dotenv import load_dotenv
from langchain import hub
from langchain_chroma import Chroma
from langchain_community.cache import SQLiteCache
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_core.globals import set_llm_cache
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_groq import ChatGroq
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()
set_llm_cache(SQLiteCache(database_path="langchain.db"))

# set the env var GROQ_API_KEY
model = ChatGroq(model="llama-3.2-1b-preview")
prompt = hub.pull("rlm/rag-prompt")

embeddings_model = OllamaEmbeddings(model="nomic-embed-text")
persistent_client = chromadb.PersistentClient()
collection = persistent_client.get_or_create_collection("qna_collection")
vector_store = Chroma(
    client=persistent_client,
    collection_name="qna_collection",
    embedding_function=embeddings_model,
)


def add_document_embeddings(filename):
    results = collection.get(where={"source": filename}, include=["metadatas"])
    if len(results["ids"]) > 0:
        print("document already embedded")
        return
    raw_document = TextLoader(filename).load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    documents = text_splitter.split_documents(raw_document)

    uuids = [str(uuid4()) for _ in range(len(documents))]
    vector_store.add_documents(documents=documents, ids=uuids)
    print(f"{filename} embeddings added")


def generate_answer(question):
    data = vector_store.similarity_search(question)
    docs_content = "\n\n".join(entry.page_content for entry in data)
    messages = prompt.invoke({"question": question, "context": docs_content})
    response = model.invoke(messages)
    print(response.content)


if __name__ == "__main__":
    add_document_embeddings("../langchain-basics/article.txt")
    while True:
        question = input("question: ")
        if question in ["q", "bye", "quit", "exit"]:
            print("thank you")
            break
        generate_answer(question)
