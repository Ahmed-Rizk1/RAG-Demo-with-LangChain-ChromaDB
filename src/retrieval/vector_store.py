import os
import shutil
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import VectorStoreRetriever
from typing import List
from langchain_core.documents import Document
from src.config import settings

def get_embeddings():
    """Returns the embedding model."""
    return OpenAIEmbeddings(model=settings.EMBEDDING_MODEL)

def create_vector_store(documents: List[Document], reset: bool = False):
    """
    Creates and persists the vector store.
    If reset is True, deletes existing DB first.
    """
    if reset and os.path.exists(settings.PERSIST_DIRECTORY):
        shutil.rmtree(settings.PERSIST_DIRECTORY)
        
    embeddings = get_embeddings()
    
    # If documents provided, create/update db
    if documents:
        print(f"Creating vector store with {len(documents)} chunks...")
        db = Chroma.from_documents(
            documents=documents,
            embedding=embeddings,
            persist_directory=settings.PERSIST_DIRECTORY,
            collection_name=settings.COLLECTION_NAME
        )
        print(f"Vector store created at {settings.PERSIST_DIRECTORY}")
        return db
    else:
        print("No documents to ingest.")
        return None

def get_retriever() -> VectorStoreRetriever:
    """
    Returns a retriever connected to the existing vector store.
    """
    embeddings = get_embeddings()
    
    if not os.path.exists(settings.PERSIST_DIRECTORY):
        raise ValueError(f"Vector store not found at {settings.PERSIST_DIRECTORY}. Please run ingestion first.")

    db = Chroma(
        persist_directory=settings.PERSIST_DIRECTORY,
        embedding_function=embeddings,
        collection_name=settings.COLLECTION_NAME
    )
    
    return db.as_retriever(
        search_type="similarity",
        search_kwargs={"k": 3}, 
    )
