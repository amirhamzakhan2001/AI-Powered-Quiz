# backend/rag_engine.py

import os
from typing import Optional

from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document



def initialize_rag_db(text_content: str, api_key: str) -> FAISS:
    """
    Initializes a FAISS vector store with the provided text content.
    Splits the text into chunks, creates embeddings, and stores them.
    """
    # Ensure API key is available for embeddings
    if not api_key:
        raise ValueError("Google Gemini API Key not provided for RAG embeddings.")

    # Initialize embeddings model
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    
    # Split the text into manageable chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_text(text_content)
    
    # Create Document objects
    docs = [Document(page_content=chunk) for chunk in chunks]
    
    # Create a FAISS vector store from the documents and embeddings
    vector_store = FAISS.from_documents(docs, embeddings)
    return vector_store










def get_rag_context(query, vector_store: FAISS, k: int = 2) -> Optional[str]:
    """
    Retrieves relevant context from the FAISS vector store based on the query.
    Returns a concatenated string of the top k relevant documents.
    """

    # Ensure always a string
    if isinstance(query, list):
        query = ", ".join(str(item) for item in query)
    docs = vector_store.similarity_search(query, k=k)
    if not docs:
        return None
    return "\n\n".join(doc.page_content for doc in docs)
