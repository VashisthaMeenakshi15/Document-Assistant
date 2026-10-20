# This file supports multi session RAG pipelines with separate ChromaDB collections per session. 
# It is designed for Streamlit apps where each user session can have its own workspace and indexed documents.
# Using GROQ LLMs for RAG (Retrieval-Augmented Generation) with LangChain and ChromaDB

import os
import re
from pathlib import Path
import streamlit as st
from dotenv import dotenv_values
from streamlit.errors import StreamlitSecretNotFoundError
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_groq import ChatGroq
from langchain_classic.chains import create_retrieval_chain
from langchain_classic.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from prompts import RAG_SYSTEM_PROMPT


class RAGPipeline:
    def __init__(
        self, 
        embedding_model_name: str = "BAAI/bge-small-en-v1.5",
        collection_name: str = "default_session"
    ):
        # Format workspace name into a valid Chroma collection identifier
        clean_name = re.sub(r'[^a-zA-Z0-9_-]', '_', collection_name).lower().strip("_")
        self.collection_name = clean_name if clean_name else "default_session"
        
        # Embeddings remain local and lightweight inside ChromaDB
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.vectorstore = None
    
    def get_retriever(self, k: int = 3):
        """Returns the retriever for the active vectorstore, or None if not initialized."""
        if self.vectorstore is None:
            return None
        return self.vectorstore.as_retriever(search_kwargs={"k": k})

    def add_to_vectorstore(self, documents):
        """Creates or incrementally appends document chunks into the workspace collection."""
        if self.vectorstore is None:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                collection_name=self.collection_name
            )
        else:
            self.vectorstore.add_documents(documents)

    def query(self, question: str, model_name: str = "qwen/qwen3.8-27b"):
        """
        Retrieves context chunks and streams Groq LLM responses with citations.
        """
        if self.vectorstore is None:
            raise ValueError("Vector store has not been initialized yet in this workspace.")

        # Read from process environment, project .env, or Streamlit secrets
        groq_api_key = os.environ.get("GROQ_API_KEY") or dotenv_values(
            Path(__file__).with_name(".env")
        ).get("GROQ_API_KEY")
        if not groq_api_key:
            try:
                groq_api_key = st.secrets.get("GROQ_API_KEY")
            except StreamlitSecretNotFoundError:
                groq_api_key = None
        if not groq_api_key:
            raise ValueError(
                "GROQ_API_KEY not found. Set it as an environment variable or "
                "in .streamlit/secrets.toml."
            )

        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})

        # Initialize Groq Cloud LLM runner
        llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name=model_name,
            temperature=0.2
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", RAG_SYSTEM_PROMPT),
            ("human", "{input}"),
        ])

        question_answer_chain = create_stuff_documents_chain(llm, prompt)
        rag_chain = create_retrieval_chain(retriever, question_answer_chain)

        response = rag_chain.invoke({"input": question})
        answer = response["answer"]

        # Extract source citations dynamically
        sources = response.get("context", [])
        doc_refs = set()
        for doc in sources:
            filename = doc.metadata.get("source", "Document")
            loc_type = doc.metadata.get("location_type", "Location")
            loc_num = doc.metadata.get("location_number", "N/A")
            doc_refs.add(f"{filename} ({loc_type} {loc_num})")

        return answer, list(doc_refs)