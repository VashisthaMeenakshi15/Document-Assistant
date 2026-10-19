#Using GROQ LLMs for RAG (Retrieval-Augmented Generation) with LangChain and ChromaDB

import os
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
    def __init__(self, embedding_model_name: str = "BAAI/bge-small-en-v1.5"):
        # Embeddings remain 100% local and lightweight inside ChromaDB
        self.embeddings = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.vectorstore = None

    def build_vectorstore(self, documents):
        """Creates an in-memory ChromaDB vectorstore from document chunks."""
        self.vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=self.embeddings
        )

    def query(self, question: str, model_name: str = "qwen/qwen3.8-27b"):
        """
        Retrieves context chunks and streams Groq LLM responses with citations.
        """
        if self.vectorstore is None:
            raise ValueError("Vector store has not been initialized yet.")

        # Read from the process environment, the project .env, or Streamlit secrets.
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