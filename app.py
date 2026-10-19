import os
import streamlit as st
from doc_parser import parse_documents
from rag import RAGPipeline

# -----------------------------------------------------------------------------
# 1. Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Enterprise Local RAG",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# -----------------------------------------------------------------------------
# 2. External CSS Loader Helper
# -----------------------------------------------------------------------------
def load_css(file_path: str = "style.css"):
    """Reads external CSS file and injects rules into Streamlit."""
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    else:
        st.warning(f"Stylesheet '{file_path}' not found.")

# Load custom styles
load_css("style.css")

# -----------------------------------------------------------------------------
# 3. Session State Initialization
# -----------------------------------------------------------------------------
if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = RAGPipeline()

if "indexed" not in st.session_state:
    st.session_state.indexed = False

if "chunk_count" not in st.session_state:
    st.session_state.chunk_count = 0

if "messages" not in st.session_state:
    st.session_state.messages = []

# -----------------------------------------------------------------------------
# 4. Sidebar Controls & Document Ingestion
# -----------------------------------------------------------------------------
with st.sidebar:
    st.markdown("### 📁 Knowledge Base")
    
    uploaded_files = st.file_uploader(
        "Upload Documents", 
        type=["pptx", "pdf", "docx", "html", "htm", "xlsx", "xls", "csv"], 
        accept_multiple_files=True,
        help="Supports PPTX, PDF, Word, HTML, Excel, and CSV files."
    )
    
    st.markdown("---")
    st.markdown("### ⚙️ LLM Configuration")
    # Ollama models
    # selected_model = st.selectbox(
        # "Model Architecture", 
        # ["qwen2.5:3b", "llama3.2:3b", "qwen2.5:7b", "mistral:7b"], 
        # index=0
    # )
    # Current Groq model IDs; older Llama and Qwen model IDs have been retired.
    selected_model = st.selectbox(
        "Model Architecture", 
        [
        "qwen/qwen3.8-27b",
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b"
        ], 
        index=0
    )
    
    process_btn = st.button("🚀 Build Vector Store", use_container_width=True)

    # File Ingestion Action
    if process_btn and uploaded_files:
        with st.spinner("Processing & indexing documents..."):
            docs = parse_documents(uploaded_files)
            st.session_state.rag_pipeline.build_vectorstore(docs)
            st.session_state.indexed = True
            st.session_state.chunk_count = len(docs)
            st.success("Vector DB successfully created!")

    # Knowledge Base Metrics Summary Card
    if st.session_state.indexed:
        st.markdown(f"""
        <div class="sidebar-card">
            <span style="color: #10B981; font-weight: bold;">● Status: Ready</span><br/>
            <small style="color: #9CA3AF;">Files Uploaded: <b>{len(uploaded_files)}</b></small><br/>
            <small style="color: #9CA3AF;">Total Chunks Indexed: <b>{st.session_state.chunk_count}</b></small>
        </div>
        """, unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 5. Header & Starter Prompt Chips
# -----------------------------------------------------------------------------
st.markdown('<p class="main-header">⚡ Local Document RAG Assistant</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-header">Multi-Format Document Intelligence powered by Groq Cloud & ChromaDB</p>', unsafe_allow_html=True)

# Quick Starter Prompt Buttons
if st.session_state.indexed and len(st.session_state.messages) == 0:
    st.markdown("##### 💡 Suggested Questions")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if st.button("📝 Summarize key topics", use_container_width=True):
            st.session_state.prompt_trigger = "Provide a high-level summary of the main topics discussed across the uploaded documents."
    with col2:
        if st.button("📊 Extract main metrics", use_container_width=True):
            st.session_state.prompt_trigger = "Extract all key numerical data, dates, and metrics mentioned in the documents."
    with col3:
        if st.button("❓ What are action items?", use_container_width=True):
            st.session_state.prompt_trigger = "List all important action items, conclusions, or next steps found in the documents."

# Handle clicked prompt chips
user_input = st.chat_input("Ask a question about your documents...")
if "prompt_trigger" in st.session_state:
    user_input = st.session_state.prompt_trigger
    del st.session_state.prompt_trigger

# -----------------------------------------------------------------------------
# 6. Chat History Rendering & Input Handling
# -----------------------------------------------------------------------------
for msg in st.session_state.messages:
    avatar = "🧑‍💻" if msg["role"] == "user" else "🤖"
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])

if user_input:
    # 1. User Input (Right Aligned via CSS)
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(user_input)

    # 2. Assistant Output (Left Aligned via CSS)
    if not st.session_state.indexed:
        with st.chat_message("assistant", avatar="🤖"):
            st.warning("Please upload documents and click 'Build Vector Store' in the sidebar first.")
    else:
        with st.chat_message("assistant", avatar="🤖"):
            with st.spinner("Searching knowledge base..."):
                answer, doc_refs = st.session_state.rag_pipeline.query(
                    question=user_input, 
                    model_name=selected_model
                )

                final_output = answer
                if doc_refs:
                    final_output += "\n\n"
                    st.markdown(answer)
                    
                    with st.expander("🔍 **Retrieved Sources & Citations**", expanded=False):
                        for ref in doc_refs:
                            st.markdown(f'<span class="citation-tag">📄 {ref}</span>', unsafe_allow_html=True)
                else:
                    st.markdown(final_output)

                st.session_state.messages.append({"role": "assistant", "content": final_output})