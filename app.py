# This file supports multi session RAG pipelines with separate ChromaDB collections per session.

import streamlit as st
from doc_parser import parse_documents
from rag import RAGPipeline

# 1. Page Configuration
st.set_page_config(
    page_title="Multi-Document RAG Workspace",
    page_icon="📚",
    layout="wide"
)

# 2. Active Model Definitions & Descriptions
MODEL_DESCRIPTIONS = {
    "openai/gpt-oss-120b": "🧠 Flagship Reasoning: High accuracy for complex document Q&A and deep analysis.",
    "openai/gpt-oss-20b": "⚡ Fast & Efficient: Sub-second responses for quick lookups and short summaries.",
    "qwen/qwen3.8-27b": "📊 Structured Data: Optimized for Excel spreadsheets, CSVs, and tabular data."
}

# 3. Initialize Global Multi-Session State
if "workspaces" not in st.session_state:
    st.session_state.workspaces = {}  # Structure: { "Workspace_Name": { "pipeline": RAGPipeline, "messages": [], "files": [] } }

if "current_workspace" not in st.session_state:
    st.session_state.current_workspace = None

# --- SIDEBAR: Model Selector & Workspace Management ---
with st.sidebar:
    st.title("⚙️ RAG Settings")
    
    # Model Dropdown Selector
    selected_model = st.selectbox(
        "Select LLM Architecture",
        options=list(MODEL_DESCRIPTIONS.keys()),
        index=0
    )
    st.caption(MODEL_DESCRIPTIONS[selected_model])
    st.markdown("---")
    
    st.title("📂 Workspace Manager")
    
    # Create Workspace Controls
    new_ws_name = st.text_input("New Workspace Name", placeholder="e.g., NLP PPTs or Resume Review")
    if st.button("➕ Create Workspace", use_container_width=True):
        clean_name = new_ws_name.strip()
        if clean_name:
            if clean_name not in st.session_state.workspaces:
                st.session_state.workspaces[clean_name] = {
                    "pipeline": RAGPipeline(collection_name=clean_name),
                    "messages": [],
                    "files": []
                }
                st.session_state.current_workspace = clean_name
                st.success(f"Workspace '{clean_name}' created!")
                st.rerun()
            else:
                st.warning("A workspace with this name already exists.")
        else:
            st.warning("Please enter a valid workspace name.")

    # Select Active Workspace
    if st.session_state.workspaces:
        st.markdown("---")
        workspace_names = list(st.session_state.workspaces.keys())
        
        # Ensure current workspace index is valid
        current_idx = workspace_names.index(st.session_state.current_workspace) if st.session_state.current_workspace in workspace_names else 0
        
        selected_ws = st.selectbox(
            "Active Workspace",
            options=workspace_names,
            index=current_idx
        )
        st.session_state.current_workspace = selected_ws

        # Active Workspace Info
        ws_info = st.session_state.workspaces[st.session_state.current_workspace]
        st.markdown(f"### 📄 Files in `{st.session_state.current_workspace}`")
        if ws_info["files"]:
            for file_name in ws_info["files"]:
                st.caption(f"• {file_name}")
        else:
            st.caption("No files uploaded to this workspace yet.")
            
        # Reset current workspace button
        if st.button("🗑️ Reset Active Workspace", use_container_width=True):
            st.session_state.workspaces[st.session_state.current_workspace] = {
                "pipeline": RAGPipeline(collection_name=st.session_state.current_workspace),
                "messages": [],
                "files": []
            }
            st.success(f"Reset workspace '{st.session_state.current_workspace}'!")
            st.rerun()

# --- MAIN INTERFACE ---
if not st.session_state.current_workspace:
    st.info("👈 Please create or select a Workspace from the sidebar to begin!")
    st.stop()

# Short reference handle for active workspace state
active_ws_name = st.session_state.current_workspace
ws_data = st.session_state.workspaces[active_ws_name]

st.header(f"💬 Chat Workspace: {active_ws_name}")

# --- DOCUMENT UPLOADER SECTION ---
with st.expander("📥 Upload & Index Documents for this Workspace", expanded=not bool(ws_data["files"])):
    uploaded_files = st.file_uploader(
        "Upload PDF, Word, PPTX, Excel, or CSV files",
        accept_multiple_files=True,
        type=["pdf", "docx", "pptx", "xlsx", "csv", "html"]
    )
    
    if st.button("⚡ Index Documents", use_container_width=True):
        if uploaded_files:
            with st.spinner(f"Parsing and indexing documents into `{active_ws_name}`..."):
                docs = parse_documents(uploaded_files)
                
                # Check for empty text extraction
                if not docs:
                    st.error("⚠️ No readable text could be extracted from the uploaded file(s). If it is a scanned PDF, OCR is required.")
                else:
                    # Append documents to workspace ChromaDB collection
                    ws_data["pipeline"].add_to_vectorstore(docs)
                    
                    # Update uploaded file registry
                    for f in uploaded_files:
                        if f.name not in ws_data["files"]:
                            ws_data["files"].append(f.name)
                            
                    st.success(f"Successfully indexed {len(docs)} text chunks into workspace '{active_ws_name}'!")
        else:
            st.warning("Please upload at least one document first.")

st.markdown("---")

# --- CHAT INTERFACE SECTION ---

# 1. Render Active Workspace Chat History
for msg in ws_data["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 2. Chat Input Handler
if user_query := st.chat_input(f"Ask anything about files in '{active_ws_name}'..."):
    # Append user question to workspace state and render UI
    ws_data["messages"].append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # Validate vectorstore initialization
    if ws_data["pipeline"].vectorstore is None:
        response_text = "⚠️ Please upload and index at least one document in this workspace before asking questions."
        citations = []
    else:
        with st.spinner("Searching workspace & generating response..."):
            try:
                response_text, citations = ws_data["pipeline"].query(
                    question=user_query,
                    model_name=selected_model
                )
            except Exception as e:
                response_text = f"An error occurred while generating the response: {str(e)}"
                citations = []

    # Format assistant response with source citations
    full_response = response_text
    if citations:
        full_response += "\n\n**Sources:**\n" + "\n".join([f"- 📄 `{c}`" for c in citations])

    # Append assistant answer to workspace state and render UI
    ws_data["messages"].append({"role": "assistant", "content": full_response})
    with st.chat_message("assistant"):
        st.markdown(full_response)