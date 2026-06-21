import streamlit as st
from Model.llm import LLMManager
from RagEngine.rag_engine import RAGEngine
from Analysis.analyzer import RequirementAnalyzer
from UI.rag_tab import render_rag_tab

from UI.analysis_tab import render_analysis_tab
from UI.chat_tab import render_chat_tab

st.set_page_config(page_title="NVIDIA Mission Critical Assistant", layout="wide", initial_sidebar_state="collapsed")

# Initialization Management
if "llm" not in st.session_state:
    st.session_state.llm = LLMManager()
if "rag" not in st.session_state:
    st.session_state.rag = RAGEngine(st.session_state.llm)
    st.session_state.rag.load_trained_engine()
if "analyzer" not in st.session_state:
    st.session_state.analyzer = RequirementAnalyzer(st.session_state.llm)
if "messages" not in st.session_state:
    st.session_state.messages = []
if "last_action" not in st.session_state:
    st.session_state.last_action = None

@st.dialog("📚 Database Chunk Explorer", width="large")
def view_chunks_dialog(collection_name: str):
    st.write(f"📂 **Viewing Collection:** `{collection_name}`")
    
    with st.spinner("Loading chunks..."):
        try:
            col_chunks = st.session_state.rag.get_all_chunks(collection_name, limit=500)
        except Exception as e:
            st.error(f"Failed to fetch chunks: {e}")
            return
            
    st.write(f"📊 **Total Chunks Ingested:** `{len(col_chunks)}`")
    
    if not col_chunks:
        st.info("No chunks found in this collection.")
        return
        
    # Group chunks by page or source
    pages = {}
    for chunk in col_chunks:
        payload = chunk.get("payload", {})
        meta = payload.get("metadata", {})
        page_num = meta.get("page", "General")
        if page_num not in pages:
            pages[page_num] = []
        pages[page_num].append(chunk)
    
    # Display sub-expanders for each page
    for page_num in sorted(pages.keys(), key=lambda x: (isinstance(x, int), x)):
        page_label = f"Page {page_num}" if isinstance(page_num, int) else str(page_num)
        page_chunks = pages[page_num]
        
        with st.expander(f"📄 {page_label} ({len(page_chunks)} chunks)", expanded=False):
            for idx, chunk in enumerate(page_chunks, 1):
                cid = chunk.get("id", "N/A")
                payload = chunk.get("payload", {})
                t = payload.get("title", "Untitled")
                txt = payload.get("text", "")
                meta = payload.get("metadata", {})
                itype = meta.get("item_type", "N/A")
                iid = meta.get("item_id") or "N/A"
                
                st.markdown(f"""
                <div style="background: rgba(30, 41, 59, 0.45); border: 1px solid rgba(255,255,255,0.08); padding: 12px; border-radius: 8px; margin-bottom: 10px;">
                    <div style="font-size: 0.78rem; color: #94a3b8; margin-bottom: 6px;">
                        <strong>Chunk #{idx}</strong> | ID: <code>{cid}</code> | Type: <code>{itype}</code> | Ref: <code>{iid}</code>
                    </div>
                    <details style="cursor: pointer;">
                        <summary style="font-size: 0.95rem; font-weight: 600; color: #60a5fa;">{t}</summary>
                        <p style="font-size: 0.88rem; margin-top: 8px; margin-bottom: 0; color: #cbd5e1; line-height: 1.5; white-space: pre-wrap;">{txt}</p>
                    </details>
                </div>
                """, unsafe_allow_html=True)

# System Configuration Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    st.markdown("Customize system parameters and API fault-tolerance.")
    
    llm_retry_limit = st.slider(
        "LLM Retry Limit",
        min_value=0,
        max_value=10,
        value=3,
        step=1,
        help="Number of retries when an LLM API request fails (e.g. transient connection error, rate limits)."
    )
    st.session_state.llm_retries = llm_retry_limit
    if "llm" in st.session_state:
        st.session_state.llm.retries = llm_retry_limit
    
    st.markdown("---")
    st.subheader("⚡ Processing Mode")
    st.session_state.batch_mode_enabled = st.toggle("Batch Processing", value=False)
    if st.session_state.batch_mode_enabled:
        st.session_state.batch_size = st.number_input("Batch Size", min_value=1, value=10, step=1)
    else:
        st.session_state.batch_size = 5

    st.markdown("---")
    st.subheader("🤖 Active LLM Model")
    st.info(f"**Model:**\n`{st.session_state.llm.model_name}`")
    st.caption("Powered by NVIDIA NIM Core engine.")

    st.markdown("---")
    st.subheader("📚 Collections & Documents")
    collections = st.session_state.rag.get_collections()
    
    if collections:
        selected_cols = st.multiselect(
            "🎯 Active Collections for Analysis",
            options=collections,
            default=[],
            help="Select which collections to use for requirement rules. Leave empty to use all."
        )
        st.session_state.target_rag_collections = selected_cols if selected_cols else None
        
        for col in collections:
            with st.expander(f"📁 {col}"):
                st.markdown(f"Inspect the chunks ingested for **{col}**.")
                if st.button("🔍 View Chunks", key=f"btn_view_{col}", use_container_width=True):
                    view_chunks_dialog(col)
    else:
        st.info("No collections found.")

# Application Layout Tabs


tab_rag, tab_analysis, tab_chat, tab_sandbox = st.tabs([
    "📂 RAG Knowledge Engine", 
    "📈 Requirements Quality Analyst", 
    "💬 Nemotron Core Chat",
    "🧪 Prompt Sandbox"
])

with tab_rag:
    render_rag_tab()

with tab_analysis:
    render_analysis_tab()

with tab_chat:
    render_chat_tab()

with tab_sandbox:
    st.header("🧪 Prompt Sandbox")
    st.markdown("Test custom system prompts for the LLM without modifying the codebase. You can configure separate overrides for different execution paths.")
    
    modes = [
        ("Analysis", "analysis"),
        ("Process", "process"),
        ("Batch Analysis", "batch_analysis"),
        ("Batch Process", "batch_process")
    ]
    
    for display_name, mode_key in modes:
        with st.expander(f"🛠️ {display_name} Prompt Override", expanded=False):
            use_custom = st.checkbox(f"Use Custom Prompt for {display_name}", key=f"use_custom_prompt_{mode_key}")
            if use_custom:
                st.warning(f"⚠️ Using Custom Prompt Override for {display_name}.")
                
            st.text_area(
                "Custom System Prompt",
                height=250,
                key=f"custom_prompt_{mode_key}",
                help=f"Enter the custom system prompt for {display_name}. The model will use this instead of the default prompts when the checkbox is enabled.",
                disabled=not use_custom
            )