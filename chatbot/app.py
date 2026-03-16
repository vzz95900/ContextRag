"""Streamlit chatbot frontend for the Context-Aware RAG Engine."""

import streamlit as st
import requests
import time

# ── Config ──────────────────────────────────────────────────

API_BASE = "http://localhost:8000"

# ── Page Setup ──────────────────────────────────────────────

st.set_page_config(
    page_title="ContextAware — RAG Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────

st.markdown("""
<style>
    /* ---------- Global ---------- */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    .stApp {
        font-family: 'Inter', sans-serif;
    }
    
    /* Hide default Streamlit chrome */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    div[data-testid="stDecoration"] {display: none;}
    
    /* Hide deploy button but keep toolbar for sidebar toggle */
    div[data-testid="stToolbar"] .stDeployButton {display: none;}
    div[data-testid="stToolbar"] button[kind="header"] {display: none;}
    
    /* ---------- Sidebar ---------- */
    section[data-testid="stSidebar"] {
        min-width: 320px !important;
        width: 320px !important;
        background: linear-gradient(180deg, #0f0f1a 0%, #1a1a2e 100%);
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    
    /* Sidebar expand button when collapsed */
    div[data-testid="stSidebarCollapsedControl"] {
        visibility: visible !important;
        display: flex !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stMarkdown li {
        color: #c8c8d4;
    }
    
    /* ---------- Branding ---------- */
    .brand-header {
        text-align: center;
        padding: 1rem 0 0.5rem 0;
    }
    .brand-header h1 {
        font-size: 1.5rem;
        font-weight: 700;
        background: linear-gradient(135deg, #6366f1, #a78bfa, #818cf8);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        letter-spacing: -0.02em;
    }
    .brand-header p {
        color: #64748b;
        font-size: 0.75rem;
        margin-top: 0.2rem;
    }
    
    /* ---------- Upload area ---------- */
    .upload-zone {
        border: 2px dashed rgba(99,102,241,0.3);
        border-radius: 12px;
        padding: 1.2rem;
        text-align: center;
        background: rgba(99,102,241,0.04);
        transition: all 0.2s;
        margin-bottom: 0.5rem;
    }
    .upload-zone:hover {
        border-color: rgba(99,102,241,0.6);
        background: rgba(99,102,241,0.08);
    }
    
    /* ---------- Document card ---------- */
    .doc-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 10px;
        padding: 0.75rem 1rem;
        margin-bottom: 0.5rem;
        transition: all 0.2s;
    }
    .doc-card:hover {
        background: rgba(255,255,255,0.07);
        border-color: rgba(99,102,241,0.3);
    }
    .doc-name {
        font-weight: 600;
        font-size: 0.85rem;
        color: #e2e8f0;
        margin: 0;
        overflow: hidden;
        text-overflow: ellipsis;
        white-space: nowrap;
    }
    .doc-meta {
        font-size: 0.7rem;
        color: #64748b;
        margin: 0.15rem 0 0 0;
    }
    
    /* ---------- Stats pills ---------- */
    .stats-row {
        display: flex;
        gap: 0.5rem;
        margin: 0.5rem 0 1rem 0;
    }
    .stat-pill {
        background: rgba(99,102,241,0.1);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 20px;
        padding: 0.3rem 0.75rem;
        font-size: 0.72rem;
        color: #a78bfa;
        font-weight: 500;
    }
    
    /* ---------- Chat area ---------- */
    .chat-header {
        text-align: center;
        padding: 2rem 0 1rem 0;
    }
    .chat-header h2 {
        font-size: 1.8rem;
        font-weight: 700;
        color: #f1f5f9;
        margin: 0;
    }
    .chat-header p {
        color: #64748b;
        font-size: 0.9rem;
        margin-top: 0.4rem;
    }
    
    /* ---------- Context selector ---------- */
    .ctx-selector {
        display: flex;
        align-items: center;
        gap: 0.5rem;
        padding: 0.5rem 0.75rem;
        background: rgba(99,102,241,0.06);
        border: 1px solid rgba(99,102,241,0.15);
        border-radius: 10px;
        margin-bottom: 0.75rem;
    }
    .ctx-selector .ctx-label {
        font-size: 0.75rem;
        font-weight: 600;
        color: #818cf8;
        white-space: nowrap;
    }
    .ctx-selector-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: rgba(99,102,241,0.12);
        border: 1px solid rgba(99,102,241,0.25);
        border-radius: 6px;
        padding: 0.2rem 0.5rem;
        font-size: 0.72rem;
        color: #c4b5fd;
        font-weight: 500;
    }
    
    /* Welcome cards */
    .welcome-grid {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 0.75rem;
        max-width: 600px;
        margin: 1.5rem auto;
    }
    .welcome-card {
        background: rgba(255,255,255,0.03);
        border: 1px solid rgba(255,255,255,0.08);
        border-radius: 12px;
        padding: 1rem;
        cursor: pointer;
        transition: all 0.2s;
    }
    .welcome-card:hover {
        background: rgba(99,102,241,0.08);
        border-color: rgba(99,102,241,0.3);
        transform: translateY(-1px);
    }
    .welcome-card .icon {
        font-size: 1.3rem;
        margin-bottom: 0.4rem;
    }
    .welcome-card .title {
        font-size: 0.82rem;
        font-weight: 600;
        color: #e2e8f0;
        margin: 0;
    }
    .welcome-card .desc {
        font-size: 0.72rem;
        color: #64748b;
        margin: 0.2rem 0 0 0;
    }
    
    /* ---------- Source citation ---------- */
    .source-chip {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        background: rgba(99,102,241,0.1);
        border: 1px solid rgba(99,102,241,0.2);
        border-radius: 8px;
        padding: 0.4rem 0.7rem;
        margin: 0.2rem 0.3rem 0.2rem 0;
        font-size: 0.75rem;
        color: #c4b5fd;
        transition: all 0.15s;
    }
    .source-chip:hover {
        background: rgba(99,102,241,0.2);
    }
    .source-detail {
        background: rgba(255,255,255,0.03);
        border-left: 3px solid #6366f1;
        border-radius: 0 8px 8px 0;
        padding: 0.6rem 0.8rem;
        margin: 0.4rem 0;
        font-size: 0.78rem;
        color: #94a3b8;
        line-height: 1.5;
    }
    
    /* ---------- Latency badge ---------- */
    .latency-badge {
        display: inline-flex;
        align-items: center;
        gap: 0.3rem;
        background: rgba(34,197,94,0.1);
        border: 1px solid rgba(34,197,94,0.2);
        border-radius: 20px;
        padding: 0.2rem 0.6rem;
        font-size: 0.7rem;
        color: #4ade80;
        margin-top: 0.3rem;
    }
    
    /* ---------- Chat messages ---------- */
    .stChatMessage {
        border-radius: 12px !important;
        margin-bottom: 0.5rem !important;
    }
    
    /* ---------- Section divider ---------- */
    .section-label {
        font-size: 0.65rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        color: #475569;
        font-weight: 600;
        margin: 1rem 0 0.5rem 0;
        padding: 0 0.2rem;
    }
    
    /* ---------- Button overrides ---------- */
    .stButton > button {
        border-radius: 8px;
        font-weight: 500;
        font-size: 0.82rem;
        transition: all 0.2s;
    }
    
    div[data-testid="stChatInput"] {
        border-radius: 12px;
    }
    div[data-testid="stChatInput"] textarea {
        font-size: 0.92rem;
    }
</style>
""", unsafe_allow_html=True)

# ── Sidebar ─────────────────────────────────────────────────

with st.sidebar:
    # Brand
    st.markdown("""
    <div class="brand-header">
        <h1>🧠 ContextAware</h1>
        <p>AI-powered document intelligence</p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-label">📤 Upload Document</div>', unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Drop a PDF here",
        type=["pdf"],
        label_visibility="collapsed",
        help="Supports text-based and scanned PDFs",
    )

    if uploaded_file is not None:
        file_size_mb = len(uploaded_file.getvalue()) / (1024 * 1024)
        st.markdown(
            f'<div class="doc-card">'
            f'<p class="doc-name">📎 {uploaded_file.name}</p>'
            f'<p class="doc-meta">{file_size_mb:.1f} MB</p>'
            f'</div>',
            unsafe_allow_html=True,
        )
        if st.button("⚡ Index Document", use_container_width=True, type="primary"):
            progress = st.progress(0, text="Parsing PDF...")
            try:
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")}
                progress.progress(20, text="Chunking & embedding...")
                resp = requests.post(f"{API_BASE}/api/upload", files=files, timeout=180)
                progress.progress(80, text="Storing vectors...")
                time.sleep(0.3)
                if resp.status_code == 200:
                    data = resp.json()
                    progress.progress(100, text="Done!")
                    time.sleep(0.3)
                    progress.empty()
                    st.success(
                        f"✅ **{data['filename']}** indexed successfully\n\n"
                        f"📄 {data['page_count']} pages  ·  🧩 {data['chunk_count']} chunks"
                    )
                    time.sleep(1.5)
                    st.rerun()
                else:
                    progress.empty()
                    detail = resp.json().get("detail", resp.text)
                    st.error(f"❌ {detail}")
            except requests.ConnectionError:
                progress.empty()
                st.error("🔌 Cannot connect to API server")
            except requests.Timeout:
                progress.empty()
                st.error("⏱️ Request timed out — PDF may be too large")

    # ── Indexed Documents ───────────────────────────────────
    st.markdown('<div class="section-label">📚 Knowledge Base</div>', unsafe_allow_html=True)

    try:
        resp = requests.get(f"{API_BASE}/api/documents", timeout=30)
        if resp.status_code == 200:
            docs = resp.json().get("documents", [])
            # Cache for context selector (avoids double API call)
            st.session_state["_cached_docs"] = docs
            if docs:
                total_chunks = sum(d["chunk_count"] for d in docs)
                total_pages = sum(d["page_count"] for d in docs)
                st.markdown(
                    f'<div class="stats-row">'
                    f'<span class="stat-pill">📄 {len(docs)} docs</span>'
                    f'<span class="stat-pill">📃 {total_pages} pages</span>'
                    f'<span class="stat-pill">🧩 {total_chunks} chunks</span>'
                    f'</div>',
                    unsafe_allow_html=True,
                )
                for doc in docs:
                    col1, col2 = st.columns([5, 1])
                    with col1:
                        st.markdown(
                            f'<div class="doc-card">'
                            f'<p class="doc-name">{doc["filename"]}</p>'
                            f'<p class="doc-meta">{doc["page_count"]} pages · {doc["chunk_count"]} chunks</p>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                    with col2:
                        if st.button("🗑️", key=f"del_{doc['doc_id']}", help=f"Remove {doc['filename']}"):
                            with st.spinner(""):
                                requests.delete(f"{API_BASE}/api/documents/{doc['doc_id']}", timeout=30)
                            st.rerun()
            else:
                st.markdown(
                    '<div style="text-align:center; padding: 1.5rem 0; color: #475569;">'
                    '<div style="font-size: 2rem; margin-bottom: 0.5rem;">📭</div>'
                    '<div style="font-size: 0.82rem;">No documents yet</div>'
                    '<div style="font-size: 0.72rem; color: #334155;">Upload a PDF to get started</div>'
                    '</div>',
                    unsafe_allow_html=True,
                )
        else:
            st.warning(f"⚠️ API returned status {resp.status_code}")
    except requests.ConnectionError:
        st.warning("🔌 API server not reachable")
    except requests.Timeout:
        st.warning("⏱️ API request timed out")
    except Exception as e:
        st.warning(f"⚠️ Could not load documents: {e}")

    # ── Footer ──────────────────────────────────────────────
    st.markdown("---")
    st.markdown(
        '<div style="text-align:center; font-size: 0.68rem; color: #334155; padding: 0.5rem 0;">'
        'Built with Hugging Face · Gemini · ChromaDB · FastAPI'
        '</div>',
        unsafe_allow_html=True,
    )

# ── Chat Interface ──────────────────────────────────────────

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "session_id" not in st.session_state:
    st.session_state.session_id = None

# ── Document Context Selector (reuse sidebar data) ──────────
_ctx_docs = st.session_state.get("_cached_docs", [])

if _ctx_docs:
    doc_options = ["All Documents"] + [d["filename"] for d in _ctx_docs]
    selected_doc = st.selectbox(
        "🎯 Query context",
        options=doc_options,
        index=0,
        help="Choose which document to search. 'All Documents' searches everything.",
    )
    # Resolve doc_id for the selected document
    if selected_doc != "All Documents":
        _selected_doc_id = next((d["doc_id"] for d in _ctx_docs if d["filename"] == selected_doc), None)
        st.markdown(
            f'<div class="ctx-selector">'
            f'<span class="ctx-label">Searching in:</span>'
            f'<span class="ctx-selector-chip">📄 {selected_doc}</span>'
            f'</div>',
            unsafe_allow_html=True,
        )
    else:
        _selected_doc_id = None
        st.markdown(
            '<div class="ctx-selector">'
            '<span class="ctx-label">Searching in:</span>'
            '<span class="ctx-selector-chip">📚 All indexed documents</span>'
            '</div>',
            unsafe_allow_html=True,
        )
else:
    _selected_doc_id = None

# Welcome state (no messages yet)
if not st.session_state.messages:
    st.markdown("""
    <div class="chat-header">
        <h2>What can I help you find?</h2>
        <p>Ask questions about your uploaded documents — every answer is grounded in your sources.</p>
    </div>
    <div class="welcome-grid">
        <div class="welcome-card">
            <div class="icon">📋</div>
            <p class="title">Summarize a document</p>
            <p class="desc">Get a concise overview of key points</p>
        </div>
        <div class="welcome-card">
            <div class="icon">🔍</div>
            <p class="title">Find specific details</p>
            <p class="desc">Search for facts, figures, or quotes</p>
        </div>
        <div class="welcome-card">
            <div class="icon">🔗</div>
            <p class="title">Compare information</p>
            <p class="desc">Cross-reference across documents</p>
        </div>
        <div class="welcome-card">
            <div class="icon">💡</div>
            <p class="title">Explain concepts</p>
            <p class="desc">Break down complex topics simply</p>
        </div>
    </div>
    """, unsafe_allow_html=True)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="🧑‍💻" if message["role"] == "user" else "🧠"):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📎 View Sources", expanded=False):
                for i, src in enumerate(message["sources"]):
                    st.markdown(
                        f'<div class="source-chip">📄 {src["document"]} · Page {src["page"]} · Score: {src["score"]}</div>',
                        unsafe_allow_html=True,
                    )
                    st.markdown(
                        f'<div class="source-detail">{src["text"][:250]}...</div>',
                        unsafe_allow_html=True,
                    )
        if message.get("latency"):
            st.markdown(
                f'<div class="latency-badge">⚡ {message["latency"]:.0f}ms · {message["model"]}</div>',
                unsafe_allow_html=True,
            )

# Chat input
if prompt := st.chat_input("Ask anything about your documents..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user", avatar="🧑‍💻"):
        st.markdown(prompt)

    # Get RAG response
    with st.chat_message("assistant", avatar="🧠"):
        with st.spinner("🔍 Searching & reasoning..."):
            try:
                payload = {
                    "query": prompt,
                    "session_id": st.session_state.session_id,
                }
                if _selected_doc_id:
                    payload["filters"] = {"doc_id": _selected_doc_id}
                resp = requests.post(f"{API_BASE}/api/chat", json=payload, timeout=180)

                if resp.status_code == 200:
                    data = resp.json()
                    answer = data["answer"]
                    sources = data.get("sources", [])
                    latency = data.get("latency_ms", 0)
                    model = data.get("model", "")
                    st.session_state.session_id = data.get("session_id")

                    st.markdown(answer)

                    if sources:
                        with st.expander("📎 View Sources", expanded=False):
                            for i, src in enumerate(sources):
                                st.markdown(
                                    f'<div class="source-chip">📄 {src["document"]} · Page {src["page"]} · Score: {src["score"]}</div>',
                                    unsafe_allow_html=True,
                                )
                                st.markdown(
                                    f'<div class="source-detail">{src["text"][:250]}...</div>',
                                    unsafe_allow_html=True,
                                )

                    st.markdown(
                        f'<div class="latency-badge">⚡ {latency:.0f}ms · {model}</div>',
                        unsafe_allow_html=True,
                    )

                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources,
                        "latency": latency,
                        "model": model,
                    })

                else:
                    error_msg = resp.json().get("detail", "Unknown error")
                    if resp.status_code == 429:
                        st.warning(
                            "⏳ **Rate limit reached** — The API returned a 429 error.\n\n"
                            "**What to do:**\n"
                            "- Wait a moment and try again\n"
                            "- If using Hugging Face free tier, try waiting 1 hour to reset quota"
                        )
                    else:
                        st.error(f"❌ {error_msg}")

            except requests.ConnectionError:
                st.error(
                    "🔌 Cannot connect to the API server. "
                    "Make sure it's running with: `uvicorn app.main:app --reload`"
                )
            except requests.Timeout:
                st.error("⏱️ Request timed out. The server may be overloaded — try again in a moment.")
