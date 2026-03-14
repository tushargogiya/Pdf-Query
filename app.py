import os
import streamlit as st

# Inject Streamlit Cloud secrets into os.environ before loading backend
for _k, _v in st.secrets.items():
    if _k not in os.environ:
        os.environ[_k] = str(_v)

from pdf_backend import PdfQAEngine

# ========================= Page Config =========================
st.set_page_config(
    page_title="PDF Query AI",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ========================= Custom CSS =========================
st.markdown("""
<style>
    .main .block-container { padding-top: 1.5rem; max-width: 880px; }

    .source-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 8px;
        padding: 10px 14px;
        margin: 6px 0;
        font-size: 0.82rem;
        line-height: 1.5;
    }
    .page-badge {
        display: inline-block;
        background: #1e3a5f;
        color: #60a5fa;
        border: 1px solid #1d4ed8;
        border-radius: 999px;
        padding: 1px 9px;
        font-size: 0.68rem;
        font-weight: 700;
        margin-right: 6px;
        vertical-align: middle;
    }
    .doc-pill {
        display: inline-block;
        background: rgba(255,255,255,0.07);
        border: 1px solid rgba(255,255,255,0.14);
        border-radius: 6px;
        padding: 4px 12px;
        font-size: 0.75rem;
        margin: 3px 2px;
    }
    .step-card {
        background: rgba(255,255,255,0.04);
        border: 1px solid rgba(255,255,255,0.1);
        border-radius: 10px;
        padding: 18px;
        text-align: center;
        height: 100%;
    }
    .tech-badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.68rem;
        font-weight: 700;
        margin: 2px;
    }
    .b-blue   { background:#1e3a5f; color:#60a5fa; border:1px solid #1d4ed8; }
    .b-green  { background:#14532d; color:#4ade80; border:1px solid #166534; }
    .b-purple { background:#2e1065; color:#c084fc; border:1px solid #7c3aed; }
    .b-orange { background:#431407; color:#fb923c; border:1px solid #c2410c; }
    .footer-text { font-size:0.65rem; color:rgba(255,255,255,0.3); text-align:center; line-height:1.7; }
</style>
""", unsafe_allow_html=True)

# ========================= Session State =========================
if "engine" not in st.session_state:
    st.session_state.engine = PdfQAEngine()
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "doc_info" not in st.session_state:
    st.session_state.doc_info = None

# ========================= Sidebar =========================
with st.sidebar:
    st.markdown("## 📄 PDF Query AI")
    st.caption("RAG · Azure GPT · Semantic Search")

    st.divider()

    st.markdown("**Pipeline**")
    st.markdown("""
    <div style="margin-bottom:12px; line-height:2.2;">
        <span class="tech-badge b-blue">📥 PDFPlumber</span>
        <span class="tech-badge b-green">🧩 Recursive Chunking</span>
        <span class="tech-badge b-purple">🔢 MiniLM Embeddings</span>
        <span class="tech-badge b-orange">🔍 MMR Retrieval</span>
        <span class="tech-badge b-blue">🤖 Azure GPT</span>
    </div>
    """, unsafe_allow_html=True)

    st.divider()

    st.markdown("**Upload PDF**")
    uploaded_file = st.file_uploader("", type="pdf", label_visibility="collapsed")

    if uploaded_file:
        if st.button("⚡ Process PDF", type="primary", use_container_width=True):
            with st.spinner("Parsing, chunking & indexing…"):
                try:
                    info = st.session_state.engine.load_pdf(
                        uploaded_file.read(), uploaded_file.name
                    )
                    st.session_state.doc_info = info
                    st.session_state.chat_history = []
                    st.success(f"Ready! {info['pages']} pages indexed.")
                except Exception as e:
                    st.error(f"Error: {e}")

    if st.session_state.doc_info:
        st.divider()
        info = st.session_state.doc_info
        st.markdown("**Document**")
        name = info["filename"]
        short_name = name[:22] + "…" if len(name) > 22 else name
        st.markdown(f"""
        <div>
            <span class="doc-pill">📄 {short_name}</span>
            <span class="doc-pill">📃 {info['pages']} pages</span>
            <span class="doc-pill">🧩 {info['chunks']} chunks</span>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("")
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.chat_history = []
            st.rerun()

    st.divider()
    st.markdown("""
    <div class="footer-text">
        FAISS · all-MiniLM-L6-v2<br>
        RecursiveTextSplitter · MMR Retrieval<br>
        Azure OpenAI GPT
    </div>
    """, unsafe_allow_html=True)

# ========================= Main Area =========================
st.markdown("## 💬 PDF Query AI")
st.markdown("Upload a PDF and ask anything — answers are grounded in your document with page-level source citations.")
st.markdown("---")

if not st.session_state.doc_info:
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="step-card">
            <h3>① Upload</h3>
            <p>Upload any PDF from the sidebar — research papers, reports, manuals, contracts</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="step-card">
            <h3>② Index</h3>
            <p>Click "Process PDF" — the doc gets chunked, embedded with MiniLM & stored in FAISS</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="step-card">
            <h3>③ Ask</h3>
            <p>Ask in natural language — Azure GPT answers with source page citations from the doc</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### How It Works")
    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("""
**RAG Pipeline:**
- 📥 **PDF Parsing** — PDFPlumber with layout awareness
- 🧩 **Smart Chunking** — RecursiveCharacterTextSplitter (600 tokens, 120 overlap)
- 🔢 **Embeddings** — `all-MiniLM-L6-v2` (semantic, CPU-friendly)
- 🗄️ **Vector Store** — FAISS in-memory for fast similarity search
        """)
    with col_b:
        st.markdown("""
**Retrieval & Generation:**
- 🔍 **MMR Retrieval** — Max Marginal Relevance (diverse + relevant chunks)
- 🤖 **Azure GPT** — Grounded answers from retrieved context only
- 📚 **Source Citations** — Every answer shows exact page & text snippet
- 💬 **Chat History** — Full multi-turn conversation per session
        """)

else:
    # Chat interface — render history
    for item in st.session_state.chat_history:
        with st.chat_message("user"):
            st.markdown(item["question"])
        with st.chat_message("assistant"):
            st.markdown(item["answer"])
            if item.get("sources"):
                with st.expander(f"📚 View {len(item['sources'])} source(s)"):
                    for src in item["sources"]:
                        st.markdown(f"""
                        <div class="source-card">
                            <span class="page-badge">Page {src['page']}</span>
                            {src['snippet']}…
                        </div>
                        """, unsafe_allow_html=True)

    question = st.chat_input("Ask a question about your PDF…")

    if question:
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching document & generating answer…"):
                try:
                    result = st.session_state.engine.answer(question)
                    answer = result["answer"]
                    sources = result["sources"]

                    st.markdown(answer)

                    if sources:
                        with st.expander(f"📚 {len(sources)} source(s) from document"):
                            for src in sources:
                                st.markdown(f"""
                                <div class="source-card">
                                    <span class="page-badge">Page {src['page']}</span>
                                    {src['snippet']}…
                                </div>
                                """, unsafe_allow_html=True)

                    st.session_state.chat_history.append({
                        "question": question,
                        "answer": answer,
                        "sources": sources,
                    })

                except Exception as e:
                    st.error(f"Error: {e}")
