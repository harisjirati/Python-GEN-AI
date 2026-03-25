import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(
    page_title="PDF RAG Assistant",
    page_icon="📄",
    layout="centered"
)

# ─── Session state ────────────────────────────────────────────────────────────

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "indexed_files" not in st.session_state:
    st.session_state.indexed_files = []

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.header("📂 Upload PDFs")
    st.caption("Select one or multiple PDFs")

    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        label_visibility="collapsed"
    )

    if uploaded_files:
        st.write(f"**{len(uploaded_files)} file(s) selected:**")
        for f in uploaded_files:
            st.write(f"• {f.name}")

    if uploaded_files and st.button("Index PDFs", type="primary", use_container_width=True):
        with st.spinner(f"Indexing {len(uploaded_files)} file(s)..."):
            files_payload = [
                ("files", (f.name, f.getvalue(), "application/pdf"))
                for f in uploaded_files
            ]
            try:
                response = requests.post(
                    f"{API_URL}/upload",
                    files=files_payload,
                    timeout=120
                )
                if response.status_code == 200:
                    data = response.json()
                    st.success(f"✅ {data['message']}")
                    st.session_state.indexed_files = [f.name for f in uploaded_files]
                    st.session_state.chat_history = []
                else:
                    st.error(f"❌ {response.json().get('detail', 'Upload failed')}")
            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot reach API. Run: `python main.py`")

    if st.session_state.indexed_files:
        st.divider()
        st.markdown("**Currently indexed:**")
        for fname in st.session_state.indexed_files:
            st.write(f"📄 {fname}")

    st.divider()

    if st.button("Check API Health", use_container_width=True):
        try:
            r = requests.get(f"{API_URL}/health", timeout=5)
            h = r.json()
            col1, col2 = st.columns(2)
            col1.metric("Chunks", h.get("chroma_chunks", 0))
            col2.metric("Ollama", "✅" if h.get("ollama_connected") else "❌")
            st.caption(f"Model: `{h.get('model')}`")
        except Exception:
            st.error("❌ API unreachable")

    st.divider()
    st.caption("💡 Tip: Start questions with **'explain'** or **'summarize'** for detailed answers.")

# ─── Main chat ────────────────────────────────────────────────────────────────

st.title("📄 Intelligent Document Processing (IDP)")
st.caption("Ask anything about your documents — powered by Llama")

if not st.session_state.indexed_files:
    st.info("👈 Upload and index one or more PDFs from the sidebar to get started.")
def _render_answer(data: dict):
    answer = data.get("answer", "Not found")
    sources = data.get("sources", [])
    detailed = data.get("detailed", False)

    # Answer text
    st.markdown(answer)

    # Detail mode badge
    if detailed:
        st.caption("📋 Detailed response")

    # Source citations
    if sources:
        source_text = " · ".join([f"📄 `{s}`" for s in sources])
        st.caption(f"Sources: {source_text}")

    # Full JSON
    with st.expander("View full response"):
        st.json(data)

# Render chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        if msg["role"] == "assistant" and isinstance(msg["content"], dict):
            _render_answer(msg["content"])
        else:
            st.markdown(msg["content"])




# Query input
if question := st.chat_input(
    "Ask anything about your PDFs...",
    disabled=not st.session_state.indexed_files
):
    st.session_state.chat_history.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                r = requests.post(
                    f"{API_URL}/query",
                    params={"question": question},
                    timeout=120
                )

                if r.status_code == 200:
                    data = r.json()
                    _render_answer(data)
                    st.session_state.chat_history.append({
                        "role": "assistant",
                        "content": data
                    })
                else:
                    err = r.json().get("detail", "Unknown error")
                    st.error(f"❌ {err}")

            except requests.exceptions.ConnectionError:
                st.error("❌ Cannot reach FastAPI. Run: `python main.py`")