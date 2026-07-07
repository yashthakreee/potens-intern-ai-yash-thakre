"""Streamlit UI for /ask and /contradict.
Run the FastAPI server first (see README), then: streamlit run ui/streamlit_app.py"""
import re

import requests
import streamlit as st

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Potens Document Q&A", page_icon="📄", layout="wide")

st.markdown(
    """
    <style>
    .stChatMessage, .stExpander, div[data-testid="stMetric"] {
        border-radius: 10px;
    }
    .badge {
        display: inline-block;
        padding: 2px 10px;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-green { background: rgba(16, 185, 129, 0.15); color: #10b981; }
    .badge-red { background: rgba(239, 68, 68, 0.15); color: #ef4444; }
    .badge-gray { background: rgba(120, 120, 120, 0.15); color: #888; }
    .verdict-banner {
        padding: 14px 18px;
        border-radius: 10px;
        font-weight: 600;
        margin-bottom: 12px;
    }
    .verdict-conflict { background: rgba(239, 68, 68, 0.12); border-left: 4px solid #ef4444; }
    .verdict-noconflict { background: rgba(16, 185, 129, 0.12); border-left: 4px solid #10b981; }
    .verdict-unknown { background: rgba(120, 120, 120, 0.12); border-left: 4px solid #888; }
    </style>
    """,
    unsafe_allow_html=True,
)

if "history" not in st.session_state:
    st.session_state.history = []
if "known_docs" not in st.session_state:
    st.session_state.known_docs = []


def check_health() -> bool:
    try:
        return requests.get(f"{API_URL}/health", timeout=3).ok
    except requests.exceptions.RequestException:
        return False


def confidence_badge(confidence: float, covered: bool) -> str:
    css = "badge-green" if covered else "badge-red"
    label = "covered" if covered else "not covered"
    return f'<span class="badge {css}">{label} · confidence {confidence:.2f}</span>'


# ---------------------------------------------------------------- Sidebar --
with st.sidebar:
    st.header("📄 Potens RAG")
    st.caption("Document Q&A with citations, refusal, and contradiction checks.")

    healthy = check_health()
    st.markdown(
        f'<span class="badge {"badge-green" if healthy else "badge-red"}">'
        f'{"API online" if healthy else "API unreachable"}</span>',
        unsafe_allow_html=True,
    )
    if not healthy:
        st.error("Start the server with:\n\n`uvicorn app.main:app --reload --port 8000`")

    st.divider()
    st.subheader("Documents")
    if st.button("🔄 Run ingestion", use_container_width=True):
        with st.spinner("Reading /docs and rebuilding the index..."):
            try:
                r = requests.post(f"{API_URL}/ingest", timeout=60)
                if r.ok:
                    data = r.json()
                    st.session_state.known_docs = data.get("files", [])
                    st.success(
                        f"{data['documents_ingested']} documents → "
                        f"{data['chunks_stored']} chunks"
                    )
                else:
                    st.error(r.text)
            except requests.exceptions.RequestException as e:
                st.error(f"Can't reach the API: {e}")

    if st.session_state.known_docs:
        for f in st.session_state.known_docs:
            st.markdown(f"- `{f}`")
    else:
        st.caption("Run ingestion to see the loaded document list.")

    st.divider()
    if st.button("🗑️ Clear chat history", use_container_width=True):
        st.session_state.history = []
        st.rerun()

# ----------------------------------------------------------------- Tabs ---
tab_ask, tab_contradict = st.tabs(["💬 Ask", "⚖️ Contradict"])

# ------------------------------------------------------------- Ask tab ----
with tab_ask:
    st.caption("Ask in any language. If the documents don't cover it, the system says so instead of guessing.")

    examples = [
        "What is the WhatsApp Business Messaging Policy about?",
        "Who books the courier and handles logistics?",
        "What does the DPDP Act say about consent?",
    ]
    if not st.session_state.history:
        st.write("Try an example:")
        cols = st.columns(len(examples))
        for col, ex in zip(cols, examples):
            if col.button(ex, use_container_width=True):
                st.session_state.pending_question = ex
                st.rerun()

    for turn in st.session_state.history:
        with st.chat_message("user"):
            st.write(turn["question"])
        with st.chat_message("assistant"):
            data = turn["data"]
            if data.get("error"):
                st.warning(f"LLM not configured yet: {data['error']}")
                st.caption("Showing the raw retrieved evidence instead:")
            elif data.get("answer"):
                st.write(data["answer"])

            st.markdown(
                confidence_badge(data.get("confidence", 0.0), data.get("covered_by_documents", False)),
                unsafe_allow_html=True,
            )

            if data.get("citations"):
                with st.expander(f"📎 {len(data['citations'])} citation(s)"):
                    for i, c in enumerate(data["citations"], 1):
                        st.markdown(f"**[{i}] `{c['source']}`** — page {c['page']}")
                        st.caption(c["snippet"] + "...")

    question = st.chat_input("Ask a question about the ingested documents...")
    if "pending_question" in st.session_state:
        question = st.session_state.pop("pending_question")

    if question:
        with st.chat_message("user"):
            st.write(question)
        with st.chat_message("assistant"):
            with st.spinner("Retrieving evidence and generating an answer..."):
                try:
                    resp = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=30)
                    data = resp.json() if resp.ok else {"answer": None, "citations": [], "confidence": 0.0,
                                                          "covered_by_documents": False, "error": resp.text}
                except requests.exceptions.ConnectionError:
                    st.error("Can't reach the API. Is `uvicorn app.main:app` running on port 8000?")
                    data = None

            if data is not None:
                if data.get("error") and not data.get("citations"):
                    st.error(data["error"])
                else:
                    if data.get("error"):
                        st.warning(f"LLM not configured yet: {data['error']}")
                        st.caption("Showing the raw retrieved evidence instead:")
                    elif data.get("answer"):
                        st.write(data["answer"])
                    st.markdown(
                        confidence_badge(data.get("confidence", 0.0), data.get("covered_by_documents", False)),
                        unsafe_allow_html=True,
                    )
                    if data.get("citations"):
                        with st.expander(f"📎 {len(data['citations'])} citation(s)"):
                            for i, c in enumerate(data["citations"], 1):
                                st.markdown(f"**[{i}] `{c['source']}`** — page {c['page']}")
                                st.caption(c["snippet"] + "...")
                st.session_state.history.append({"question": question, "data": data})

# --------------------------------------------------------- Contradict tab -
with tab_contradict:
    st.caption("Retrieves evidence from two documents independently, then asks the model whether they conflict.")

    doc_options = st.session_state.known_docs
    col1, col2 = st.columns(2)
    with col1:
        if doc_options:
            doc_a = st.selectbox("Document A", doc_options, index=0 if doc_options else None)
        else:
            doc_a = st.text_input("Document A filename", placeholder="Msme-idea.pdf")
    with col2:
        if doc_options:
            default_b = 1 if len(doc_options) > 1 else 0
            doc_b = st.selectbox("Document B", doc_options, index=default_b)
        else:
            doc_b = st.text_input("Document B filename", placeholder="AI-Social-Commerce-Complete-Project-Document.pdf")

    topic = st.text_input("Topic to compare", placeholder="who books the courier and handles logistics")
    run = st.button("Check for contradiction", type="primary")

    if run and doc_a and doc_b and topic:
        with st.spinner("Comparing independent evidence from both documents..."):
            try:
                resp = requests.post(
                    f"{API_URL}/contradict",
                    json={"doc_a": doc_a, "doc_b": doc_b, "topic": topic},
                    timeout=30,
                )
                result = resp.json() if resp.ok else None
                if result is None:
                    st.error(resp.text)
            except requests.exceptions.ConnectionError:
                st.error("Can't reach the API. Is `uvicorn app.main:app` running on port 8000?")
                result = None

        if result is not None:
            raw = result.get("raw_model_output")
            if raw:
                m = re.search(r"CONFLICT:\s*(yes|no)", raw, re.IGNORECASE)
                reasoning_m = re.search(r"REASONING:\s*(.*)", raw, re.IGNORECASE | re.DOTALL)
                verdict = m.group(1).lower() if m else None
                reasoning = reasoning_m.group(1).strip() if reasoning_m else raw

                if verdict == "yes":
                    st.markdown(
                        '<div class="verdict-banner verdict-conflict">⚠️ Conflict detected</div>',
                        unsafe_allow_html=True,
                    )
                elif verdict == "no":
                    st.markdown(
                        '<div class="verdict-banner verdict-noconflict">✅ No conflict</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        '<div class="verdict-banner verdict-unknown">🤔 Could not parse a clear verdict</div>',
                        unsafe_allow_html=True,
                    )
                st.write(reasoning)
            elif result.get("conflict") is None:
                st.markdown(
                    '<div class="verdict-banner verdict-unknown">ℹ️ Not enough evidence to compare</div>',
                    unsafe_allow_html=True,
                )
                st.write(result.get("reasoning", ""))

            ev_col1, ev_col2 = st.columns(2)
            for col, label, key in [(ev_col1, doc_a, "doc_a_evidence"), (ev_col2, doc_b, "doc_b_evidence")]:
                with col:
                    st.markdown(f"**`{label}`**")
                    for h in result.get(key, []):
                        with st.container(border=True):
                            st.caption(f"page {h['page']} · confidence {h['confidence']:.3f}")
                            st.write(h["text"])
    elif run:
        st.warning("Fill in both documents and a topic first.")
