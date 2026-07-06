"""Minimal Streamlit UI so /ask and /contradict can be tried without Postman.
Run the FastAPI server first (see README), then: streamlit run ui/streamlit_app.py"""
import streamlit as st
import requests

API_URL = "http://localhost:8000"

st.set_page_config(page_title="Potens Document Q&A", page_icon="📄")
st.title("📄 Document Q&A with Citations")
st.caption("Ask in any language. If the documents don't cover it, the system says so instead of guessing.")

tab_ask, tab_contradict = st.tabs(["Ask", "Contradict"])

with tab_ask:
    question = st.text_input("Ask a question about the ingested documents:")
    if st.button("Ask", key="ask_btn") and question:
        with st.spinner("Retrieving evidence and generating an answer..."):
            try:
                resp = requests.post(f"{API_URL}/ask", json={"question": question}, timeout=30)
            except requests.exceptions.ConnectionError:
                st.error("Can't reach the API. Is `uvicorn app.main:app` running on port 8000?")
                resp = None

        if resp is not None:
            if resp.ok:
                data = resp.json()
                if data.get("error"):
                    st.warning(f"LLM not configured yet: {data['error']}")
                    st.markdown("Showing the raw retrieved evidence instead:")
                elif data["answer"]:
                    st.markdown(f"**Answer:** {data['answer']}")

                st.caption(
                    f"Confidence: {data['confidence']}  |  "
                    f"Covered by documents: {data['covered_by_documents']}"
                )
                if data["citations"]:
                    st.markdown("**Citations:**")
                    for c in data["citations"]:
                        st.markdown(f"- `{c['source']}` (page {c['page']}): _{c['snippet']}..._")
            else:
                st.error(resp.text)

with tab_contradict:
    col1, col2 = st.columns(2)
    doc_a = col1.text_input("Document A filename", placeholder="Msme-idea.pdf")
    doc_b = col2.text_input("Document B filename", placeholder="AI-Social-Commerce-Complete-Project-Document.pdf")
    topic = st.text_input("Topic to compare", placeholder="risk scoring approach")
    if st.button("Check for contradiction") and doc_a and doc_b and topic:
        with st.spinner("Comparing..."):
            try:
                resp = requests.post(
                    f"{API_URL}/contradict",
                    json={"doc_a": doc_a, "doc_b": doc_b, "topic": topic},
                    timeout=30,
                )
            except requests.exceptions.ConnectionError:
                st.error("Can't reach the API. Is `uvicorn app.main:app` running on port 8000?")
                resp = None

        if resp is not None:
            if resp.ok:
                st.json(resp.json())
            else:
                st.error(resp.text)
