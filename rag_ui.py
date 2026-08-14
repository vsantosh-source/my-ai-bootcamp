"""Minimal Streamlit UI for a live RAG API (POST /ingest, POST /ask).

Run:
  streamlit run rag_ui.py

Point it at your live Render URL by setting API_BASE_URL before launching:
  API_BASE_URL=https://your-service.onrender.com streamlit run rag_ui.py

or just type the URL into the sidebar once the page is open. This UI only
calls the API - it never talks to OpenAI or Pinecone directly, so the API
stays the single source of truth for retrieval and generation.
"""

import os

import httpx
import streamlit as st

DEFAULT_API_BASE_URL = os.environ.get("API_BASE_URL", "http://127.0.0.1:8000")


def call_api(
    method: str, base_url: str, path: str, payload: dict | None = None
) -> tuple[int, dict | str]:
    url = f"{base_url.rstrip('/')}{path}"
    try:
        if method == "POST":
            response = httpx.post(url, json=payload, timeout=60.0)
        else:
            response = httpx.get(url, params=payload, timeout=60.0)
        try:
            return response.status_code, response.json()
        except ValueError:
            return response.status_code, response.text
    except httpx.ConnectError:
        return 0, {"error": f"Cannot reach {url}."}
    except httpx.HTTPError as exc:
        return 0, {"error": str(exc)}


st.set_page_config(page_title="RAG Demo", layout="centered")
st.title("RAG Demo: Ingest + Ask")

base_url = st.sidebar.text_input("API base URL", DEFAULT_API_BASE_URL)
st.sidebar.caption(
    "Defaults to the API_BASE_URL environment variable if set. "
    "Point this at your live Render URL to test the deployed service."
)

ask_tab, ingest_tab = st.tabs(["Ask", "Ingest"])

with ingest_tab:
    st.subheader("Add a document to the vector store")
    with st.form("ingest_form"):
        document_id = st.text_input("document_id", placeholder="handbook-v1")
        source = st.text_input("source (optional)", placeholder="handbook.pdf")
        text = st.text_area("Text", height=250, placeholder="Paste the document text here...")
        ingest_submitted = st.form_submit_button("Ingest", type="primary")

    if ingest_submitted:
        payload = {"document_id": document_id, "text": text}
        if source:
            payload["source"] = source

        with st.spinner("Calling /ingest..."):
            status, data = call_api("POST", base_url, "/ingest", payload)

        if status == 200 and isinstance(data, dict):
            st.success(
                f"Indexed {data.get('chunks_indexed')} chunk(s) for "
                f"document_id={data.get('document_id')!r} (status: {data.get('status')})"
            )
        else:
            st.error(f"HTTP {status or 'connection failed'}")
        st.json(data)

with ask_tab:
    st.subheader("Ask a question")
    with st.form("ask_form"):
        question = st.text_area(
            "Question", height=100, placeholder="What is the mileage rate?"
        )
        top_k = st.slider("top_k (chunks retrieved)", min_value=1, max_value=10, value=5)
        ask_submitted = st.form_submit_button("Ask", type="primary")

    if ask_submitted:
        payload = {"question": question, "top_k": top_k}
        with st.spinner("Calling /ask..."):
            status, data = call_api("POST", base_url, "/ask", payload)

        if status != 200 or not isinstance(data, dict):
            st.error(f"HTTP {status or 'connection failed'}")
            st.json(data)
        else:
            answer = data.get("answer", {})
            chunk_ids = data.get("retrieved_chunk_ids", [])
            refused = bool(answer.get("sources_needed"))

            if refused:
                st.warning("⚠️ Refused / insufficient context")
            else:
                st.success("✅ Grounded answer")

            st.markdown("### Answer")
            st.write(answer.get("answer", ""))
            st.caption(
                f"confidence: {answer.get('confidence')} | "
                f"sources_needed: {answer.get('sources_needed')}"
            )

            st.markdown("### Citations (retrieved chunk IDs)")
            if chunk_ids:
                st.markdown("\n".join(f"- `{chunk_id}`" for chunk_id in chunk_ids))
            else:
                st.write("_No chunks were retrieved._")

            metric_cols = st.columns(4)
            metric_cols[0].metric("Model", str(data.get("model", "-")))
            metric_cols[1].metric("Tokens", str(data.get("tokens_used", "-")))
            metric_cols[2].metric("Latency", f"{data.get('latency_ms', '-')} ms")
            metric_cols[3].metric("Cost", f"${data.get('cost_usd', '-')}")

            with st.expander("Raw JSON"):
                st.json(data)
