import os
import re

import requests
import streamlit as st

API_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000").rstrip("/")
MAX_FILE_SIZE = 10 * 1024 * 1024


def filter_out_filename(filename:str)->str:
    return re.sub(
        r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}-',
        '',filename)


def get_documents() -> list[dict]:
    response = requests.get(f"{API_URL}/documents", timeout=20)
    response.raise_for_status()

    data = response.json()
    # Supports either a direct list or {"documents": [...]}.
    return data if isinstance(data, list) else data.get("documents", [])


def show_api_error(exc: requests.RequestException) -> None:
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            detail = response.json().get("detail", str(exc))
        except ValueError:
            detail = response.text or str(exc)
        st.error(f"API error: {detail}")
    else:
        st.error(
            f"Could not reach the RAG API at {API_URL}. "
            "Check that FastAPI is running."
        )


st.set_page_config(page_title="RAG Assistant", page_icon="📚")
st.title("📚 PDF RAG Assistant")
st.caption("Upload PDFs and ask questions grounded in their contents.")

# Show upload success after the app reruns and refreshes the document list.
if notice := st.session_state.pop("upload_notice", None):
    st.success(notice)

# Get the indexed PDFs from FastAPI/Chroma.
try:
    documents = get_documents()
    documents_error = None
except requests.RequestException as exc:
    documents = []
    documents_error = exc


with st.sidebar:
    st.subheader("Indexed PDFs")

    if documents_error:
        show_api_error(documents_error)
    elif documents:
        for document in documents:
            clean_name = filter_out_filename(
                document.get('filename', 'Unknown PDF')
            )
            st.write(
                f"**{clean_name}**  \n"
                f"{document.get('chunks', 0)} chunks"
            )
    else:
        st.caption("No indexed PDFs yet.")

st.subheader("Upload a PDF")

with st.form("upload_form"):
    uploaded_file = st.file_uploader(
        "Choose a PDF",
        type=["pdf"],
        max_upload_size=10,
        accept_multiple_files=False,
    )
    upload_submitted = st.form_submit_button("Upload and index")

if upload_submitted:
    if uploaded_file is None:
        st.warning("Choose a PDF first.")
    elif uploaded_file.size > MAX_FILE_SIZE:
        st.error("PDF must be 10 MB or smaller.")
    else:
        try:
            with st.spinner("Uploading and indexing PDF..."):
                response = requests.post(
                    f"{API_URL}/upload",
                    files={
                        "file": (
                            uploaded_file.name,
                            uploaded_file.getvalue(),
                            "application/pdf",
                        )
                    },
                    timeout=300,
                )
                response.raise_for_status()
                result = response.json()

            st.session_state["upload_notice"] = (
                f"Indexed {result['filename']} "
                f"({result['chunks_stored']} chunks)."
            )
            st.rerun()

        except requests.RequestException as exc:
            show_api_error(exc)

st.divider()
st.subheader("Ask a question")

if documents:
    document_by_id = {
        document["doc_id"]: document
        for document in documents
        if document.get("doc_id")
    }
    document_options = [None, *document_by_id.keys()]

    with st.form("question_form"):
        clean_name = filter_out_filename(
            document.get('filename', 'Unknown PDF')
        )
        selected_doc_id = st.selectbox(
            "Search scope",
            options=document_options,
            format_func=lambda doc_id: (
                "All uploaded PDFs"
                if doc_id is None
                else re.sub(
            r'^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}-',
            '',document_by_id[doc_id].get("filename", doc_id))
            ),
        )
        question = st.text_area(
            "Your question",
            placeholder="What does the document say about...?",
            height=100,
        )
        question_submitted = st.form_submit_button("Ask")

    if question_submitted:
        if not question.strip():
            st.warning("Enter a question first.")
        else:
            params = {"query": question.strip()}
            if selected_doc_id is not None:
                params["doc_id"] = selected_doc_id

            try:
                with st.spinner("Searching the documents and preparing an answer..."):
                    response = requests.get(
                        f"{API_URL}/query",
                        params=params,
                        timeout=120,
                    )
                    response.raise_for_status()
                    result = response.json()

                st.subheader("Answer")
                st.markdown(result["answer"])

                citations = result.get("citations", [])
                if citations:
                    st.subheader("Sources")
                    for citation in citations:
                        clean_name = filter_out_filename(
                            citation.get("source")
                        )
                        details = [
                            clean_name,
                            f"page {citation['page']}" if citation.get("page") else None,
                            citation.get("section"),
                        ]
                        details = [item for item in details if item]
                        st.markdown(
                            f"**[{citation['id']}]** " + " · ".join(details)
                        )
                elif result.get("abstained"):
                    st.info("No supporting source was found in the uploaded PDFs.")

            except requests.RequestException as exc:
                show_api_error(exc)
else:
    st.info("Upload and index a PDF before asking a question.")