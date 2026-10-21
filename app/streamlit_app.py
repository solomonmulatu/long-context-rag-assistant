import os

import requests
import streamlit as st

API_URL = os.getenv("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="RAG Assistant", page_icon="📚")
st.title("📚 RAG Assistant")
st.caption("Ask questions about *How Language Models Use Long Contexts*.")

with st.form("question_form"):
    question = st.text_area(
        "Your question",
        placeholder="What is the lost-in-the-middle effect?",
        height=100,
    )
    submitted = st.form_submit_button("Ask")

if submitted:
    if not question.strip():
        st.warning("Enter a question first.")
    else:
        try:
            with st.spinner("Searching the paper and preparing an answer..."):
                response = requests.get(
                    f"{API_URL}/query",
                    params={"query": question.strip()},
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
                    details = [
                        citation.get("source"),
                        f"page {citation['page']}" if citation.get("page") else None,
                        citation.get("section"),
                    ]
                    details = [item for item in details if item]

                    st.markdown(
                        f"**[{citation['id']}]** " + " · ".join(details)
                    )
            elif result.get("abstained"):
                st.info("No supporting source was found in the provided documents.")

        except requests.RequestException as exc:
            st.error(
                "Could not reach the RAG API. Check that FastAPI is running "
                f"at {API_URL}.\n\nDetails: {exc}"
            )