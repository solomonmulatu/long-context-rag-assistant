from typing import List

from langchain_core.callbacks import CallbackManagerForRetrieverRun
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from rag_assistant import retrieve
from rag_assistant.retrieve import Retrieve


class FakeVectorRetriever(BaseRetriever):
    docs: List[Document] = []

    def _get_relevant_documents(
        self,
        query: str,
        *,
        run_manager: CallbackManagerForRetrieverRun,
    ) -> List[Document]:
        return list(self.docs)


class FakeChroma:
    last_get_where = None
    last_search_kwargs = None

    def __init__(self, *args, documents=None, metadatas=None, vector_docs=None, **kwargs):
        self._documents = documents if documents is not None else []
        self._metadatas = metadatas if metadatas is not None else []
        self._vector_docs = vector_docs if vector_docs is not None else [
            Document(page_content="semantic match about neural networks", metadata={"doc_id": "doc-1"})
        ]

    def get(self, where=None, include=None):
        FakeChroma.last_get_where = where
        return {
            "documents": list(self._documents),
            "metadatas": list(self._metadatas),
        }

    def as_retriever(self, search_kwargs=None):
        FakeChroma.last_search_kwargs = search_kwargs or {}
        return FakeVectorRetriever(docs=self._vector_docs)


class FakeEmbeddings:
    def __init__(self, *args, **kwargs):
        pass


def _patch_retrieve(monkeypatch, *, documents, metadatas, vector_docs=None):
    FakeChroma.last_get_where = None
    FakeChroma.last_search_kwargs = None

    def chroma_factory(*args, **kwargs):
        return FakeChroma(
            documents=documents,
            metadatas=metadatas,
            vector_docs=vector_docs,
        )

    monkeypatch.setattr(retrieve, "Chroma", chroma_factory)
    monkeypatch.setattr(retrieve, "OpenAIEmbeddings", FakeEmbeddings)


def test_retrieve_returns_chunks_for_query(monkeypatch):
    _patch_retrieve(
        monkeypatch,
        documents=["neural networks and backpropagation"],
        metadatas=[{"doc_id": "doc-1"}],
    )

    chunks = Retrieve.retrieve_relevant_chunks("neural networks")

    assert chunks
    assert all(isinstance(chunk, Document) for chunk in chunks)


def test_retrieve_passes_doc_id_filter_to_get(monkeypatch):
    _patch_retrieve(
        monkeypatch,
        documents=["chunk about transformers"],
        metadatas=[{"doc_id": "doc-1"}],
    )

    Retrieve.retrieve_relevant_chunks("transformers", doc_id="doc-1")

    assert FakeChroma.last_get_where == {"doc_id": "doc-1"}
    assert FakeChroma.last_search_kwargs == {
        "k": 5,
        "filter": {"doc_id": "doc-1"},
    }


def test_retrieve_returns_empty_when_store_empty(monkeypatch):
    _patch_retrieve(
        monkeypatch,
        documents=[],
        metadatas=[],
    )

    chunks = Retrieve.retrieve_relevant_chunks("anything")

    assert chunks == []
