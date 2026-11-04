from pathlib import Path
from typing import List

from langchain_chroma import Chroma
from langchain_classic.retrievers import ensemble
from langchain_community.retrievers import BM25Retriever
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings

from rag_assistant.config import settings


class Retrieve:

    @staticmethod
    def check_db_store(
            persist_directory: str = settings.chroma_persist_directory,
    ) -> bool:
        """Return True if the Chroma collection contains records."""
        if not Path(persist_directory).is_dir():
            print("Vector store directory does not exist.")
            return False

        vectorstore = Chroma(
            persist_directory=persist_directory,
            collection_name="rag_assistant",
        )

        count = vectorstore._collection.count()
        print(f"Found {count} records in the vector store.")
        return count > 0

    @staticmethod
    def retrieve_relevant_chunks(
            query: str,
            doc_id: str | None = None,
            retriever: str = "hybrid",
    ) -> List[Document]:
        if retriever not in {"vector", "bm25", "hybrid"}:
            raise ValueError(
                f"Unsupported retriever '{retriever}'. "
                "Use 'vector', 'bm25', or 'hybrid'."
            )

        persist_directory = settings.chroma_persist_directory
        embedding_model = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )

        db = Chroma(
            collection_name="rag_assistant",
            persist_directory=persist_directory,
            embedding_function=embedding_model,
        )

        search_kwargs = {"k": 5}
        if doc_id:
            search_kwargs["filter"] = {"doc_id": doc_id}

        if retriever == "vector":
            vector_retriever = db.as_retriever(search_kwargs=search_kwargs)
            return vector_retriever.invoke(query)

        stored = db.get(
            where={"doc_id": doc_id} if doc_id else None,
            include=["documents", "metadatas"],
        )
        docs = [
            Document(page_content=text, metadata=meta or {})
            for text, meta in zip(stored["documents"], stored["metadatas"])
            if text
        ]
        if not docs:
            return []

        bm25_retriever = BM25Retriever.from_documents(docs, k=5)

        if retriever == "bm25":
            return bm25_retriever.invoke(query)

        vector_retriever = db.as_retriever(search_kwargs=search_kwargs)
        hybrid_retriever = ensemble.EnsembleRetriever(
            retrievers=[vector_retriever, bm25_retriever],
            weights=[0.7, 0.3],
        )
        return hybrid_retriever.invoke(query)

    @staticmethod
    def list_indexed_documents() -> List[dict]|None:
        embedding_model = OpenAIEmbeddings(
            model=settings.openai_embedding_model,

        )

        db = Chroma(
            collection_name="rag_assistant",
            persist_directory=settings.chroma_persist_directory,
            embedding_function=embedding_model,
        )

        stored = db.get(include=['metadatas'])
        documents_by_id={}

        for metadata in stored['metadatas']:
            if not metadata:
                continue

            doc_id=metadata.get('doc_id')

            if not doc_id:
                continue


            if doc_id not in documents_by_id:
                documents_by_id[doc_id] = {
                    'doc_id': doc_id,
                    'filename': metadata.get('filename','unknown filename'),
                    'chunks':0
                }


            documents_by_id[doc_id]['chunks'] += 1


        return list(documents_by_id.values())
