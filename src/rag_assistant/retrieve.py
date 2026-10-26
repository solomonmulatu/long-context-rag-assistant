from pathlib import Path
from typing import List

from langchain_chroma import Chroma
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
    ) -> List[Document]:
        persist_directory = settings.chroma_persist_directory
        embedding_model = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )

        db= Chroma(
            collection_name="rag_assistant",
            persist_directory=persist_directory,
            embedding_function=embedding_model,
        )

        search_kwargs = {"k": 5}
        if doc_id:
            search_kwargs["filter"] = {"doc_id": doc_id}

        retriever = db.as_retriever(search_kwargs=search_kwargs)
        chunks = retriever.invoke(query)
        return chunks

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
