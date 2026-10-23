import json
import os
from pathlib import Path
from typing import TypedDict, List

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from unstructured.documents.elements import Element
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.pdf import partition_pdf

from rag_assistant.config import settings
from rag_assistant.generate import Generate
from rag_assistant.retrieve import Retrieve


class ChunkContent(TypedDict):
    text: str
    tables: list[str]
    images: list[str]
    types: list[str]


class Ingest:

    @staticmethod
    def partition_document(file_path: str) -> list[Element]:
        """Extract elements from PDF using unstructured"""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File {file_path} does not exist")

        print(f"📄 Partitioning document: {file_path}")

        elements = partition_pdf(
            filename=file_path,  # path to your pdf file
            strategy="hi_res",  # Use the most accurate (but slower) processing  method of extraction
            infer_table_structure=True,  # Keep tables as structured HTML, not jumbled text
            extract_image_block_types=["Image"],  # Grab Images found in the pdf
            extract_image_block_to_payload=True  # Store images as base64 data you can actually use

        )
        print(f"✅ Extracted {len(elements)} elements")

        return elements

    @staticmethod
    def create_chunks_by_title(elements: list[Element]) -> list[Element]:
        """Create intelligent chunks using title-based strategy"""
        print("🔨 Creating smart chunks...")

        chunks = chunk_by_title(
            elements,  # parsed pdf element from previous step
            max_characters=3000,  # Hard Limit  -never exceed 3000 chars per chunk
            new_after_n_chars=2400,  # Try to start a new chunk after 2400 characters
            combine_text_under_n_chars=500  # Merg tiny chunks under 500 chars with neighbors

        )
        print(f"✅ Created {len(chunks)} chunks")
        return chunks

    @staticmethod
    def separate_content_type(chunk: Element) -> ChunkContent:
        """Collect text, table content, and image payloads from a chunk."""

        content_data: ChunkContent = {
            "text": chunk.text,
            "tables": [],
            "images": [],
            "types": ["text"],
        }

        # check for images and tables from the original element
        if hasattr(chunk, 'metadata') and hasattr(chunk.metadata, 'orig_elements'):
            for element in chunk.metadata.orig_elements:
                element_type = type(element).__name__

                print(f"Element Type {element_type}")

                # Handle tables
                if element_type == 'Table':
                    content_data['types'].append('table')
                    table_html = getattr(element.metadata, 'text_as_html', None)

                    content_data['tables'].append(table_html or element.text)

                # Handle images
                if element_type == 'Image':
                    if hasattr(element, 'metadata') and hasattr(element.metadata, 'image_base64'):
                        content_data['types'].append('image')
                        content_data['images'].append(element.metadata.image_base64)

        content_data['types'] = list(set(content_data['types']))

        return content_data

    @staticmethod
    def summarise_chunks(chunks: List[Element],doc_id:str,file_name:str) -> List[Document]:
        """Process chunks and create LangChain documents with AI summaries."""
        documents: list[Document] = []
        total_chunks = len(chunks)

        print("Processing chunks with AI summaries...")

        for index, chunk in enumerate(chunks, start=1):
            print(f"Processing chunk {index}/{total_chunks}")

            content = Ingest.separate_content_type(chunk)

            print(f"  Types found: {content['types']}")
            print(
                f"  Tables: {len(content['tables'])}, "
                f"Images: {len(content['images'])}"
            )

            if content["tables"] or content["images"]:
                try:
                    page_content = Generate.create_ai_enhanced_summary(
                        content["text"],
                        content["tables"],
                        content["images"],
                    )
                    print("  AI summary created successfully")
                except Exception as exc:
                    print(f"  AI summary failed: {exc}")
                    page_content = content["text"]
            else:
                page_content = content["text"]

            chunk_metadata = getattr(chunk, "metadata", None)
            metadata = {
                "doc_id":doc_id,
                "filename":file_name,
                "chunk_index": index,
                "content_types": json.dumps(content["types"]),
                "original_content": json.dumps({
                    "raw_text": content["text"],
                    "tables_html": content["tables"],
                    "images_base64": content["images"],
                }),
            }

            # Preserve available fields for source citations.
            for key in ("filename", "page_number", "section"):
                value = getattr(chunk_metadata, key, None)
                if value is not None:
                    metadata[key] = value

            document = Document(
                page_content=page_content,
                metadata=metadata,
            )
            documents.append(document)

        print(f"Processed {len(documents)} chunks")
        return documents

    @staticmethod
    def export_chunks_to_json(
            chunks: list[Document],
            filename: str | Path = "chunks_export.json",
    ) -> list[dict]:
        """Export processed chunks and their metadata to a JSON file."""
        export_data = []

        for index, doc in enumerate(chunks, start=1):
            metadata = dict(doc.metadata)

            original_content = metadata.get("original_content", "{}")
            if isinstance(original_content, str):
                original_content = json.loads(original_content)

            metadata["original_content"] = original_content

            export_data.append({
                "chunk_id": index,
                "enhanced_content": doc.page_content,
                "metadata": metadata,
            })

        with open(filename, "w", encoding="utf-8") as file:
            json.dump(export_data, file, indent=2, ensure_ascii=False)

        print(f"Exported {len(export_data)} chunks to {filename}")
        return export_data

    @staticmethod
    def create_vector_store(
            documents: list[Document],
            persist_directory: str = settings.chroma_persist_directory,
    ) -> Chroma:
        """Create OpenAI embeddings and store documents in persistent Chroma."""
        if not documents:
            raise ValueError("Cannot create a vector store with no documents.")

        Path(persist_directory).mkdir(parents=True, exist_ok=True)

        print("Creating embeddings and storing documents in Chroma...")

        embedding_model = OpenAIEmbeddings(
            model=settings.openai_embedding_model
        )

        vectorstore = Chroma.from_documents(
            documents=documents,
            embedding=embedding_model,
            persist_directory=persist_directory,
            collection_name="rag_assistant",
            collection_metadata={"hnsw:space": "cosine"},
        )

        print(f"Vector store created at {persist_directory}")
        return vectorstore





    @staticmethod
    def run_complete_ingestion_pipeline(pdf_path: str | Path,doc_id: str,file_name:str) -> Chroma:
        """Run the complete RAG ingestion pipeline."""
        print("Starting RAG ingestion pipeline")
        print("=" * 50)

        persist_directory = settings.chroma_persist_directory
        embedding_model = OpenAIEmbeddings(
            model=settings.openai_embedding_model,
            api_key=settings.openai_api_key,
        )


        elements = Ingest.partition_document(pdf_path)
        chunks = Ingest.create_chunks_by_title(elements)
        documents = Ingest.summarise_chunks(chunks,doc_id,file_name)

        vectorstore = Ingest.create_vector_store(
            documents,
            persist_directory=persist_directory,
        )
        Ingest.export_chunks_to_json(documents)

        print("Ingestion pipeline completed successfully")
        return vectorstore


