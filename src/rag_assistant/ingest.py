from pathlib import Path
from unstructured.documents.elements import Element
from unstructured.chunking.title import chunk_by_title
from unstructured.partition.pdf import partition_pdf
class Ingest:

    @staticmethod
    def partition_document(file_path:str) -> list[Element]:
        """Extract elements from PDF using unstructured"""
        path = Path(file_path)
        if not path.is_file():
            raise FileNotFoundError(f"File {file_path} does not exist")

        print(f"📄 Partitioning document: {file_path}")

        elements = partition_pdf(
            filename=file_path, #path to your pdf file
            strategy="hi_res", # Use the most accurate (but slower) processing  method of extraction
            infer_table_structure=True, # Keep tables as structured HTML, not jumbled text
            extract_image_block_types=["Image"], # Grab Images found in the pdf
            extract_image_block_to_payload=True # Store images as base64 data you can actually use

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
            new_after_n_chars=2400, # Try to start a new chunk after 2400 characters
            combine_text_under_n_chars=500 # Merg tiny chunks under 500 chars with neighbors

        )
        print(f"✅ Created {len(chunks)} chunks")
        return chunks





