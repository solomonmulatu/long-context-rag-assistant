import uuid
from pathlib import Path
from typing import Dict

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from pypdf import PdfReader

from rag_assistant.generate import Generate
from rag_assistant.ingest import Ingest
from rag_assistant.retrieve import Retrieve

app = FastAPI()


MAX_FILE_SIZE = 10 * 1024 * 1024  # 10 MB
MAX_PAGE_COUNT = 50
PROJECT_ROOT = Path(__file__).resolve().parents[2]
UPLOAD_TEMP_DIR = PROJECT_ROOT / "uploads" / "tmp"
UPLOAD_TEMP_DIR.mkdir(parents=True, exist_ok=True)

# In-memory database to store indexing status for portfolio endpoints
DOCUMENT_STATUS_DB: Dict[str, dict] = {}


class DocumentStatusResponse(BaseModel):
    doc_id: str
    filename: str
    status: str
    pages: int
    chunks_stored: int
    error: str | None = None


@app.get("/")
async def root():
    return {"message": "RAG assistant API is running"}



@app.get("/query")
def query_document(query: str | None = None, doc_id: str | None = None):

    chunks=Retrieve.retrieve_relevant_chunks(query, doc_id=doc_id)

    response=Generate.generate_final_answer(chunks,query)

    return response

@app.post("/upload", status_code=201)
async def upload_pdf(file: UploadFile = File(...)):
    filename = Path(file.filename or "").name

    if not filename.lower().endswith(".pdf"):
        raise HTTPException(
            status_code=415,
            detail="Please upload a PDF file.",
        )

    doc_id = str(uuid.uuid4())
    temp_pdf_path = UPLOAD_TEMP_DIR / f"{doc_id}-{filename}"

    DOCUMENT_STATUS_DB[doc_id] = {
        "doc_id": doc_id,
        "filename": filename,
        "status": "validating",
        "stage": "Checking PDF",
        "pages": None,
        "chunks_stored": 0,
        "error": None,
    }
    # 1. Stream file to disk and validate file size dynamically
    total_size = 0
    try:
        with temp_pdf_path.open("wb") as destination:
            while chunk := await file.read(1024 * 1024):
                total_size += len(chunk)
                if total_size > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=413,
                        detail="PDF must be 10 MB or smaller.",
                    )
                destination.write(chunk)

        # 2. Check Magic Bytes Header (%PDF-)
        with temp_pdf_path.open("rb") as uploaded_pdf:
            header = uploaded_pdf.read(1024)

        if b"%PDF-" not in header:
            raise HTTPException(
                status_code=415,
                detail="The uploaded file is not a valid PDF.",
            )

        # 3. Read page count & encryption status
        reader = PdfReader(str(temp_pdf_path), strict=True)
        if reader.is_encrypted:
            raise HTTPException(
                status_code=400,
                detail="Encrypted PDFs are not supported.",
            )

        page_count = len(reader.pages)
        if page_count > MAX_PAGE_COUNT:
            raise HTTPException(
                status_code=413,
                detail=f"PDF must have {MAX_PAGE_COUNT} pages or fewer.",
            )

        DOCUMENT_STATUS_DB[doc_id].update({
            "status": "processing",
            "stage": "Parsing and indexing",
            "pages": page_count,
        })

        # 4. Synchronous Processing & Vector Database Ingestion
        db = Ingest.run_complete_ingestion_pipeline(
            pdf_path=temp_pdf_path,
            doc_id=doc_id,
            file_name=filename,
        )
        stored_chunks = db.get(
            where={"doc_id": doc_id},
            include=["metadatas"],
        )
        chunks_stored = len(stored_chunks["ids"])

        # Record success state
        DOCUMENT_STATUS_DB[doc_id] = {
            "doc_id": doc_id,
            "filename": filename,
            "status": "completed",
            "pages": page_count,
            "chunks_stored": chunks_stored,
            "error": None,
        }

        return DOCUMENT_STATUS_DB[doc_id]

    except HTTPException:
        raise
    except ValueError as exc:
        # Catch custom ingestion errors (e.g., failed text extraction)
        DOCUMENT_STATUS_DB[doc_id] = {
            "doc_id": doc_id,
            "filename": filename,
            "status": "failed",
            "pages": 0,
            "chunks_stored": 0,
            "error": str(exc),
        }
        raise HTTPException(
            status_code=422,
            detail=str(exc),
        ) from exc
    except Exception as exc:
        DOCUMENT_STATUS_DB[doc_id] = {
            "doc_id": doc_id,
            "filename": filename,
            "status": "failed",
            "pages": 0,
            "chunks_stored": 0,
            "error": "Internal processing error.",
        }
        raise HTTPException(
            status_code=500,
            detail="The PDF passed validation but could not be processed.",
        ) from exc
    finally:
        # Guarantee local file removal after processing
        if temp_pdf_path.exists():
            temp_pdf_path.unlink()


@app.get("/documents/{doc_id}", response_model=DocumentStatusResponse)
def get_document_status(doc_id: str):
    """Retrieve metadata and status for an ingested document."""
    doc_info = DOCUMENT_STATUS_DB.get(doc_id)
    if not doc_info:
        raise HTTPException(
            status_code=404,
            detail="Document ID not found.",
        )
    return doc_info


@app.get('/documents')
def get_all_documents():
    """Retrieve all ingested documents."""
    return Retrieve.list_indexed_documents()
