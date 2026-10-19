from pathlib import Path

from fastapi import FastAPI
from rag_assistant.ingest import Ingest

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "RAG assistant API is running"}


@app.post("/ingest")
def ingest_document():
    project_root = Path(__file__).resolve().parents[2]
    pdf_path = project_root / "data" / "lost-in-the-middle.pdf"

    db = Ingest.run_complete_ingestion_pipeline(pdf_path)
    chunk_count = db._collection.count()

    return {
        "message": "Ingestion completed successfully",
        "chunks_stored": chunk_count,
    }