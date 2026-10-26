from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from pypdf import PdfWriter

from rag_assistant import api


class FakeVectorStore:
    def get(self, *, where, include):
        return {"ids": ["chunk-1"], "metadatas": [{"doc_id": where["doc_id"]}]}


def make_blank_pdf() -> bytes:
    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)

    buffer = BytesIO()
    writer.write(buffer)
    return buffer.getvalue()


@pytest.fixture
def client(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "UPLOAD_TEMP_DIR", tmp_path)
    monkeypatch.setattr(api, "MAX_FILE_SIZE", 1024)

    # Mock ingestion so these tests don't call OpenAI or modify Chroma.
    monkeypatch.setattr(
        api.Ingest,
        "run_complete_ingestion_pipeline",
        lambda **kwargs: FakeVectorStore(),
    )

    return TestClient(api.app)


def test_rejects_oversized_pdf(client):
    response = client.post(
        "/upload",
        files={"file": ("large.pdf", b"%PDF-" + b"x" * 12000000, "application/pdf")},
    )

    assert response.status_code == 413


def test_rejects_wrong_file_type(client):
    response = client.post(
        "/upload",
        files={"file": ("notes.txt", b"not a PDF", "text/plain")},
    )

    assert response.status_code == 415


def test_rejects_empty_file(client):
    response = client.post(
        "/upload",
        files={"file": ("empty.pdf", b"", "application/pdf")},
    )

    assert response.status_code == 415


def test_rejects_pdf_with_no_extractable_text(client, monkeypatch):
    def no_text(**kwargs):
        raise ValueError("No extractable text was found in this PDF.")

    monkeypatch.setattr(api.Ingest, "run_complete_ingestion_pipeline", no_text)

    response = client.post(
        "/upload",
        files={
            "file": ("blank.pdf", make_blank_pdf(), "application/pdf")
        },
    )

    assert response.status_code == 422
    assert "No extractable text" in response.json()["detail"]