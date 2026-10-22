# PDF RAG Assistant

A retrieval-augmented generation (RAG) assistant that answers questions about user-uploaded PDFs and cites the source passages used.

## What it does

- Lets users upload PDFs through the Streamlit interface.
- Validates uploads before processing: PDF format, maximum file size of 10 MB, and maximum page count of 50. Encrypted or corrupt PDFs are rejected with a clear error.
- Parses PDFs with Unstructured and Poppler. Tesseract OCR can be enabled for scanned documents; if OCR is disabled and no text is extracted, the app asks the user to upload a text-based PDF.
- Cleans and chunks extracted content, preserving citation metadata.
- Assigns each PDF a unique `doc_id` and stores it with every chunk.
- Embeds chunks with the OpenAI Embeddings API and stores them in a shared Chroma collection.
- Lets users query one uploaded document or search across all uploaded documents.
- Generates answers from retrieved excerpts, with source citations. It abstains when the documents do not provide enough evidence.

## Shared knowledge base

Uploaded PDFs and their indexed chunks are stored in a **shared knowledge base**. Visitors can query documents uploaded by other visitors. The app does not provide per-user or per-session isolation, so the UI should tell users that uploads are shared.

Documents remain available until removed or expired under the configured retention policy. Removing a document must delete its vectors from Chroma by `doc_id` and delete its stored PDF if one is retained.

## Non-goals

- **Authentication or role-based access control:** This demo does not identify individual users or provide private document collections.
- **Model fine-tuning:** Uses existing language models without training them on uploaded documents.
- **Web search or external knowledge retrieval:** Answers are grounded in the shared document collection only.
- **Document editing:** Users can query documents but cannot edit or annotate them.
- **Autonomous agents or external tool execution:** The assistant answers questions and does not take actions outside the RAG workflow.
- **Guaranteed answers to every question:** The assistant may decline when retrieved context does not provide enough evidence.
- **Production-scale infrastructure:** Focuses on demonstrating RAG, citation accuracy, and document question answering.

## Tech stack

- **Language:** Python 3.12+
- **Document parsing:** Unstructured Python library
- **PDF utilities:** Poppler (`poppler-utils`)
- **OCR:** Tesseract (`tesseract-ocr`), optional for scanned PDFs
- **File type detection:** `libmagic`, where required by the platform and Unstructured configuration
- **Embeddings:** OpenAI Embeddings API, such as `text-embedding-3-small`
- **Vector store:** Chroma
- **Language model:** OpenAI API
- **API:** FastAPI
- **UI:** Streamlit
- **Python environment and dependencies:** `uv`

## Architecture

### PDF upload and ingestion

```text
User uploads PDF in Streamlit
    → FastAPI validates file type, size, page count, and readability
    → Unstructured parses PDF (Poppler as needed)
    → Tesseract OCR for scanned PDFs, when enabled
    → Clean and chunk content
    → Attach doc_id, source, page, and section metadata
    → OpenAI embeddings
    → Shared Chroma collection
```

Each PDF gets a unique `doc_id`. Every chunk retains that ID and its citation metadata. The temporary upload file is removed after processing unless the app is configured to retain the original PDF.

### Question and answer

```text
Question + optional doc_id
    → Streamlit UI
    → FastAPI
    → OpenAI query embedding
    → Chroma top-k retrieval, filtered by doc_id when selected
    → Optional reranking
    → Grounded prompt with citation metadata
    → OpenAI LLM
    → Answer with citations, or abstention
```

See [`docs/design.md`](docs/design.md) for chunking, retrieval, prompt, and abstention choices. See [`docs/rag-architecture.png`](docs/rag-architecture.png) for the architecture diagram.

## Project structure

```text
rag-assistant/
├── src/rag_assistant/
│   ├── ingest.py        # Validate, parse, chunk, embed, and store PDFs
│   ├── retrieve.py      # Retrieve chunks from the shared Chroma collection
│   ├── generate.py      # Build grounded prompts and generate cited answers
│   ├── config.py        # Application settings and environment variables
│   ├── schemas.py       # API request and response models
│   └── api.py           # FastAPI upload, query, and document-management routes
├── app/
│   └── streamlit_app.py # UI that calls the FastAPI service
├── data/                # Optional sample documents; check licensing before sharing
├── uploads/             # Temporary/runtime uploads; do not commit
├── eval/
│   ├── questions.json   # Evaluation questions
│   └── run_eval.py      # Evaluation runner
├── tests/
├── docs/
│   ├── design.md
│   └── rag-architecture.png
├── .env.example
├── .gitignore
├── pyproject.toml
├── uv.lock
├── LICENSE
└── README.md
```

## Getting started

### Prerequisites

- Python 3.12+
- `uv`
- An OpenAI API key
- Poppler installed and available on your system path
- Tesseract installed if OCR for scanned PDFs is enabled
- `libmagic` if required by your platform and Unstructured configuration

### Install dependencies

From the project root in PowerShell:

```powershell
uv sync
```

Poppler, Tesseract, and `libmagic` are system dependencies; `uv sync` does not install them.

### Configure environment variables

Copy the example file and add your API key to `.env`:

```powershell
Copy-Item .env.example .env
```

Example `.env.example`:

```dotenv
OPENAI_API_KEY=replace-me
OPENAI_CHAT_MODEL=gpt-4o
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
CHROMA_PERSIST_DIRECTORY=chroma_data
API_BASE_URL=http://127.0.0.1:8000
```

Keep `.env` and API keys out of version control. Commit `.env.example` with placeholders only. Resolve the Chroma path relative to the project root so it does not depend on the directory from which the app is launched.

### Run the application

Start the FastAPI backend in one terminal:

```powershell
uv run uvicorn rag_assistant.api:app --reload
```

Start the Streamlit UI in another terminal:

```powershell
uv run streamlit run app/streamlit_app.py
```

The Streamlit app sends uploads and questions to FastAPI. After an upload succeeds, the UI should show the document’s name and `doc_id`, and let the user query that document or the full shared collection.

## Evaluation

Add representative questions and expected evidence to `eval/questions.json`, then run:

```powershell
uv run python eval/run_eval.py
```

Review answer correctness, citation accuracy, document filtering by `doc_id`, upload validation behavior, and whether the assistant abstains when the documents do not support an answer.