# RAG Assistant: How Language Models Use Long Contexts

A retrieval-augmented generation (RAG) assistant that answers questions about **How Language Models Use Long Contexts** using the provided document and citations to its source passages.

## What it does

- Parses the source PDF, including OCR for scanned pages when needed.
- Cleans and chunks extracted content while retaining source, page, and section metadata.
- Embeds chunks with the OpenAI Embeddings API and stores them in Chroma.
- Retrieves relevant passages for a user question and uses them as context for an OpenAI language model.
- Returns an answer with citations. If the document does not provide enough evidence, the assistant says so rather than guessing.

## Non-goals

- **Multi-user authentication or role-based access control:** Intended for a single user or trusted demo environment.
- **Model fine-tuning:** Uses existing language models without training them on the source material.
- **Web search or external knowledge retrieval:** Answers are grounded only in the configured documents.
- **Document modification:** Users can query the knowledge base but cannot edit or annotate source documents.
- **Autonomous agents or external tool execution:** The assistant answers questions and does not take actions outside the RAG workflow.
- **Guaranteed answers to every question:** The assistant may decline when retrieved context does not provide enough evidence.
- **Production-scale infrastructure:** Focuses on demonstrating RAG, citation accuracy, and long-context handling rather than large-scale deployment.

## Tech stack

- **Language:** Python 3.12+
- **Document parsing:** Unstructured Python library
- **PDF utilities:** Poppler (`poppler-utils`)
- **OCR:** Tesseract (`tesseract-ocr`)
- **File type detection:** `libmagic`
- **Embeddings:** OpenAI Embeddings API (for example, `text-embedding-3-small`)
- **Vector store:** Chroma
- **Language model:** OpenAI API
- **API:** FastAPI
- **UI:** Streamlit

## Architecture

### Ingestion

```text
PDF → Unstructured parsing → clean and normalize → chunk with metadata
    → OpenAI embeddings → Chroma
```

Poppler and Tesseract support PDF processing and OCR as needed. Each stored chunk retains `source`, `page`, and `section` metadata for citations.

### Question and answer

```text
User → Streamlit UI → FastAPI → OpenAI query embedding → Chroma top-k retrieval
     → optional reranking → grounded prompt → OpenAI LLM → cited answer
```

See [`docs/design.md`](docs/design.md) for the initial chunking, retrieval, prompt, and abstention choices. See [`docs/rag-architecture.svg`](docs/rag-architecture.svg) for the architecture diagram.

## Project structure

```text
rag-assistant/
├── src/rag_assistant/
│   ├── ingest.py        # Parse, clean, chunk, embed, and store documents
│   ├── retrieve.py      # Embed questions and retrieve relevant chunks
│   ├── generate.py      # Build the grounded prompt and generate answers
│   ├── config.py        # Application settings and environment variables
│   ├── schemas.py       # Shared request and response models
│   └── api.py           # FastAPI routes
├── app/
│   └── streamlit_app.py # UI that calls the FastAPI service
├── data/                # Source documents; check licensing before sharing
├── eval/
│   ├── questions.json   # Evaluation questions
│   └── run_eval.py      # Evaluation runner
├── tests/
├── docs/
│   ├── design.md
│   └── rag-architecture.svg
├── .env.example
├── .gitignore
├── pyproject.toml
├── LICENSE
└── README.md
```

## Getting started

### Prerequisites

- Python 3.12 or newer
- Poppler and Tesseract installed and available on your system path for PDF and OCR workflows
- `libmagic` installed if required by your platform and Unstructured configuration
- An OpenAI API key

### Set up a virtual environment

In PowerShell, from the project root:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
```

If PowerShell blocks activation, allow it for the current terminal session and activate again:

```powershell
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass
.\.venv\Scripts\Activate.ps1
```

### Install Python dependencies

Declare the application’s Python packages in `pyproject.toml`, then install the project from the root directory:

```powershell
python -m pip install --upgrade pip
python -m pip install -e .
```

Poppler, Tesseract, and `libmagic` are system dependencies; installing Python packages does not install those programs.

### Configure environment variables

Copy the example file and add your key to the local `.env` file:

```powershell
Copy-Item .env.example .env
```

Example `.env.example`:

```dotenv
OPENAI_API_KEY=replace-me
OPENAI_CHAT_MODEL=your-chat-model
OPENAI_EMBEDDING_MODEL=text-embedding-3-small
CHROMA_PERSIST_DIRECTORY=./chroma_data
API_BASE_URL=http://127.0.0.1:8000
```

Keep `.env` and API keys out of version control. Commit `.env.example` with placeholders only.

## Running the application

Start the FastAPI backend in one terminal:

```powershell
uvicorn rag_assistant.api:app --reload
```

Start the Streamlit UI in another terminal:

```powershell
streamlit run app/streamlit_app.py
```

The Streamlit app sends questions to FastAPI, which coordinates retrieval and generation. Add the source PDF to the configured data location and run the ingestion workflow before asking questions.

## Evaluation

Add representative questions and expected evidence to `eval/questions.json`, then run:

```powershell
python eval/run_eval.py
```

Evaluate answer correctness, citation accuracy, and whether the assistant abstains when the document does not support an answer.

## Licensing and source documents

Check the source PDF’s license and distribution terms before committing or sharing it. Keep raw documents out of the repository unless you have permission to redistribute them. The project’s `LICENSE` covers this repository’s code and original materials; it may not grant rights to the source document.
