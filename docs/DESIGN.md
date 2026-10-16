# RAG design notes

## Chunking

- Split extracted text into chunks of about **700 tokens**, with **100 tokens of overlap**. This is a starting point: keep sections and page boundaries where practical, and avoid combining unrelated sections.
- Attach `source`, `page`, and `section` metadata to every chunk. Preserve page references through parsing and chunking so citations can point back to the PDF.
- Tune chunk size and overlap against a small set of representative questions; use retrieved context quality and citation accuracy as the guide.

## Retrieval

- Retrieve **top-k = 5** chunks from Chroma for each question using the OpenAI query embedding model that matches the ingestion embedding model.
- Keep reranking optional. If enabled, rerank the retrieved candidates and pass the best few chunks to the answer prompt. Ensure each result retains its citation metadata.

## Answer prompt

Use a grounded prompt along these lines:

> You answer questions about “How Language Models Use Long Contexts.” Use only the document excerpts below. If they do not contain enough evidence to answer, say that the answer is not available in the documents. Do not guess or use outside knowledge. Cite each factual claim with the supplied source and page (and section when available), using a format such as `[How Language Models Use Long Contexts, p. 7, §3.2]`.
>
> **Question:** {question}
>
> **Document excerpts:**
> {retrieved_chunks_with_source_page_section}
>
> **Answer:**

## When the answer is not in the documents

- Abstain when retrieval returns no relevant evidence, or when the excerpts do not support the requested answer. Respond plainly: **“I couldn’t find that in the provided document.”**
- Do not fill gaps with model knowledge. If only part of the question is supported, answer that part with citations and state which part the document does not establish.
- Include citations only for claims supported by retrieved excerpts; never invent page or section references.
