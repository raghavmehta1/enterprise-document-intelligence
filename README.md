# Enterprise Document Intelligence

A compact, fully offline document-retrieval demo for policies and technical manuals. It chunks source text, builds a TF-IDF index, retrieves the most relevant passages, and returns an extractive answer with verifiable citations.

## Why this project

This repository demonstrates the core mechanics behind a retrieval-augmented generation (RAG) workflow without claiming that a lexical model is an LLM. Every answer is copied from a retrieved passage and includes its source, section, and chunk identifier.

## Features

- Recursive text ingestion from `data/documents/`
- Section-aware overlapping chunks
- TF-IDF retrieval with cosine similarity
- Extractive answers with citations and confidence scores
- Honest abstention when evidence is weak
- Command-line interface and automated tests

## Run

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux: source .venv/bin/activate
pip install -r requirements.txt
python -m document_intelligence.cli "How quickly must a security incident be reported?"
pytest -q
```

## Example

```text
Answer: Security incidents must be reported to the Security Operations team within one hour of discovery.
Citation: security_policy.txt > Incident Reporting [chunk 0]
```

## Architecture

`documents -> section parser -> overlapping chunks -> TF-IDF matrix -> ranked passages -> extractive answer + citations`

## Limitations

This is a portfolio-scale baseline. TF-IDF is lexical rather than semantic, text files are used instead of production PDF/OCR pipelines, and the extractive answerer does not perform generative reasoning. A production extension could replace the retriever with embeddings/vector search and orchestrate ingestion, retrieval, verification, and response generation with LangGraph.

