# Stitch — Enterprise Document Intelligence & Knowledge Retrieval Platform

[![CI](https://github.com/KoppisettiGnanaVishnu/Stitch-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/KoppisettiGnanaVishnu/Stitch-rag/actions/workflows/ci.yml)

> *Connecting the threads of knowledge.*

**Category:** Retrieval-Augmented Generation (RAG)

Stitch is a Retrieval-Augmented Generation (RAG) system that answers questions **strictly from your own documents** — PDFs, Word files and plain text — with source citations and per-answer confidence scoring. Everything runs locally: embeddings and generation never leave your machine.

**What makes it different:** most RAG demos only show an answer. Stitch also tells you *how much to trust it* — every answer comes with a confidence, groundedness and completeness score, plus automatic flags for unsupported claims or missing citations.

---

## Features

- 📄 **Multi-document workspace** — upload any mix of PDF / DOCX / TXT and query them together
- 🔎 **Two retrieval strategies** — plain similarity search and Maximal Marginal Relevance (MMR) for diverse results
- 📚 **Citations by default** — answers cite source filename and page number
- 🚦 **Answer quality panel** — confidence, groundedness and completeness scores with issue detection (hallucination and citation checks)
- 🔒 **Fully local** — Ollama for generation, `all-MiniLM-L6-v2` for embeddings
- ⚙️ **Configurable** — chunk size, overlap, top-k, models and more via a single config class

## Architecture

```
                      ┌──────────────────────────────┐
                      │   Streamlit app (app/main.py) │
                      │   upload · chat · quality UI  │
                      └──────────────┬───────────────┘
                                     │
                      ┌──────────────▼───────────────┐
                      │      RAGPipeline (src/rag)   │
                      └──────────────────────────────┘

  1. document_loader  → PDF/DOCX/TXT → {text, source, page}
  2. chunker          → sentence-boundary chunks with overlap
  3. embeddings       → all-MiniLM-L6-v2 (sentence-transformers)
  4. retriever        → cosine similarity, top-k or MMR
  5. generator        → Ollama (llama3.2:3b), grounded prompt
  6. evaluator        → confidence · groundedness · issues
```

## Project structure

```
app/            Streamlit chat interface
src/rag/        Core pipeline (loader, chunker, embeddings, retriever, generator, evaluator, config)
evaluation/     Retrieval comparison (normal vs MMR) and confidence benchmarks
tests/          pytest suite (evaluator unit tests + pipeline integration tests)
notebook/       Exploration and prototyping scripts
data/           Sample documents
```

## Setup

Requirements: Python 3.11+ and [Ollama](https://ollama.com).

```bash
# 1. Install dependencies (uv)
uv sync

# ...or with pip
pip install -r requirements.txt

# 2. Install the local generation model
ollama pull llama3.2:3b

# 3. Make sure the Ollama app/service is running
```

## Usage

### Web app

```bash
streamlit run app/main.py
```

1. Upload documents in the sidebar → **PROCESS DOCUMENTS**
2. Ask questions in the chat; expand **VIEW SOURCES** to see the passages behind each answer
3. Check the **ANSWER QUALITY** panel (🟢 confident / 🟡 partial / 🔴 unconfident) and detected issues
4. Switch retrieval strategy in the sidebar: *normal* vs *diverse (MMR)*

### Python API

```python
import sys; sys.path.insert(0, "src")

from rag.pipeline import RAGPipeline
from rag.config import RAGConfig

pipeline = RAGPipeline(
    ["data/sample.txt"],
    source_name=["sample.txt"],
    config=RAGConfig(top_k=3, chunk_size=200, chunk_overlap=40),
)

# Simple answer
print(pipeline.ask("What are the library timings?"))

# Answer with sources and quality evaluation
result = pipeline.ask_detailed("What are the library timings?")
print(result["answer"])
print(result["evaluation"])   # confidence, groundedness, verdict, issues
```

## Tests

```bash
pytest tests/ -q
```

Note: the pipeline integration tests (`test_rag_pipeline.py`) use `data/psy-ca1.pdf` and call the Ollama generator, so Ollama must be running. The evaluator tests run instantly with no external services.

## Configuration

All defaults live in `src/rag/config.py` (`RAGConfig`):

| Setting | Default | Purpose |
|---|---|---|
| `chunk_size` / `chunk_overlap` | 200 / 40 | Sentence-based chunking |
| `top_k` | 3 | Passages retrieved per query |
| `candidate_k`, `lambda_param` | 10, 0.7 | MMR candidate pool and relevance/diversity balance |
| `generator_model` | `llama3.2:3b` | Ollama model used for answers |
| `embedding_model` | `all-MiniLM-L6-v2` | sentence-transformers model |
| `temperature` | 0.1 | Low = strictly grounded answers |

## Roadmap

- [x] Evaluator wired into the UI with per-answer quality scores
- [x] Central config (`RAGConfig`) for all pipeline settings
- [x] Streaming answers and conversation memory
- [x] Embedding persistence (no re-indexing per session)
- [x] NumPy-vectorized similarity search
- [ ] Repeatable evaluation benchmark with published metrics
- [x] Docker Compose setup (app + Ollama sidecar, persistent volumes)
- [ ] FastAPI service mode (platform API: /index, /ask)

## Deployment (Docker)

Run Stitch + Ollama with one command. The index and chat history are
persisted in a named volume.

```bash
# Start everything (first run pulls llama3.2:3b — takes a few minutes)
docker compose up -d

# Open the app
open http://localhost:8501

# Stop
docker compose down
```

Override the model or port:

```bash
STITCH_GENERATOR_MODEL=mistral:7b STITCH_PORT=9090 docker compose up -d
```

## License

MIT
