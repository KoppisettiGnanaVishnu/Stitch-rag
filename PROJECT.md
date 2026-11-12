# Stitch — Enterprise Document Intelligence & Knowledge Retrieval Platform

> *Connecting the threads of knowledge.*
>
> **Category:** Retrieval-Augmented Generation (RAG)
> **Stack:** Python · Streamlit · sentence-transformers · Ollama (llama3.2:3b) · pytest
> **Type:** Fully local — no data ever leaves the machine

---

## 1. Project Idea

Stitch is a **grounded document question-answering system**. Users upload their own documents (PDF, DOCX, TXT), and Stitch answers questions **strictly from those documents** — every answer is traced back to the exact source file and page, and every answer reports **how much it should be trusted**.

The core insight: most RAG demos answer questions. **Stitch tells you how much to trust the answer.** That trust layer — confidence scoring, groundedness measurement, hallucination flagging, and citation compliance checking on every single response — is the differentiator.

---

## 2. The Problem

Large Language Models have two failure modes that make them unusable for serious document work:

1. **Hallucination** — the model confidently states things that are not in the source material. For enterprise documents (policies, contracts, academic material), a confident wrong answer is worse than no answer.
2. **No provenance** — when an answer *is* correct, users can't verify it. Without citations, trust is impossible.

Additional practical problems:

- Cloud LLM APIs mean sensitive documents leave the user's machine — a non-starter for compliance contexts.
- Retrieval quality is fragile: naively retrieving the "top 3 most similar passages" often returns three near-duplicates, wasting context on redundant text instead of covering the question from multiple angles.
- RAG systems are typically black boxes — the user gets an answer and nothing else, with no signal about whether retrieval failed, whether the model went off-context, or whether the answer is complete.

---

## 3. The Approach

Stitch processes every question through a fully transparent, inspectable pipeline:

```
Upload (PDF/DOCX/TXT)
        │
        ▼
1. DOCUMENT LOADER ──── pypdf page-by-page extraction; preserves
        │                {text, source filename, page number} metadata
        ▼
2. CHUNKER ──────────── sentence-boundary chunking (never cuts a
        │                sentence in half), word-level splitting for
        ▼                oversized sentences, sentence-based overlap
3. EMBEDDINGS ───────── all-MiniLM-L6-v2 (sentence-transformers)
        │
        ▼
4. RETRIEVER ────────── two strategies:
        │                • similarity search (plain top-k cosine)
        │                • MMR — Maximal Marginal Relevance for diversity
        ▼
5. CONTEXT BUILDER ──── "[Source: file.pdf, Page: N]" + passage text
        │
        ▼
6. GENERATOR ────────── Ollama (llama3.2:3b, temperature 0.1) with a
        │                12-rule grounded prompt: answer ONLY from
        │                context, refuse when evidence is insufficient,
        ▼                cite sources, never invent page numbers
7. EVALUATOR ────────── confidence · groundedness · completeness ·
        │                hallucination checks · citation compliance
        ▼
ANSWER + QUALITY VERDICT (🟢 confident / 🟡 partial / 🔴 unconfident)
```

**Design decisions that matter:**

- **Sentence-boundary chunking** instead of fixed character windows, so passages never begin or end mid-sentence (the prompt explicitly instructs the model not to guess across cut sentences).
- **MMR retrieval** (lambda = 0.7) balances relevance against redundancy: it picks passages that are similar to the query *and dissimilar to each other*.
- **Refusal as a feature**: the system is engineered to say "The information is unavailable in the provided documents" rather than guess — and a test asserts this behavior.
- **Local-first**: embeddings and generation both run on-device via Ollama.

---

## 4. Implementation

**Structure:**

```
app/            Streamlit interface (chat, upload, quality panel)
src/rag/        Core pipeline
  pipeline.py         Orchestrates all stages; ask() and ask_detailed()
  config.py           RAGConfig dataclass — every knob in one place
  document_loader.py  PDF/DOCX/TXT → metadata-aware documents
  chunker.py          Sentence-boundary chunking with overlap
  embeddings.py       sentence-transformers wrapper
  retriever.py        Pure-Python cosine similarity + MMR implementation
  generator.py        Ollama client, grounded prompt, history support
  evaluator.py        Heuristic trust scoring engine
evaluation/     Retrieval benchmarks (normal vs MMR), confidence tests
tests/          pytest suite (evaluator unit tests + pipeline integration)
```

**The evaluator (the differentiator)** scores each answer on three weighted signals:

- **Retrieval confidence (50%)** — is the best retrieved passage actually similar to the question?
- **Groundedness (30%)** — what proportion of the answer's vocabulary is supported by the retrieved context?
- **Completeness (20%)** — does the answer address the query's terms, with sufficient length, without contradicting the context?

It then applies **hallucination heuristics** (unsupported dates/years, unsupported percentages and statistics, contradiction detection) that halve the confidence when triggered, and a **citation compliance check** (are sources cited, and do the cited sources actually exist in the retrieval results?). Output: a 0–1 confidence score and a verdict.

**Conversation awareness:** recent turns are passed to the generator so follow-up questions ("what about the exam fee?") resolve correctly — while the prompt still forbids answering from history instead of documents.

**Persistence:** chat history survives restarts (JSON on disk) and documents are embedded only once — the flat NumPy index (`data/.index/`, keyed by content hash) reloads previously-seen documents instantly.

---

## 5. What Makes It Stand Out (Novelty)

| Most RAG projects | Stitch |
|---|---|
| Answer only | Answer **+ trust verdict + detected issues** |
| Black-box retrieval | Toggleable **similarity vs MMR** strategies in the UI |
| "Trust me" answers | **Source + page citations** rendered under every answer |
| Cloud APIs | **Fully local** — privacy as an architectural principle |
| Dumb refusal or confident hallucination | **Engineered, tested refusal** behavior |
| Hardcoded parameters | Single **RAGConfig** controlling the entire pipeline |
| Generic chat UI | **Nike-inspired premium interface** — floating cards, motion design, stitch motif branding |

The trust layer is also *transparent about its own limits*: the evaluator is explicitly heuristic (keyword overlap, red-flag patterns), not an LLM judge — a deliberate v1 tradeoff for speed and determinism, with LLM-as-judge on the roadmap.

---

## 6. Challenges Faced

1. **Hallucination control** — Off-the-shelf LLMs invent answers. Solved with a strict 12-rule grounded prompt, temperature 0.1, refusal phrasing, and a regression test that asks an unanswerable question and asserts a refusal.
2. **Chunking quality** — Fixed-size windows cut sentences mid-thought, producing passages that begin and end in fragments. Built sentence-boundary chunking with word-level overflow handling and sentence-based overlap.
3. **Redundant retrieval** — Top-k similarity returned near-duplicate passages. Implemented MMR from scratch (cosine similarity matrix, greedy selection with lambda-weighted relevance/penalty term) and an evaluation script comparing both strategies.
4. **Evaluator correctness bugs** — Unit tests immediately caught two real bugs: a percentage regex (`\b\d+\s*%\b`) that could never match because `%` is a non-word character, and greedy citation extraction that flagged valid sources as unsupported. Fixed and regression-tested.
5. **Latency** — Local generation felt slow. Root cause: Ollama reloading the model from disk per request. Fixed with `keep_alive` (model stays resident in memory) and output-length caps; conversation history trimming keeps prompts small.
6. **Streamlit theming** — Streamlit's dark-theme internals silently fought custom CSS (unreadable upload chips, dark alert boxes). Solved by pinning the light theme in `.streamlit/config.toml` and replacing native widgets with palette-exact custom components.
7. **Multi-document metadata** — PDFs carry per-page metadata, DOCX/TXT do not; the loader, chunker, retriever, and UI all had to preserve and display page numbers consistently, including "page unavailable" states.

---

## 7. Technology Stack

| Layer | Technology | Why |
|---|---|---|
| Language | Python 3.11 | Ecosystem for ML/NLP |
| UI | Streamlit + custom CSS | Rapid app development with full design control |
| Embeddings | sentence-transformers `all-MiniLM-L6-v2` | Fast, high-quality, runs locally on CPU |
| Generation | Ollama · `llama3.2:3b` | Fully local inference, no data leaves the machine |
| Document parsing | pypdf, python-docx | Page-level metadata for PDFs |
| Retrieval | Custom implementation | Cosine similarity + MMR written from scratch for transparency |
| Testing | pytest | 5 evaluator unit tests + pipeline integration tests |
| Packaging | uv, pyproject.toml | Reproducible dependency management |

---

## 8. Roadmap

**Done**
- [x] Full RAG pipeline with source + page citations
- [x] Answer quality evaluation (confidence / groundedness / completeness / verdict / issues)
- [x] Dual retrieval strategies (similarity + MMR) with UI toggle
- [x] Chat history persistence + follow-up question awareness
- [x] Streaming-friendly latency fixes (model warm-keeping)
- [x] Live token streaming + persistent index store (flat NumPy vectors, content-hash keyed)
- [x] NumPy-vectorized similarity search (with pure-Python reference fallback)
- [x] Premium UI with motion design and trust panel
- [x] Evaluator test suite + regression coverage

**Next — Retrieval quality**
- [ ] Hybrid retrieval (BM25 + embeddings)
- [ ] Cross-encoder reranking

**Then — Experience**
- [ ] Token streaming into the chat (perceived speed)
- [ ] Conversation memory beyond the last 4 turns
- [ ] Chunk inspector ("see how your document was indexed")
- [ ] Summarize / key-points mode

**Later — Scale & credibility**
- [ ] Repeatable evaluation benchmark with published metrics (precision@k, refusal accuracy)
- [ ] Pluggable LLM providers (OpenAI-compatible endpoints alongside Ollama)
- [ ] Docker Compose one-command deployment
- [ ] FastAPI mode for programmatic access
- [ ] CI (GitHub Actions)

---

## 9. Running It

```bash
uv sync                      # or: pip install -r requirements.txt
ollama pull llama3.2:3b      # local generation model
streamlit run app/main.py    # launch
pytest tests/                # verify
```

Upload documents in the sidebar → ask questions → check the **ANSWER QUALITY** panel and **VIEW SOURCES** under each answer.
