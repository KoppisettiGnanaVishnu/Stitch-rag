---
title: Stitch — Document Intelligence
emoji: 🧵
colorFrom: red
colorTo: black
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: RAG that answers strictly from your documents — with citations and trust scores
---

# Stitch on Hugging Face Spaces

Upload a PDF / DOCX / TXT and ask questions. Every answer cites its
source file and page, and comes with a confidence, groundedness and
completeness score.

- **Retrieval:** all-MiniLM-L6-v2 + MMR (written from scratch)
- **Generation:** llama3.2:3b via Ollama (baked into this image)
- **Fully self-contained:** Ollama runs inside the container

Free CPU tier ⇒ answers take ~30–60 s. Be patient with the first
question after the Space wakes up.
