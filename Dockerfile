# syntax=docker/dockerfile:1

FROM python:3.11-slim

# System deps for pypdf / sentence-transformers wheels
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install uv (fast resolver) and sync dependencies from pyproject
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
COPY .streamlit ./.streamlit

# Create the data directory (chat history + vector index live here)
RUN mkdir -p /app/data

# Install deps (no dev group in production image)
RUN uv sync --no-dev --no-install-project \
    && uv pip install --system -e . --no-deps

ENV PYTHONPATH=/app/src
ENV STITCH_OLLAMA_URL=http://ollama:11434/api/generate
ENV STITCH_INDEX_DIR=/app/data/.index

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["uv", "run", "streamlit", "run", "app/main.py", \
     "--server.port=8501", "--server.address=0.0.0.0", \
     "--server.headless=true"]
