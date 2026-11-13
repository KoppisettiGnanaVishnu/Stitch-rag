# syntax=docker/dockerfile:1
#
# Self-contained image for Hugging Face Spaces (free CPU tier).
# Ollama runs INSIDE the container (HF gives you a single
# container), and the LLM is baked into the image at build
# time so restarts don't re-download 2 GB.

FROM python:3.11-slim

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Install the Ollama runtime

RUN curl -fsSL https://ollama.com/install.sh | sh

WORKDIR /app

COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /usr/local/bin/

COPY pyproject.toml README.md ./
COPY src ./src
COPY app ./app
COPY .streamlit ./.streamlit

RUN mkdir -p /app/data

RUN uv sync --no-dev --no-install-project \
    && uv pip install --system -e . --no-deps

# Bake the LLM into the image: start ollama briefly,
# pull the model, stop. The model then lives in
# /root/.ollama inside this layer.

ARG GENERATOR_MODEL=llama3.2:3b

RUN ollama serve & \
    sleep 5 \
    && ollama pull ${GENERATOR_MODEL} \
    && pkill ollama

ENV PYTHONPATH=/app/src
ENV STITCH_OLLAMA_URL=http://127.0.0.1:11434/api/generate
ENV STITCH_INDEX_DIR=/app/data/.index
ENV STITCH_GENERATOR_MODEL=${GENERATOR_MODEL}

# HF Spaces requires the app on port 7860

EXPOSE 7860

COPY start.sh /app/start.sh
RUN chmod +x /app/start.sh

CMD ["/app/start.sh"]
