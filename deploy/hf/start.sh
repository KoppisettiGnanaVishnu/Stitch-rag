#!/bin/bash
# Start Ollama in the background, wait until it is
# ready, then launch Streamlit on HF's required port.

set -e

ollama serve &

# Wait for the Ollama API to come up

until curl -s http://127.0.0.1:11434/api/tags >/dev/null 2>&1; do
    sleep 1
done

exec uv run streamlit run app/main.py \
    --server.port 7860 \
    --server.address 0.0.0.0 \
    --server.headless=true
