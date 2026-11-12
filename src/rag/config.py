
import os
from dataclasses import dataclass


@dataclass
class RAGConfig:
    """
    Central configuration for the RAG pipeline.

    Every value has a sensible default, so the pipeline
    works out of the box. Override anything by passing
    a config instance to RAGPipeline.

    Values can also be set via environment variables
    (STITCH_*) which take precedence over defaults —
    this is how the Docker image picks up the Ollama
    sidecar URL without code changes.
    """

    # Chunking

    chunk_size: int = 200
    chunk_overlap: int = 40

    # Retrieval

    top_k: int = 3
    candidate_k: int = 10
    lambda_param: float = 0.7

    # Answer generation (Ollama)

    generator_model: str = os.environ.get(
        "STITCH_GENERATOR_MODEL", "llama3.2:3b"
    )
    ollama_url: str = os.environ.get(
        "STITCH_OLLAMA_URL",
        "http://localhost:11434/api/generate"
    )
    temperature: float = float(os.environ.get(
        "STITCH_TEMPERATURE", "0.1"
    ))
    request_timeout: int = int(os.environ.get(
        "STITCH_REQUEST_TIMEOUT", "180"
    ))

    # Embeddings

    embedding_model: str = os.environ.get(
        "STITCH_EMBEDDING_MODEL", "all-MiniLM-L6-v2"
    )

    # Persistence — local flat vector store directory.

    index_dir: str = os.environ.get(
        "STITCH_INDEX_DIR", "data/.index"
    )
