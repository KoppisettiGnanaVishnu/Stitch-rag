
from dataclasses import dataclass


@dataclass
class RAGConfig:
    """
    Central configuration for the RAG pipeline.

    Every value has a sensible default, so the pipeline
    works out of the box. Override anything by passing
    a config instance to RAGPipeline.
    """

    # Chunking

    chunk_size: int = 200
    chunk_overlap: int = 40

    # Retrieval

    top_k: int = 3
    candidate_k: int = 10
    lambda_param: float = 0.7

    # Answer generation (Ollama)

    generator_model: str = "llama3.2:3b"
    ollama_url: str = "http://localhost:11434/api/generate"
    temperature: float = 0.1
    request_timeout: int = 180

    # Embeddings

    embedding_model: str = "all-MiniLM-L6-v2"
