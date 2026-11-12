
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    """
    Sentence embedding model wrapper.

    Encodes documents in one batched call (much faster
    than a per-chunk loop) and queries one at a time.
    """

    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def create_embeddings(self, chunks):
        """
        Encode every chunk's text in a single batched
        call. Returns a list of per-chunk embeddings.
        """

        if not chunks:
            return []

        texts = [chunk["text"] for chunk in chunks]

        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=False
        )

        return list(embeddings)

    def create_query_embedding(self, query):
        query_embedding = self.model.encode(query)

        return query_embedding
