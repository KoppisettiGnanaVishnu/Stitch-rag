
from sentence_transformers import SentenceTransformer


class EmbeddingModel:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)

    def create_embeddings(self, chunks):
        embeddings = []

        for chunk in chunks:
            chunk_text = chunk["text"]

            embedding = self.model.encode(chunk_text)

            embeddings.append(embedding)

        return embeddings

    def create_query_embedding(self, query):
        query_embedding = self.model.encode(query)

        return query_embedding
