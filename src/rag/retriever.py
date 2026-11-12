
import math

import numpy as np


class Retriever:
    """
    Similarity search over chunk embeddings.

    Fast path: when the chunk embeddings form a NumPy
    matrix (the production case), all similarities are
    computed with one vectorized matrix operation.

    Fallback path: the original pure-Python loop, kept
    deliberately so the math stays readable and to run
    equivalence tests against the fast path.
    """

    def __init__(self, chunks, chunk_embeddings):

        self.chunks = chunks
        self.chunk_embeddings = chunk_embeddings

        # Vectorized view, built lazily on first use.
        # None means the fallback path is active.

        self._matrix = None
        self._norms = None

        self._prepare_matrix()

    # ------------------------------------------
    # Vectorized index preparation
    # ------------------------------------------

    def _prepare_matrix(self):

        if not self.chunk_embeddings:
            return

        first = self.chunk_embeddings[0]

        if not isinstance(first, np.ndarray):
            return

        self._matrix = np.asarray(
            self.chunk_embeddings,
            dtype="float32"
        )

        norms = np.linalg.norm(self._matrix, axis=1)

        # Zero-norm rows would divide by 0 later;
        # keep them but pin their norm to 1 so
        # their similarity simply becomes 0.

        norms[norms == 0] = 1.0

        self._norms = norms

    def _matrix_available(self):
        return self._matrix is not None

    # ------------------------------------------
    # Similarity primitives
    # ------------------------------------------

    def cosine_similarity(self, A, B):
        """Reference implementation kept for clarity."""

        dot_product = 0

        for a, b in zip(A, B):
            dot_product = dot_product + (a * b)

        magnitude_A = 0
        for a in A:
            magnitude_A = magnitude_A + (a * a)
        magnitude_A = math.sqrt(magnitude_A)

        magnitude_B = 0
        for b in B:
            magnitude_B = magnitude_B + (b * b)
        magnitude_B = math.sqrt(magnitude_B)

        if magnitude_A == 0 or magnitude_B == 0:
            return 0.0

        similarity = dot_product / (magnitude_A * magnitude_B)

        return similarity

    def _query_similarities(self, query_embedding):
        """
        Cosine similarity of the query against every
        chunk. Vectorized when possible, loop otherwise.
        """

        if self._matrix_available():

            query = np.asarray(
                query_embedding,
                dtype="float32"
            ).ravel()

            query_norm = np.linalg.norm(query)

            if query_norm == 0:
                return np.zeros(
                    self._matrix.shape[0],
                    dtype="float32"
                )

            dots = self._matrix @ query

            similarities = dots / (self._norms * query_norm)

            return similarities

        # Pure-Python fallback

        similarities = []

        for chunk_embedding in self.chunk_embeddings:

            score = self.cosine_similarity(
                query_embedding,
                chunk_embedding
            )

            similarities.append(score)

        return similarities

    def _max_redundancy(self, embedding, selected_indices):
        """
        Highest cosine similarity between one embedding
        and a set of already-selected embeddings.
        """

        if self._matrix_available():

            vector = np.asarray(
                embedding,
                dtype="float32"
            ).ravel()

            vector_norm = np.linalg.norm(vector)

            if vector_norm == 0:
                return 0.0

            dots = self._matrix[list(selected_indices)] @ vector

            similarities = dots / (
                self._norms[list(selected_indices)] * vector_norm
            )

            return float(np.max(similarities))

        # Pure-Python fallback

        max_redundancy = 0

        for index in selected_indices:

            selected_embedding = self.chunk_embeddings[index]

            similarity = self.cosine_similarity(
                embedding,
                selected_embedding
            )

            max_redundancy = max(
                max_redundancy,
                similarity
            )

        return max_redundancy

    # ------------------------------------------
    # Search
    # ------------------------------------------

    def search(self, query_embedding, top_k=2):
        """
        Plain top-k similarity search.
        """

        similarities = self._query_similarities(
            query_embedding
        )

        # Enumerate, sort by score, take top_k.
        # Works for both NumPy arrays and lists.

        scored = [
            {
                "index": i,
                "score": float(score)
            }
            for i, score in enumerate(similarities)
        ]

        scored.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        results = []

        for item in scored[:top_k]:

            chunk = self.chunks[item["index"]]

            results.append({
                "score": item["score"],
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"]
            })

        return results

    def search_mmr(
        self,
        query_embedding,
        top_k=3,
        candidate_k=10,
        lambda_param=0.7
    ):
        """
        Maximal Marginal Relevance: pick chunks that are
        relevant to the query AND diverse between
        themselves, instead of near-duplicates.
        """

        if not 0 <= lambda_param <= 1:
            raise ValueError("lambda_param must be between 0 and 1")

        # Step 1: Calculate query similarity for every chunk

        similarities = self._query_similarities(
            query_embedding
        )

        candidates = [
            {
                "index": i,
                "score": float(score)
            }
            for i, score in enumerate(similarities)
        ]

        # Step 2: Keep the most relevant candidate chunks

        candidates.sort(
            key=lambda item: item["score"],
            reverse=True
        )

        candidates = candidates[:max(candidate_k, top_k)]

        if not candidates:
            return []

        selected = []

        selected_indices = []

        # Step 3: Select the best relevant chunk first

        first = candidates[0]
        selected.append(first)
        selected_indices.append(first["index"])

        # Step 4: Select remaining chunks using MMR

        while len(selected) < min(top_k, len(candidates)):

            best_candidate = None
            best_mmr_score = float("-inf")

            for candidate in candidates:

                if candidate["index"] in selected_indices:
                    continue

                candidate_embedding = self.chunk_embeddings[
                    candidate["index"]
                ]

                # Find maximum similarity with already
                # selected chunks (vectorized)

                max_redundancy = self._max_redundancy(
                    candidate_embedding,
                    selected_indices
                )

                # MMR formula

                mmr_score = (
                    lambda_param * candidate["score"]
                    - (1 - lambda_param) * max_redundancy
                )

                if mmr_score > best_mmr_score:
                    best_mmr_score = mmr_score
                    best_candidate = candidate

            if best_candidate is None:
                break

            best_candidate["mmr_score"] = float(best_mmr_score)

            selected.append(best_candidate)
            selected_indices.append(best_candidate["index"])

        # Step 5: Return selected chunks with their metadata

        results = []

        for item in selected:

            chunk = self.chunks[item["index"]]

            result = {
                "score": item["score"],
                "text": chunk["text"],
                "source": chunk["source"],
                "page": chunk["page"]
            }

            if "mmr_score" in item:
                result["mmr_score"] = item["mmr_score"]

            results.append(result)

        return results
