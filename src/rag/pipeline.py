
from pathlib import Path

from rag.config import RAGConfig
from rag.document_loader import load_document_with_metadata
from rag.chunker import create_chunks
from rag.embeddings import EmbeddingModel
from rag.retriever import Retriever
from rag.generator import Generator
from rag.evaluator import AnswerEvaluator


class RAGPipeline:

    def __init__(self, file_path, source_name=None, config=None):

        self.config = config or RAGConfig()

        # Step 1: Support single and multiple documents
        if isinstance(file_path, (str, Path)):
            self.file_paths = [file_path]
        else:
            self.file_paths = list(file_path)

        # Step 2: Validate document paths
        if not self.file_paths:
            raise ValueError(
                "Please provide at least one document."
            )

        for path in self.file_paths:
            if not Path(path).is_file():
                raise FileNotFoundError(
                    f"Document not found: {path}"
                )

        # Step 3: Normalize source names
        if source_name is None:
            source_names = [None] * len(self.file_paths)

        elif isinstance(source_name, str):
            source_names = [source_name]

        else:
            source_names = list(source_name)

        if len(source_names) != len(self.file_paths):
            raise ValueError(
                "The number of source names must match "
                "the number of files."
            )

        self.file_path = (
            self.file_paths[0]
            if len(self.file_paths) == 1
            else self.file_paths
        )

        # Step 4: Load all documents
        self.documents = []

        for path, name in zip(self.file_paths, source_names):

            loaded_documents = load_document_with_metadata(path)

            # Preserve original filenames
            if name:
                for document in loaded_documents:
                    document["source"] = name

            self.documents.extend(loaded_documents)

        # Step 5: Divide all documents into chunks
        self.chunks = create_chunks(
            self.documents,
            chunk_size=self.config.chunk_size,
            chunk_overlap=self.config.chunk_overlap
        )

        if not self.chunks:
            raise ValueError(
                "No readable text was found in the documents."
            )

        # Step 6: Generate embeddings
        self.embedding_model = EmbeddingModel(
            self.config.embedding_model
        )

        self.chunk_embeddings = (
            self.embedding_model.create_embeddings(
                self.chunks
            )
        )

        # Step 7: Initialize retriever
        self.retriever = Retriever(
            self.chunks,
            self.chunk_embeddings
        )

        # Step 8: Initialize answer generator
        self.generator = Generator(
            model=self.config.generator_model
        )

        # Step 9: Initialize answer evaluator
        self.evaluator = AnswerEvaluator()

    # ==========================================
    # NORMAL SIMILARITY RETRIEVAL
    # ==========================================

    def retrieve(self, query):

        query_embedding = (
            self.embedding_model.create_query_embedding(
                query
            )
        )

        results = self.retriever.search(
            query_embedding,
            top_k=self.config.top_k
        )

        return results

    # ==========================================
    # MAXIMUM MARGINAL RELEVANCE RETRIEVAL
    # ==========================================

    def retrieve_mmr(self, query):

        query_embedding = (
            self.embedding_model.create_query_embedding(
                query
            )
        )

        results = self.retriever.search_mmr(
            query_embedding,
            top_k=self.config.top_k,
            candidate_k=self.config.candidate_k,
            lambda_param=self.config.lambda_param
        )

        return results

    # ==========================================
    # CREATE CONTEXT WITH SOURCE CITATIONS
    # ==========================================

    def create_context(self, results):

        context_parts = []

        for result in results:

            source = result["source"]
            page = result["page"]
            text = result["text"]

            if page is not None:
                citation = (
                    f"Source: {source}, Page: {page}"
                )
            else:
                citation = f"Source: {source}"

            context_part = (
                f"[{citation}]\n"
                f"{text}"
            )

            context_parts.append(context_part)

        return "\n\n".join(context_parts)

    # ==========================================
    # COMPLETE QUESTION-ANSWERING PIPELINE
    # ==========================================

    def ask(self, query):

        results = self.retrieve(query)

        context = self.create_context(results)

        answer = self.generator.generate(
            query,
            context
        )

        return answer

    # ==========================================
    # QUESTION-ANSWERING WITH FULL DETAILS
    # ==========================================

    def ask_detailed(self, query):

        results = self.retrieve(query)

        context = self.create_context(results)

        answer = self.generator.generate(
            query,
            context
        )

        evaluation = self.evaluator.evaluate_answer(
            query,
            context,
            answer,
            results
        )

        return {
            "answer": answer,
            "results": results,
            "context": context,
            "evaluation": evaluation
        }
