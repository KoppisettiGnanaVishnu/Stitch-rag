
from rag.pipeline import RAGPipeline


QUESTIONS = [
    "What is procrastination?",
    "What are the psychological causes of procrastination?",
    "How does procrastination affect stress and anxiety?",
    "What are effective time management strategies?",
    "What are the consequences of delaying tasks?"
]


def calculate_average_score(results):
    if not results:
        return 0.0

    total = sum(result["score"] for result in results)

    return total / len(results)


def evaluate_retrieval(pipeline, query):
    query_embedding = pipeline.embedding_model.create_query_embedding(
        query
    )

    normal_results = pipeline.retriever.search(
        query_embedding,
        top_k=3
    )

    mmr_results = pipeline.retriever.search_mmr(
        query_embedding,
        top_k=3,
        candidate_k=10,
        lambda_param=0.7
    )

    normal_pages = {result["page"] for result in normal_results}
    mmr_pages = {result["page"] for result in mmr_results}

    print("\n" + "=" * 65)
    print("QUESTION:", query)

    print("\nNORMAL RETRIEVAL")
    print("Pages:", [r["page"] for r in normal_results])
    print("Average similarity:",
          round(calculate_average_score(normal_results), 4))
    print("Unique pages:", len(normal_pages))

    print("\nMMR RETRIEVAL")
    print("Pages:", [r["page"] for r in mmr_results])
    print("Average similarity:",
          round(calculate_average_score(mmr_results), 4))
    print("Unique pages:", len(mmr_pages))

    print("\nCOMPARISON")
    print("Common pages:", sorted(normal_pages & mmr_pages))
    print("Normal-only pages:", sorted(normal_pages - mmr_pages))
    print("MMR-only pages:", sorted(mmr_pages - normal_pages))


def main():
    pipeline = RAGPipeline("data/psy-ca1.pdf")

    print("\nSTITCH RETRIEVAL EVALUATION")
    print("Document:", pipeline.file_path)
    print("Total chunks:", len(pipeline.chunks))

    for query in QUESTIONS:
        evaluate_retrieval(pipeline, query)


if __name__ == "__main__":
    main()
