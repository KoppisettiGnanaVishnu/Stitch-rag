
from rag.pipeline import RAGPipeline

pipeline = RAGPipeline("data/psy-ca1.pdf")

questions = [
    "What is procrastination?",
    "What are the psychological causes of procrastination?",
    "How does procrastination affect stress and anxiety?",
    "What are effective time management strategies?",
    "What are the consequences of delaying tasks?"
]

for query in questions:

    query_embedding = pipeline.embedding_model.create_query_embedding(query)

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

    normal_pages = [r["page"] for r in normal_results]
    mmr_pages = [r["page"] for r in mmr_results]

    print("\n" + "=" * 60)
    print("QUESTION:", query)

    print("Similarity pages:", normal_pages)
    print("MMR pages:       ", mmr_pages)

    print("\nMMR selected passages:")

    for result in mmr_results:
        print(
            "Page:", result["page"],
            "| Similarity:", round(result["score"], 4),
            "| MMR:", round(result.get("mmr_score", 0), 4)
        )
