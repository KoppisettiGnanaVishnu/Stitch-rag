
from rag.pipeline import RAGPipeline

pipeline = RAGPipeline("data/psy-ca1.pdf")

query = "What are the consequences of delaying tasks?"

# Test 1: Normal similarity retrieval
normal_results = pipeline.retrieve(query)
normal_context = pipeline.create_context(normal_results)

# Test 2: MMR retrieval
mmr_results = pipeline.retrieve_mmr(query)
mmr_context = pipeline.create_context(mmr_results)

print("\n========== NORMAL RETRIEVAL ==========")

for result in normal_results:
    print("Source:", result["source"], "| Page:", result["page"])

normal_answer = pipeline.generator.generate(
    query,
    normal_context
)

print("\nNORMAL ANSWER:")
print(normal_answer)

print("\n========== MMR RETRIEVAL ==========")

for result in mmr_results:
    print("Source:", result["source"], "| Page:", result["page"])

mmr_answer = pipeline.generator.generate(
    query,
    mmr_context
)

print("\nMMR ANSWER:")
print(mmr_answer)
