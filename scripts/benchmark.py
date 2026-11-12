
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag.config import RAGConfig
from rag.pipeline import RAGPipeline


def main():

    benchmark_path = PROJECT_ROOT / "benchmarks" / "qa.json"

    spec = json.loads(benchmark_path.read_text(encoding="utf-8"))

    document = PROJECT_ROOT / spec["document"]

    config = RAGConfig(index_dir=str(PROJECT_ROOT / "data" / ".index_bench"))

    pipeline = RAGPipeline(document, config=config)

    total = len(spec["cases"])

    retrieval_hits = 0
    answerable = 0
    answered_correctly = 0
    refusal_correct = 0
    unanswerable = 0
    cited = 0

    rows = []

    for case in spec["cases"]:

        start = time.time()

        results = pipeline.retrieve(case["question"])

        context = pipeline.create_context(results)

        answer = pipeline.generator.generate(
            case["question"],
            context
        )

        evaluation = pipeline.evaluator.evaluate_answer(
            case["question"],
            context,
            answer,
            results
        )

        elapsed = time.time() - start

        # Retrieval hit: top-k contains a chunk with an
        # expected keyword (answerable cases only).

        if case["should_answer"]:

            answerable += 1

            hit = any(
                any(
                    keyword.lower() in result["text"].lower()
                    for keyword in case["expected_keywords"]
                )
                for result in results
            )

            if hit:
                retrieval_hits += 1

            if any(
                keyword.lower() in answer.lower()
                for keyword in case["expected_keywords"]
            ):
                answered_correctly += 1

        else:

            unanswerable += 1

            refused = "unavailable" in answer.lower()

            if refused:
                refusal_correct += 1

        if "Source:" in answer:
            cited += 1

        rows.append({
            "id": case["id"],
            "should_answer": case["should_answer"],
            "top_score": round(results[0]["score"], 3) if results else 0.0,
            "verdict": evaluation["verdict"],
            "confidence": evaluation["confidence"],
            "elapsed_s": round(elapsed, 2),
        })

    print("\nStitch benchmark —", document.name)
    print("=" * 78)
    print(f"{'case':<22}{'answerable':<12}{'top_score':<11}{'verdict':<12}{'conf':<8}{'time':<6}")
    print("-" * 78)

    for row in rows:

        print(
            f"{row['id']:<22}"
            f"{str(row['should_answer']):<12}"
            f"{row['top_score']:<11}"
            f"{row['verdict']:<12}"
            f"{row['confidence']:<8}"
            f"{row['elapsed_s']:<6}"
        )

    print("=" * 78)
    print(f"cases:              {total}")
    print(f"retrieval hit@k:    {retrieval_hits}/{answerable} "
          f"({100 * retrieval_hits / answerable:.0f}%)")
    print(f"answer keyword hit: {answered_correctly}/{answerable} "
          f"({100 * answered_correctly / answerable:.0f}%)")
    print(f"refusal accuracy:   {refusal_correct}/{unanswerable} "
          f"({100 * refusal_correct / unanswerable:.0f}%)")
    print(f"citation rate:      {cited}/{total} "
          f"({100 * cited / total:.0f}%)")


if __name__ == "__main__":

    main()
