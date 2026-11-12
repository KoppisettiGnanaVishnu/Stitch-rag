
from pathlib import Path
import sys

# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Add src directory to Python path
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag.pipeline import RAGPipeline


# Documents used for evaluation
documents = [
    PROJECT_ROOT / "data" / "psy-ca1.pdf",
    PROJECT_ROOT / "data" / "sample.txt"
]

source_names = [
    "psy-ca1.pdf",
    "sample.txt"
]


# Initialize RAG pipeline
print("\nInitializing RAG pipeline...")

pipeline = RAGPipeline(
    documents,
    source_name=source_names
)


# Evaluation questions
test_questions = [

    # Answerable questions - psy-ca1.pdf
    {
        "question": "What is procrastination?",
        "expected": "ANSWERABLE"
    },
    {
        "question": "What are the practical strategies to overcome procrastination?",
        "expected": "ANSWERABLE"
    },

    # Answerable questions - sample.txt
    {
        "question": "What are the library timings?",
        "expected": "ANSWERABLE"
    },
    {
        "question": "What is the minimum attendance required for final examinations?",
        "expected": "ANSWERABLE"
    },
    {
        "question": "What should students carry when entering the library?",
        "expected": "ANSWERABLE"
    },
    {
        "question": "When must the examination fee be paid?",
        "expected": "ANSWERABLE"
    },

    # Unanswerable questions
    {
        "question": "What is the annual revenue of the company?",
        "expected": "UNANSWERABLE"
    },
    {
        "question": "Who won the football world cup in 1998?",
        "expected": "UNANSWERABLE"
    },
    {
        "question": "What is the population of India in 2050?",
        "expected": "UNANSWERABLE"
    }
]


# Run retrieval evaluation
print("\nRETRIEVAL CONFIDENCE EVALUATION")
print("=" * 70)

for index, test in enumerate(test_questions, start=1):

    question = test["question"]
    expected = test["expected"]

    # Retrieve relevant passages
    results = pipeline.retrieve(question)

    print("\n" + "-" * 70)
    print(f"Test Question {index}")
    print(f"Question: {question}")
    print(f"Expected: {expected}")

    if not results:
        print("No passages retrieved.")
        continue

    # Display retrieved passages
    for passage_index, result in enumerate(results, start=1):

        print(f"\nPassage {passage_index}")
        print(f"Source: {result['source']}")
        print(f"Page: {result['page']}")
        print(f"Similarity: {result['score']:.4f}")
        print(f"Text: {result['text']}")

    # Calculate retrieval scores
    scores = [
        result["score"]
        for result in results
    ]

    print("\nHighest similarity:", round(max(scores), 4))


# Evaluation completed
print("\n" + "=" * 70)
print("Total questions:", len(test_questions))
print(
    "Answerable questions:",
    sum(1 for q in test_questions if q["expected"] == "ANSWERABLE")
)
print(
    "Unanswerable questions:",
    sum(1 for q in test_questions if q["expected"] == "UNANSWERABLE")
)

print("\nRetrieval evaluation completed.")
