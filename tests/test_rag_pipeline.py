
from pathlib import Path

from rag.pipeline import RAGPipeline


PDF_PATH = Path("data/psy-ca1.pdf")


def test_retrieval_returns_source_metadata():
    pipeline = RAGPipeline(str(PDF_PATH))

    results = pipeline.retrieve(
        "What is this assignment about?"
    )

    assert len(results) > 0

    for result in results:
        assert "text" in result
        assert "score" in result
        assert "source" in result
        assert "page" in result
        assert result["source"] == PDF_PATH.name
        assert isinstance(result["page"], int)


def test_context_includes_source_and_page():
    pipeline = RAGPipeline(str(PDF_PATH))

    results = pipeline.retrieve(
        "What is this assignment about?"
    )

    context = pipeline.create_context(results)

    assert "Source: psy-ca1.pdf" in context
    assert "Page:" in context


def test_known_answer():
    pipeline = RAGPipeline(str(PDF_PATH))

    answer = pipeline.ask(
        "What is procrastination?"
    )

    assert "procrastination" in answer.lower()
    assert "delay" in answer.lower()



def test_unknown_answer_does_not_invent_budget():
    pipeline = RAGPipeline(str(PDF_PATH))

    answer = pipeline.ask(
        "What is the exact budget allocated for this assignment?"
    )

    answer_lower = answer.lower()

    refusal_phrases = [
        "unavailable",
        "not specified",
        "not provided",
        "does not provide information",
        "no information",
        "cannot find",
        "not mentioned",
        "not available",
    ]

    assert any(
        phrase in answer_lower
        for phrase in refusal_phrases
    ), f"Expected an uncertainty response, got: {answer}"
