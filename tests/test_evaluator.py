
import sys
from pathlib import Path

import pytest

# Add src directory to Python path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rag.evaluator import AnswerEvaluator


@pytest.fixture
def evaluator():
    return AnswerEvaluator()


CONTEXT = (
    "[Source: policy.pdf, Page: 1]\n"
    "Students must maintain a minimum attendance of "
    "75% to be eligible to attend the final examinations. "
    "The college library is open from 8 AM to 8 PM on "
    "weekdays."
)

RESULTS = [
    {
        "score": 0.82,
        "text": "Students must maintain a minimum "
                "attendance of 75%.",
        "source": "policy.pdf",
        "page": 1
    },
    {
        "score": 0.61,
        "text": "The college library is open from "
                "8 AM to 8 PM on weekdays.",
        "source": "policy.pdf",
        "page": 1
    }
]


def test_confident_answer_scores_high(evaluator):

    answer = (
        "Students must maintain a minimum attendance of "
        "75% to be eligible to attend the final "
        "examinations. [Source: policy.pdf, Page: 1]"
    )

    evaluation = evaluator.evaluate_answer(
        "What is the minimum attendance?",
        CONTEXT,
        answer,
        RESULTS
    )

    assert evaluation["verdict"] == "confident"
    assert evaluation["confidence"] > 0.7
    assert evaluation["groundedness"] > 0.7


def test_unavailable_answer_is_unconfident(evaluator):

    answer = (
        "The information is unavailable in the "
        "provided documents."
    )

    evaluation = evaluator.evaluate_answer(
        "What is the budget?",
        CONTEXT,
        answer,
        RESULTS
    )

    assert evaluation["verdict"] == "unconfident"
    assert evaluation["confidence"] == 0.0
    assert "issues" in evaluation


def test_unsupported_statistic_is_flagged(evaluator):

    answer = (
        "Students must maintain 95% attendance. "
        "[Source: policy.pdf, Page: 1]"
    )

    evaluation = evaluator.evaluate_answer(
        "What is the minimum attendance?",
        CONTEXT,
        answer,
        RESULTS
    )

    assert any(
        "unsupported" in issue.lower()
        for issue in evaluation["issues"]
    )


def test_missing_citation_is_flagged(evaluator):

    answer = (
        "Students must maintain a minimum attendance "
        "of 75% to be eligible to attend the final "
        "examinations."
    )

    evaluation = evaluator.evaluate_answer(
        "What is the minimum attendance?",
        CONTEXT,
        answer,
        RESULTS
    )

    assert any(
        "citation" in issue.lower()
        for issue in evaluation["issues"]
    )


def test_evaluation_fields_are_bounded(evaluator):

    evaluation = evaluator.evaluate_answer(
        "What is the minimum attendance?",
        CONTEXT,
        "Students must maintain 75% attendance.",
        RESULTS
    )

    for metric in (
        "confidence",
        "groundedness",
        "completeness"
    ):

        assert 0.0 <= evaluation[metric] <= 1.0

    assert evaluation["verdict"] in (
        "confident",
        "partial",
        "unconfident"
    )
