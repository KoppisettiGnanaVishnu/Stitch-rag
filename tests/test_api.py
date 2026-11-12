
from pathlib import Path

import pytest

from fastapi.testclient import TestClient

from rag.api import create_app, _PIPELINES


@pytest.fixture
def client():

    _PIPELINES.clear()

    app = create_app()

    return TestClient(app)


def test_health(client):

    response = client.get("/health")

    assert response.status_code == 200

    assert response.json()["status"] == "ok"


def test_index_requires_files(client):

    response = client.post("/index")

    assert response.status_code == 422


def test_index_and_ask_end_to_end(client, monkeypatch):

    sample = Path(__file__).resolve().parents[1] / "data" / "sample.txt"

    if not sample.is_file():

        pytest.skip("data/sample.txt not available")

    # Avoid needing a running Ollama during tests

    from rag import generator as generator_module

    def fake_generate(self, query, context, history=None):

        return "Canned answer from the test."

    monkeypatch.setattr(
        generator_module.Generator,
        "generate",
        fake_generate,
    )

    with sample.open("rb") as handle:

        response = client.post(
            "/index",
            files={"files": ("sample.txt", handle, "text/plain")},
        )

    assert response.status_code == 200

    payload = response.json()

    assert payload["chunks"] > 0

    assert "sample.txt" in payload["sources"]

    index_id = payload["index_id"]

    ask = client.post(
        "/ask",
        json={
            "index_id": index_id,
            "question": "What are the library timings?",
            "strategy": "similarity",
        },
    )

    assert ask.status_code == 200

    body = ask.json()

    assert body["answer"] == "Canned answer from the test."

    assert body["results"]

    assert "evaluation" in body


def test_ask_unknown_index_returns_404(client):

    response = client.post(
        "/ask",
        json={
            "index_id": "does-not-exist",
            "question": "hi",
        },
    )

    assert response.status_code == 404


def test_delete_index(client):

    sample = Path(__file__).resolve().parents[1] / "data" / "sample.txt"

    if not sample.is_file():

        pytest.skip("data/sample.txt not available")

    with sample.open("rb") as handle:

        response = client.post(
            "/index",
            files={"files": ("sample.txt", handle, "text/plain")},
        )

    index_id = response.json()["index_id"]

    delete = client.delete(f"/index/{index_id}")

    assert delete.status_code == 200

    # Second delete is a 404

    assert client.delete(f"/index/{index_id}").status_code == 404
