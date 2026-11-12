
import json
from pathlib import Path

import numpy as np
import pytest

from rag.index_store import IndexStore, fingerprint_source
from rag.retriever import Retriever


# ==================================================
# FINGERPRINTING
# ==================================================


def test_fingerprint_is_content_sensitive(tmp_path):

    first = fingerprint_source("doc.txt", b"hello")

    second = fingerprint_source("doc.txt", b"hellp")

    assert first != second


def test_fingerprint_is_name_sensitive():

    assert (
        fingerprint_source("a.txt", b"data")
        != fingerprint_source("b.txt", b"data")
    )


def test_fingerprint_is_deterministic():

    assert (
        fingerprint_source("doc.txt", b"same")
        == fingerprint_source("doc.txt", b"same")
    )


def test_fingerprint_respects_extra_settings():

    base = fingerprint_source("doc.txt", b"same")

    with_overlap = fingerprint_source(
        "doc.txt",
        b"same",
        extra=b"200:40"
    )

    assert base != with_overlap


# ==================================================
# INDEX STORE PERSISTENCE
# ==================================================


def test_store_roundtrip(tmp_path):

    store = IndexStore(tmp_path / "index")

    chunks = [
        {"text": "alpha", "source": "doc.txt", "page": 1},
        {"text": "beta", "source": "doc.txt", "page": 2},
    ]

    embeddings = [
        np.array([1.0, 0.0, 0.0], dtype="float32"),
        np.array([0.0, 1.0, 0.0], dtype="float32"),
    ]

    store.put("entry1", chunks, embeddings)

    # A second store instance reads from disk only —
    # proving real persistence.

    fresh_store = IndexStore(tmp_path / "index")

    cached_chunks, cached_embeddings = fresh_store.get("entry1")

    assert cached_chunks == chunks

    assert np.allclose(
        np.asarray(cached_embeddings),
        np.asarray(embeddings)
    )


def test_store_missing_entry_returns_none(tmp_path):

    store = IndexStore(tmp_path / "index")

    assert store.get("never-stored") is None


def test_store_corrupt_entry_returns_none(tmp_path):

    index_dir = tmp_path / "index"

    store = IndexStore(index_dir)

    entry_hash = "broken"

    (index_dir / f"entry_{entry_hash}.json").write_text(
        "not json at all{",
        encoding="utf-8"
    )

    assert store.get(entry_hash) is None


def test_store_prune_evicts_oldest(tmp_path):

    store = IndexStore(tmp_path / "index")

    # Fill the store beyond its cap (MAX_ENTRIES = 32)

    for i in range(34):

        chunks = [{"text": f"c{i}", "source": "s", "page": None}]

        embeddings = [np.array([float(i)], dtype="float32")]

        store.put(f"entry_{i:02d}", chunks, embeddings)

    # The first two entries are now oldest and must
    # be evicted once pruning runs.

    active = {f"entry_{i:02d}" for i in range(30, 34)}

    store.prune(active)

    assert store.get("entry_00") is None
    assert store.get("entry_01") is None

    # Recently used entries survive

    assert store.get("entry_33") is not None


def test_store_manifest_tracks_entries(tmp_path):

    store = IndexStore(tmp_path / "index")

    store.put(
        "tracked",
        [{"text": "t", "source": "s", "page": None}],
        [np.array([1.0], dtype="float32")]
    )

    manifest = json.loads(
        (tmp_path / "index" / "manifest.json").read_text(
            encoding="utf-8"
        )
    )

    assert manifest["entries"]["tracked"]["chunks"] == 1


# ==================================================
# RETRIEVER: FAST PATH == FALLBACK PATH
# ==================================================


def _sample_corpus():

    chunks = [
        {"text": f"chunk {i}", "source": "s.txt", "page": i + 1}
        for i in range(8)
    ]

    embeddings = []

    for i in range(8):

        vector = np.zeros(6, dtype="float32")

        vector[i % 6] = 1.0

        vector[(i + 1) % 6] = 0.5 + 0.1 * i

        embeddings.append(vector)

    return chunks, embeddings


def test_fast_path_and_fallback_agree():

    chunks, embeddings = _sample_corpus()

    fast = Retriever(chunks, embeddings)

    assert fast._matrix_available()

    # Force the fallback path

    slow = Retriever(chunks, embeddings)

    slow._matrix = None

    query = np.array([1.0, 0.2, 0.1, 0.0, 0.0, 0.0], dtype="float32")

    fast_results = fast.search(query, top_k=3)

    slow_results = slow.search(query, top_k=3)

    assert [r["text"] for r in fast_results] == (
        [r["text"] for r in slow_results]
    )

    for a, b in zip(fast_results, slow_results):
        assert a["score"] == pytest.approx(b["score"], abs=1e-5)


def test_mmr_fast_and_fallback_agree():

    chunks, embeddings = _sample_corpus()

    fast = Retriever(chunks, embeddings)

    slow = Retriever(chunks, embeddings)

    slow._matrix = None

    query = np.array([1.0, 0.2, 0.1, 0.0, 0.0, 0.0], dtype="float32")

    fast_results = fast.search_mmr(
        query,
        top_k=4,
        candidate_k=6,
        lambda_param=0.7
    )

    slow_results = slow.search_mmr(
        query,
        top_k=4,
        candidate_k=6,
        lambda_param=0.7
    )

    assert [r["text"] for r in fast_results] == (
        [r["text"] for r in slow_results]
    )

    assert fast_results[0]["text"] == slow_results[0]["text"]


def test_search_with_python_list_embeddings():

    # Pure-Python lists must work too (fallback path
    # selected automatically).

    chunks = [
        {"text": "a", "source": "s", "page": 1},
        {"text": "b", "source": "s", "page": 2},
    ]

    embeddings = [
        [1.0, 0.0],
        [0.0, 1.0],
    ]

    retriever = Retriever(chunks, embeddings)

    assert not retriever._matrix_available()

    results = retriever.search([1.0, 0.1], top_k=1)

    assert results[0]["text"] == "a"


# ==================================================
# PIPELINE CACHE INTEGRATION
# ==================================================


def test_pipeline_reuses_cache_on_second_build(tmp_path):

    # Small config so chunking stays deterministic and
    # light; index into an isolated temp directory.

    from rag.config import RAGConfig
    from rag.pipeline import RAGPipeline

    sample = Path(__file__).resolve().parents[1] / "data" / "sample.txt"

    if not sample.is_file():
        pytest.skip("data/sample.txt not available")

    config = RAGConfig(index_dir=str(tmp_path / "index"))

    first = RAGPipeline(sample, config=config)

    # Cache hit must actually skip the encoding work:
    # make encoding fatal and rebuild from the store.

    from rag import embeddings as embeddings_module

    def _explode(self, chunks):
        raise AssertionError("cache hit should not re-encode")

    original = embeddings_module.EmbeddingModel.create_embeddings
    embeddings_module.EmbeddingModel.create_embeddings = _explode

    try:
        second = RAGPipeline(sample, config=config)
    finally:
        embeddings_module.EmbeddingModel.create_embeddings = original

    assert second.cache_hits == [sample.name]

    assert len(second.chunks) == len(first.chunks)

    assert np.allclose(
        np.asarray(second.chunk_embeddings),
        np.asarray(first.chunk_embeddings),
        atol=1e-5
    )


def test_pipeline_results_usable_after_cache_hit(tmp_path):

    from rag.config import RAGConfig
    from rag.pipeline import RAGPipeline

    sample = Path(__file__).resolve().parents[1] / "data" / "sample.txt"

    if not sample.is_file():
        pytest.skip("data/sample.txt not available")

    config = RAGConfig(index_dir=str(tmp_path / "index"))

    RAGPipeline(sample, config=config)

    warm = RAGPipeline(sample, config=config)

    results = warm.retrieve("library timings")

    assert results

    assert all("source" in result for result in results)
