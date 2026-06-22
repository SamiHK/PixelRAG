import json
import tempfile
from pathlib import Path

import numpy as np
import faiss

from pixelrag_visual.index_builder import build_index, load_index, search_index


def _make_dummy_embeddings(n: int, dim: int = 4) -> np.ndarray:
    """Create random L2-normalized embeddings for testing."""
    rng = np.random.RandomState(42)
    vecs = rng.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


def test_build_index_saves_files():
    with tempfile.TemporaryDirectory() as tmpdir:
        embeddings = _make_dummy_embeddings(10, dim=4)
        paths = [f"/img/{i}.png" for i in range(10)]
        captions = [f"Caption {i}" for i in range(10)]

        result = build_index(embeddings, paths, captions, tmpdir)

        assert "index.faiss" in result["index_path"]
        assert "metadata.json" in result["metadata_path"]
        assert Path(result["index_path"]).exists()
        assert Path(result["metadata_path"]).exists()


def test_build_index_stores_metadata():
    with tempfile.TemporaryDirectory() as tmpdir:
        embeddings = _make_dummy_embeddings(3, dim=4)
        paths = ["/a.png", "/b.png", "/c.png"]
        captions = ["A cat.", "A dog.", "A bird."]

        build_index(embeddings, paths, captions, tmpdir)

        with open(Path(tmpdir) / "metadata.json") as f:
            meta = json.load(f)

        assert len(meta["paths"]) == 3
        assert meta["paths"][0] == "/a.png"
        assert len(meta["captions"]) == 3
        assert meta["captions"][1] == "A dog."


def test_load_index_returns_matching_data():
    with tempfile.TemporaryDirectory() as tmpdir:
        embeddings = _make_dummy_embeddings(5, dim=4)
        paths = [f"/img/{i}.png" for i in range(5)]
        captions = [f"C{i}" for i in range(5)]

        build_index(embeddings, paths, captions, tmpdir)
        index, meta = load_index(tmpdir)

    assert len(meta["paths"]) == 5
    assert index.ntotal == 5


def test_search_index_returns_ranked_results():
    with tempfile.TemporaryDirectory() as tmpdir:
        embeddings = _make_dummy_embeddings(10, dim=4)
        paths = [f"/img/{i}.png" for i in range(10)]
        captions = [f"C{i}" for i in range(10)]

        build_index(embeddings, paths, captions, tmpdir)
        index, meta = load_index(tmpdir)

    query = _make_dummy_embeddings(1)[0]
    results = search_index(index, meta, query, k=3)

    assert len(results) == 3
    assert all(k in results[0] for k in ("score", "path", "caption"))
    assert results[0]["score"] >= results[1]["score"] >= results[2]["score"]


def test_search_index_with_flat_vs_ivf():
    """Verify FlatIP is used for small datasets, IVF for larger."""
    with tempfile.TemporaryDirectory() as tmpdir:
        embeddings = _make_dummy_embeddings(5, dim=4)
        paths = [f"/img/{i}.png" for i in range(5)]
        captions = [f"C{i}" for i in range(5)]

        result = build_index(embeddings, paths, captions, tmpdir)
        assert result["index_type"] == "flat"

    with tempfile.TemporaryDirectory() as tmpdir:
        embeddings = _make_dummy_embeddings(5000, dim=4)
        paths = [f"/img/{i}.png" for i in range(5000)]
        captions = [f"C{i}" for i in range(5000)]

        result = build_index(embeddings, paths, captions, tmpdir)
        assert result["index_type"] == "ivf"


def test_search_index_respects_k_parameter():
    with tempfile.TemporaryDirectory() as tmpdir:
        embeddings = _make_dummy_embeddings(20, dim=4)
        paths = [f"/img/{i}.png" for i in range(20)]
        captions = [f"C{i}" for i in range(20)]

        build_index(embeddings, paths, captions, tmpdir)
        index, meta = load_index(tmpdir)

    query = _make_dummy_embeddings(1)[0]
    results = search_index(index, meta, query, k=5)
    assert len(results) == 5

    results = search_index(index, meta, query, k=100)
    assert len(results) == 20  # capped at total vectors
