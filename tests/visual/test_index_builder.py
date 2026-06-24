import json
import tempfile
from pathlib import Path

import numpy as np
import pytest
import faiss

from pixelrag_visual.index_builder import build_index, load_index, merge_index, search_index


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


def test_merge_index_adds_new_vectors():
    """Merging 3 new vectors into a 5-vector index yields 8 total."""

    with tempfile.TemporaryDirectory() as tmpdir:
        # Build initial index with 5 vectors
        old_embeddings = _make_dummy_embeddings(5, dim=4)
        old_paths = [f"/img/{i}.png" for i in range(5)]
        old_captions = [f"C{i}" for i in range(5)]
        build_index(old_embeddings, old_paths, old_captions, tmpdir)

        # Merge 3 new vectors with different paths
        new_embeddings = _make_dummy_embeddings(3, dim=4)
        new_paths = [f"/img/{i}.png" for i in range(5, 8)]
        new_captions = [f"C{i}" for i in range(5, 8)]

        result = merge_index(tmpdir, new_embeddings, new_paths, new_captions)

        assert result["total_vectors"] == 8
        assert result["added_vectors"] == 3

        # Verify loaded index reflects merged state
        index, metadata = load_index(tmpdir)
        assert index.ntotal == 8
        assert len(metadata["paths"]) == 8
        assert metadata["paths"][5] == "/img/5.png"


def test_merge_index_skips_duplicate_paths():
    """If a new path already exists in the index, it is skipped."""

    with tempfile.TemporaryDirectory() as tmpdir:
        old_embeddings = _make_dummy_embeddings(3, dim=4)
        old_paths = ["/a.png", "/b.png", "/c.png"]
        old_captions = ["A.", "B.", "C."]
        build_index(old_embeddings, old_paths, old_captions, tmpdir)

        # All new paths overlap with existing
        new_embeddings = _make_dummy_embeddings(3, dim=4)
        new_paths = ["/a.png", "/b.png", "/d.png"]  # /d.png is new
        new_captions = ["D1.", "D2.", "D3."]

        result = merge_index(tmpdir, new_embeddings, new_paths, new_captions)

        assert result["total_vectors"] == 4
        assert result["added_vectors"] == 1
        assert result["skipped_duplicates"] == 2


def test_merge_index_rejects_dimension_mismatch():
    """Merging embeddings with wrong dimension raises ValueError."""

    with tempfile.TemporaryDirectory() as tmpdir:
        old_embeddings = _make_dummy_embeddings(3, dim=4)
        old_paths = ["/a.png", "/b.png", "/c.png"]
        old_captions = ["A.", "B.", "C."]
        build_index(old_embeddings, old_paths, old_captions, tmpdir)

        new_embeddings = _make_dummy_embeddings(2, dim=8)  # wrong dim
        new_paths = ["/d.png", "/e.png"]
        new_captions = ["D.", "E."]

        with pytest.raises(ValueError, match="dimension"):
            merge_index(tmpdir, new_embeddings, new_paths, new_captions)


def test_merge_index_empty_new_set():
    """Merging zero vectors is a no-op that returns added_vectors=0."""

    with tempfile.TemporaryDirectory() as tmpdir:
        old_embeddings = _make_dummy_embeddings(3, dim=4)
        old_paths = ["/a.png", "/b.png", "/c.png"]
        old_captions = ["A.", "B.", "C."]
        build_index(old_embeddings, old_paths, old_captions, tmpdir)

        result = merge_index(tmpdir, np.empty((0, 4), dtype=np.float32), [], [])

        assert result["total_vectors"] == 3
        assert result["added_vectors"] == 0


def test_merge_index_updates_config():
    """After merge, config.json reflects the new total_vectors."""

    with tempfile.TemporaryDirectory() as tmpdir:
        old_embeddings = _make_dummy_embeddings(2, dim=4)
        old_paths = ["/a.png", "/b.png"]
        old_captions = ["A.", "B."]
        build_index(old_embeddings, old_paths, old_captions, tmpdir)

        new_embeddings = _make_dummy_embeddings(3, dim=4)
        new_paths = ["/c.png", "/d.png", "/e.png"]
        new_captions = ["C.", "D.", "E."]

        merge_index(tmpdir, new_embeddings, new_paths, new_captions)

        with open(Path(tmpdir) / "config.json") as f:
            config = json.load(f)

        assert config["total_vectors"] == 5


def test_merge_index_preserves_captions():
    """Merged captions are correctly appended and retrievable."""

    with tempfile.TemporaryDirectory() as tmpdir:
        old_embeddings = _make_dummy_embeddings(2, dim=4)
        old_paths = ["/a.png", "/b.png"]
        old_captions = ["Old A.", "Old B."]
        build_index(old_embeddings, old_paths, old_captions, tmpdir)

        new_embeddings = _make_dummy_embeddings(2, dim=4)
        new_paths = ["/c.png", "/d.png"]
        new_captions = ["New C.", "New D."]

        merge_index(tmpdir, new_embeddings, new_paths, new_captions)

        _, metadata = load_index(tmpdir)
        assert metadata["captions"] == ["Old A.", "Old B.", "New C.", "New D."]


def test_build_flag_parses_incremental():
    """The --incremental flag is parsed correctly on the build subcommand."""
    from pixelrag_visual.cli import parse_args

    args = parse_args(["build", "--input-dir", "/img", "--output-dir", "/idx", "--incremental"])
    assert args.incremental is True

    args = parse_args(["build", "--input-dir", "/img", "--output-dir", "/idx"])
    assert args.incremental is False


def test_build_flag_not_on_search():
    """The --incremental flag only exists on build, not search."""
    from pixelrag_visual.cli import parse_args

    args = parse_args(["search", "--index-dir", "/idx", "--query", "hello"])
    assert not hasattr(args, "incremental") or args.incremental is None
