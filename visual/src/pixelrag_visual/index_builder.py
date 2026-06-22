"""Build a FAISS IVF/Flat index from image embeddings and metadata."""

import json
from pathlib import Path

import faiss
import numpy as np


def build_index(
    embeddings: np.ndarray,
    paths: list[str],
    captions: list[str],
    output_dir: str,
    nlist: int = 1024,
) -> dict:
    """Build FAISS index and save to disk.

    Auto-selects index type: IndexFlatIP for N < 5000, IndexIVFFlat otherwise.

    Args:
        embeddings: N x D float32 array (L2-normalized).
        paths: List of absolute image file paths.
        captions: List of caption strings.
        output_dir: Directory to write index files.
        nlist: Number of IVF clusters (ignored for FlatIP).

    Returns:
        Summary dict with index stats and file paths.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    n, dim = embeddings.shape

    if n < 5000:
        index = faiss.IndexFlatIP(dim)
        index_type = "flat"
    else:
        nlist = min(nlist, n)
        quantizer = faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
        index.train(embeddings)
        index_type = "ivf"

    index.add(embeddings)

    metadata = {
        "paths": paths,
        "captions": captions,
        "dimension": dim,
    }
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)

    config = {
        "nlist": nlist,
        "nprobe": max(1, int(n**0.5)),
        "dimension": dim,
        "total_vectors": n,
        "index_type": index_type,
    }
    config_path = output_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    index_path = output_path / "index.faiss"
    faiss.write_index(index, str(index_path))

    return {
        "total_vectors": n,
        "dimension": dim,
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
        "config_path": str(config_path),
        "index_type": index_type,
    }


def load_index(index_dir: str) -> tuple[faiss.Index, dict]:
    """Load a previously built index and its metadata.

    Args:
        index_dir: Directory containing index.faiss, metadata.json, config.json.

    Returns:
        Tuple of (faiss.Index, metadata_dict with 'paths' and 'captions').
    """
    index_path = Path(index_dir) / "index.faiss"
    metadata_path = Path(index_dir) / "metadata.json"

    index = faiss.read_index(str(index_path))
    with open(metadata_path) as f:
        metadata = json.load(f)

    return index, metadata


def search_index(
    index: faiss.Index,
    metadata: dict,
    query_vec: np.ndarray,
    k: int = 10,
) -> list[dict]:
    """Search the index and return ranked results.

    Args:
        index: Loaded FAISS index.
        metadata: Metadata dict with 'paths' and 'captions'.
        query_vec: L2-normalized float32 query vector, shape (D,) or (1, D).
        k: Number of results to return.

    Returns:
        List of dicts sorted by descending score, each with 'score', 'path', 'caption'.
    """
    if query_vec.ndim == 1:
        query_vec = query_vec.reshape(1, -1)

    scores, indices = index.search(query_vec, min(k, index.ntotal))

    results: list[dict] = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue

        results.append(
            {
                "score": float(score),
                "path": metadata["paths"][idx],
                "caption": metadata["captions"][idx],
            }
        )

    return results
