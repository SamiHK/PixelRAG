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


def merge_index(
    existing_dir: str,
    new_embeddings: np.ndarray,
    new_paths: list[str],
    new_captions: list[str],
    nlist: int = 1024,
) -> dict:
    """Merge new image vectors into an existing FAISS index.

    Loads the existing index, deduplicates new paths against it, stacks
    embeddings, and rebuilds a fresh FAISS index from the combined set.

    Args:
        existing_dir: Directory containing an existing built index.
        new_embeddings: M x D float32 array of new L2-normalized vectors.
        new_paths: List of absolute paths for the new images.
        new_captions: List of caption strings for the new images.
        nlist: Number of IVF clusters (ignored for FlatIP).

    Returns:
        Summary dict with merge stats and file paths.

    Raises:
        ValueError: If new embedding dimension doesn't match existing index.
    """
    # Load existing index and metadata
    index, metadata = load_index(existing_dir)

    old_paths = list(metadata["paths"])
    old_captions = list(metadata["captions"])

    # Build a set of existing paths for O(1) dedup
    existing_path_set = set(old_paths)

    # Deduplicate: keep only new paths not already in the index
    dedup_mask = [p not in existing_path_set for p in new_paths]
    filtered_paths = [p for p, keep in zip(new_paths, dedup_mask) if keep]
    filtered_captions = [c for c, keep in zip(new_captions, dedup_mask) if keep]
    filtered_embeddings = new_embeddings[dedup_mask]

    added_vectors = len(filtered_paths)
    skipped_duplicates = len(new_paths) - added_vectors

    # Handle empty merge (all duplicates or no new embeddings)
    if filtered_embeddings.shape[0] == 0:
        return {
            "total_vectors": len(old_paths),
            "dimension": metadata["dimension"],
            "index_path": str(Path(existing_dir) / "index.faiss"),
            "metadata_path": str(Path(existing_dir) / "metadata.json"),
            "config_path": str(Path(existing_dir) / "config.json"),
            "index_type": metadata.get("index_type", "flat"),
            "added_vectors": 0,
            "skipped_duplicates": skipped_duplicates,
        }

    # Check dimension compatibility
    old_dim = metadata["dimension"]
    new_dim = filtered_embeddings.shape[1]
    if old_dim != new_dim:
        raise ValueError(
            f"Embedding dimension mismatch: existing index has {old_dim}D, "
            f"new embeddings are {new_dim}D. Rebuild the index instead."
        )

    # Load existing embeddings from FAISS index for stacking
    old_embeddings = index.reconstruct_batch(np.arange(index.ntotal)).astype(np.float32)

    # Stack all vectors together
    combined_embeddings = np.vstack([old_embeddings, filtered_embeddings])
    combined_paths = old_paths + filtered_paths
    combined_captions = old_captions + filtered_captions

    n = len(combined_paths)
    dim = old_dim

    # Rebuild FAISS index from combined set (same logic as build_index)
    if n < 5000:
        new_index = faiss.IndexFlatIP(dim)
        index_type = "flat"
    else:
        nlist_clamped = min(nlist, n)
        quantizer = faiss.IndexFlatIP(dim)
        new_index = faiss.IndexIVFFlat(quantizer, dim, nlist_clamped, faiss.METRIC_INNER_PRODUCT)
        new_index.train(combined_embeddings)
        index_type = "ivf"

    new_index.add(combined_embeddings)

    # Save updated files
    output_path = Path(existing_dir)

    metadata_out = {
        "paths": combined_paths,
        "captions": combined_captions,
        "dimension": dim,
    }
    metadata_path = output_path / "metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata_out, f, indent=2)

    config = {
        "nlist": nlist if index_type == "ivf" else 0,
        "nprobe": max(1, int(n**0.5)),
        "dimension": dim,
        "total_vectors": n,
        "index_type": index_type,
    }
    config_path = output_path / "config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    index_path = output_path / "index.faiss"
    faiss.write_index(new_index, str(index_path))

    return {
        "total_vectors": n,
        "dimension": dim,
        "index_path": str(index_path),
        "metadata_path": str(metadata_path),
        "config_path": str(config_path),
        "index_type": index_type,
        "added_vectors": added_vectors,
        "skipped_duplicates": skipped_duplicates,
    }
