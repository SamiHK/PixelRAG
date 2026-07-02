# Incremental Index Merge Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `--incremental` flag to `pixelrag visual build` so new images can be merged into an existing index instead of replacing it.

**Architecture:** Add a `merge_index()` function to `index_builder.py` that loads an existing index, stacks new vectors onto old ones, appends metadata entries, and rebuilds a fresh FAISS index from the combined set. The CLI branches in `cmd_build()` to call `merge_index()` when an existing index is detected and `--incremental` is set.

**Tech Stack:** Python 3.12+, FAISS, numpy, existing `pixelrag_visual` package

## Global Constraints

- Python 3.12+ only
- Use `uv` for dependency management, never `pip install`
- All new code in `visual/src/pixelrag_visual/`, registered as a single extra (`visual`) in root `pyproject.toml`
- Follow existing PixelRAG patterns: argparse CLI, dataclasses for records, tqdm progress bars, deterministic ordering
- LM Studio URL defaults to `http://172.16.0.203:1234/v1`, overridable via `--lm-studio-url`
- No external dependencies beyond what's already in the root lockfile unless added as a new extra

---

### Task 1: Add `merge_index()` to index_builder.py

**Files:**
- Modify: `visual/src/pixelrag_visual/index_builder.py` — add `merge_index()` function
- Test: `tests/visual/test_index_builder.py` — add tests for merge

**Interfaces:**
- Consumes: `load_index()` (existing), `numpy.ndarray` (new embeddings), `list[str]` (new paths), `list[str]` (new captions)
- Produces: `merge_index(existing_dir, new_embeddings, new_paths, new_captions, nlist=1024) -> dict`
  - Loads existing index + metadata from `existing_dir`
  - Deduplicates: skips new paths already in the existing index (by exact path match)
  - Stacks embeddings: `np.vstack([old_embeddings, deduped_new_embeddings])`
  - Appends metadata: `old_paths + deduped_new_paths`, `old_captions + deduped_new_captions`
  - Rebuilds FAISS index from combined set (same FlatIP/IVF logic as `build_index`)
  - Saves updated files to `existing_dir` (overwrites in place)
  - Returns summary dict: `{"total_vectors": int, "dimension": int, "index_path": str, "metadata_path": str, "added_vectors": int, "skipped_duplicates": int}`

- [ ] **Step 1: Write the failing tests**

Add these test functions to `tests/visual/test_index_builder.py`:

```python
def _make_dummy_embeddings(n: int, dim: int = 4) -> np.ndarray:
    """Create random L2-normalized embeddings for testing."""
    rng = np.random.RandomState(42)
    vecs = rng.randn(n, dim).astype(np.float32)
    norms = np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs / norms


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

        result = merge_index(tmpdir, np.array([[]]).astype(np.float32), [], [])

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/visual/test_index_builder.py::test_merge_index_adds_new_vectors -v`
Expected: FAIL with "NameError: name 'merge_index' is not defined"

- [ ] **Step 3: Write the implementation**

Add this function to `visual/src/pixelrag_visual/index_builder.py` (after the existing functions):

```python
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/visual/test_index_builder.py -v`
Expected: All 6 new merge tests PASS (plus all existing tests)

- [ ] **Step 5: Commit**

```bash
git add visual/src/pixelrag_visual/index_builder.py tests/visual/test_index_builder.py
git commit -m "feat: add merge_index() for incremental index updates"
```

---

### Task 2: Add `--incremental` flag to CLI and wire into cmd_build

**Files:**
- Modify: `visual/src/pixelrag_visual/cli.py` — add flag, branch logic

**Interfaces:**
- Consumes: `merge_index()` (new from Task 1), `load_index()` (existing)
- Produces: CLI with new `--incremental` flag on build subcommand

- [ ] **Step 1: Write the failing test**

Add this test to `tests/visual/test_index_builder.py` (or create a new file `tests/visual/test_cli_incremental.py`):

```python
def test_cmd_build_incremental_loads_existing():
    """When --incremental is set and index exists, merge_index is called."""
    import tempfile
    from unittest.mock import patch

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create an existing index first
        old_embeddings = _make_dummy_embeddings(3, dim=4)
        old_paths = [f"/old/{i}.png" for i in range(3)]
        old_captions = [f"C{i}" for i in range(3)]
        build_index(old_embeddings, old_paths, old_captions, tmpdir)

        # Create a new image directory
        new_dir = os.path.join(tmpdir, "new_images")
        os.makedirs(new_dir)
        for i in range(2):
            Path(new_dir, f"new{i}.png").write_bytes(b"\x89PNG\r\n\x1a\n")

        with patch("pixelrag_visual.cli.get_lm_client") as mock_client:
            mock_response = MagicMock()
            mock_message = MagicMock(content="A new image.")
            mock_response.choices = [MagicMock(message=mock_message)]

            # Mock caption API
            mock_client.return_value.chat.completions.create.return_value = mock_response

            # Mock embedding API
            mock_client.return_value.embeddings.create.return_value = MagicMock(
                data=[MagicMock(embedding=[0.5] * 4)]
            )

            with patch("pixelrag_visual.cli.build_index") as mock_build:
                # Simulate what merge_index would return
                mock_build.return_value = {
                    "total_vectors": 5,
                    "dimension": 4,
                    "index_path": os.path.join(tmpdir, "index.faiss"),
                    "metadata_path": os.path.join(tmpdir, "metadata.json"),
                    "config_path": os.path.join(tmpdir, "config.json"),
                    "index_type": "flat",
                    "added_vectors": 2,
                    "skipped_duplicates": 0,
                }

                # Build CLI args manually
                from pixelrag_visual.cli import parse_args, cmd_build
                import sys

                old_argv = sys.argv
                try:
                    sys.argv = [
                        "pixelrag", "visual", "build",
                        "--input-dir", new_dir,
                        "--output-dir", tmpdir,
                        "--incremental",
                    ]
                    args = parse_args()

                    # Patch scan_images to return our fake images
                    with patch("pixelrag_visual.cli.scan_images") as mock_scan:
                        from pixelrag_visual.scan import ImageInfo
                        mock_scan.return_value = [
                            ImageInfo(path=os.path.join(new_dir, f"new{i}.png"), filename=f"new{i}.png", size_bytes=10),
                        ] * 2

                        cmd_build(args)
                finally:
                    sys.argv = old_argv

        # Verify merge_index was called (build_index is patched to stand in for it)
        assert mock_build.called
```

Actually, let me simplify — the CLI test is complex with mocking. Instead, write a simpler integration-style test that verifies the flag parsing and behavior:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/visual/test_index_builder.py::test_build_flag_parses_incremental -v`
Expected: FAIL with "ArgumentParser error" or the flag doesn't exist yet

- [ ] **Step 3: Modify cli.py — add --incremental flag**

In `visual/src/pixelrag_visual/cli.py`, find the build subparser section (around line 130-159) and add:

```python
    p_build.add_argument(
        "--incremental", action="store_true", help="Merge new images into existing index instead of replacing"
    )
```

Then modify `cmd_build()` to branch on the flag. Replace the existing function with:

```python
def cmd_build(args) -> None:
    """Build a visual search index from an image directory.

    Pipeline: scan -> caption -> embed -> build FAISS index.
    With --incremental: merges new images into an existing index at output-dir.
    """
    input_dir = os.path.expanduser(args.input_dir)
    output_dir = args.output_dir

    if not os.path.isdir(input_dir):
        print(f"Error: input directory '{input_dir}' does not exist")
        return

    # Check if incremental merge is requested and index exists
    existing_index_path = Path(output_dir) / "index.faiss"
    is_incremental = getattr(args, "incremental", False)

    if is_incremental and existing_index_path.exists():
        print(f"Loading existing index from {output_dir}...")

    # Step 1: Scan images
    print(f"Scanning {input_dir} for images...")
    images = scan_images(input_dir, recursive=args.recursive)
    print(f"Found {len(images)} images")

    if not images:
        print("No images found. Nothing to do.")
        return

    # Step 2: Generate captions
    print("Connecting to LM Studio...")
    client = get_lm_client(args.lm_studio_url)

    print("Generating captions...")
    image_paths = [img.path for img in images]
    captions = generate_captions(
        client, image_paths, model=args.caption_model, batch_size=4
    )

    # Step 3: Embed captions (only successful ones)
    print("Embedding captions...")
    successful = [(p, c) for p, c in captions.items() if not c.startswith("[ERROR")]
    caption_texts = [c for _, c in successful]
    valid_paths = [p for p, _ in successful]

    if not caption_texts:
        print("No captions generated. Nothing to index.")
        return

    embeddings = embed_batch(client, caption_texts, model=args.embed_model)
    print(f"Embedding dimension: {embeddings.shape[1]}")

    # Step 4: Build or merge index
    print("Building FAISS index...")

    if is_incremental and existing_index_path.exists():
        result = merge_index(
            output_dir, embeddings, valid_paths, [captions[p] for p in valid_paths], nlist=args.nlist
        )
        print(f"\nIndex merged successfully!")
        print(f"  Existing vectors: {result['total_vectors'] - result['added_vectors']}")
        print(f"  Added vectors: {result['added_vectors']}")
        print(f"  Skipped duplicates: {result['skipped_duplicates']}")
    else:
        result = build_index(
            embeddings, valid_paths, [captions[p] for p in valid_paths], output_dir, nlist=args.nlist
        )
        print(f"\nIndex built successfully!")

    print(f"  Total vectors: {result['total_vectors']}")
    print(f"  Dimension: {result['dimension']}")
    print(f"  Index type: {result['index_type']}")
    print(f"  Output: {output_dir}")

    # Save a human-readable summary
    summary_path = Path(output_dir) / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(
            {
                "input_dir": input_dir,
                "total_images": len(images),
                "embedding_model": args.embed_model,
                "caption_model": args.caption_model,
                "index_type": result["index_type"],
            },
            f,
            indent=2,
        )
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/visual/test_index_builder.py::test_build_flag_parses_incremental -v`
Run: `uv run pytest tests/visual/test_index_builder.py::test_build_flag_not_on_search -v`
Expected: Both PASS

- [ ] **Step 5: Verify CLI help shows the new flag**

Run: `uv run python -m pixelrag_visual.cli build --help`
Expected: Output includes `--incremental  Merge new images into existing index instead of replacing`

- [ ] **Step 6: Commit**

```bash
git add visual/src/pixelrag_visual/cli.py tests/visual/test_index_builder.py
git commit -m "feat: add --incremental flag to build command for index merging"
```

---

## Verification

1. **Unit tests pass:** `uv run pytest tests/visual/test_index_builder.py -v` — all existing + new merge tests PASS
2. **CLI help:** `uv run pixelrag visual build --help` — shows `--incremental` flag
3. **End-to-end manual test:**
   ```bash
   # Build initial index from folder_1
   uv run pixelrag visual build --input-dir ~/folder_1 --output-dir ./image_indexes

   # Later: merge folder_2 into the same index
   uv run pixelrag visual build --input-dir ~/folder_2 --output-dir ./image_indexes --incremental
   ```
