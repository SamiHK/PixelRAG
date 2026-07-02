# PixelRAG Visual — Personal Photo Search

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `pixelrag visual` subcommand with `build`, `search`, and optional `serve` stages so users can index a directory of personal photos (e.g. `~/Desktop/image_dataset`) and search them by natural language description ("smiling people", "cars") using LM Studio models.

**Architecture:** New `visual/src/pixelrag_visual/` subpackage following PixelRAG's existing pattern. Pipeline: scan images → generate visual captions via VL model (LM Studio) → embed caption text as vectors (LM Studio `/v1/embeddings`) → build FAISS index. Search: embed query text → FAISS lookup → return ranked images with captions. The matching is text-to-text in vector space — your query vs. VL-generated captions. Images are attached as results.

**Tech Stack:** Python 3.12+, FAISS, OpenAI SDK (LM Studio compatibility), tqdm

## Global Constraints

- Python 3.12+ only
- Use `uv` for dependency management, never `pip install`
- All new code in `visual/src/pixelrag_visual/`, registered as a single extra (`visual`) in root `pyproject.toml`
- Follow existing PixelRAG patterns: argparse CLI, dataclasses for records, tqdm progress bars, deterministic ordering
- LM Studio URL defaults to `http://172.16.0.203:1234/v1`, overridable via `--lm-studio-url`
- No external dependencies beyond what's already in the root lockfile unless added as a new extra

---

## File Structure

```
visual/src/pixelrag_visual/
├── __init__.py              # empty (package marker)
├── scan.py                  # ImageInfo dataclass + scan_images()
├── caption.py               # generate_caption() via LM Studio VL model
├── embed_client.py          # embed_text() via LM Studio /v1/embeddings
├── index_builder.py         # build_index(), load_index(), search_index()
├── cli.py                   # argparse: build + search subcommands
tests/visual/
├── test_scan.py             # Tests for scan_images()
├── test_embed_client.py     # Tests for embed_text/embed_batch (mocked)
└── test_index_builder.py    # Tests for FAISS build/search (in-memory)
```

---

### Task 1: Package Scaffold + scan.py

**Files:**
- Create: `visual/src/pixelrag_visual/__init__.py`
- Create: `visual/src/pixelrag_visual/scan.py`
- Test: `tests/visual/test_scan.py`

**Interfaces:**
- Produces: `ImageInfo(path: str, filename: str, size_bytes: int)` frozen dataclass
- Produces: `scan_images(directory: str, recursive: bool = True) -> list[ImageInfo]`

- [ ] **Step 1: Create package scaffold**

Create `visual/src/pixelrag_visual/__init__.py` as an empty file (follows existing PixelRAG pattern).

- [ ] **Step 2: Write the failing test**

```python
# tests/visual/test_scan.py
import os
import tempfile
from pathlib import Path

from pixelrag_visual.scan import scan_images, ImageInfo


def test_scan_images_finds_png_and_jpg():
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "photo1.png").write_bytes(b"fake")
        Path(tmpdir, "photo2.jpg").write_bytes(b"fake")
        Path(tmpdir, "readme.txt").write_bytes(b"ignore me")

    result = scan_images(tmpdir)
    assert len(result) == 2
    paths = {r.path for r in result}
    assert any("photo1.png" in p for p in paths)
    assert any("photo2.jpg" in p for p in paths)


def test_scan_images_ignores_non_image_extensions():
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "a.png").write_bytes(b"")
        Path(tmpdir, "b.jpg").write_bytes(b"")
        Path(tmpdir, "c.jpeg").write_bytes(b"")
        Path(tmpdir, "d.txt").write_bytes(b"")
        Path(tmpdir, "e.pdf").write_bytes(b"")

    result = scan_images(tmpdir)
    assert len(result) == 3


def test_scan_images_recursive():
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "top.png").write_bytes(b"")
        subdir = Path(tmpdir, "sub")
        subdir.mkdir()
        (subdir / "nested.png").write_bytes(b"")

    result = scan_images(tmpdir, recursive=True)
    assert len(result) == 2


def test_scan_images_non_recursive():
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "top.png").write_bytes(b"")
        subdir = Path(tmpdir, "sub")
        subdir.mkdir()
        (subdir / "nested.png").write_bytes(b"")

    result = scan_images(tmpdir, recursive=False)
    assert len(result) == 1


def test_scan_images_deterministic_order():
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "z.png").write_bytes(b"")
        Path(tmpdir, "a.png").write_bytes(b"")
        Path(tmpdir, "m.png").write_bytes(b"")

    result = scan_images(tmpdir)
    assert [r.filename for r in result] == ["a.png", "m.png", "z.png"]


def test_image_info_is_frozen():
    info = ImageInfo(path="/tmp/x.png", filename="x.png", size_bytes=100)
    try:
        info.path = "/tmp/y.png"
        assert False, "should have raised"
    except AttributeError:
        pass  # expected on frozen dataclass


def test_scan_images_returns_absolute_paths():
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "x.png").write_bytes(b"")

    result = scan_images(tmpdir)
    assert os.path.isabs(result[0].path)
```

- [ ] **Step 3: Run test to verify it fails**

Run: `uv run pytest tests/visual/test_scan.py -v`
Expected: FAIL with "ModuleNotFoundError: No module named 'pixelrag_visual'"

- [ ] **Step 4: Write minimal implementation**

```python
# visual/src/pixelrag_visual/scan.py
"""Scan a directory for image files."""

import os
from dataclasses import dataclass


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


@dataclass(frozen=True)
class ImageInfo:
    """Metadata for a single image file."""

    path: str           # absolute path to the image file
    filename: str       # basename
    size_bytes: int     # file size


def scan_images(directory: str, recursive: bool = True) -> list[ImageInfo]:
    """Walk directory and collect image files.

    Args:
        directory: Path to scan.
        recursive: If True, recurse into subdirectories.

    Returns:
        Sorted list of ImageInfo (sorted by path for deterministic ordering).
    """
    results: list[ImageInfo] = []

    if recursive:
        iterator = os.walk(directory)
    else:
        files = os.listdir(directory)
        iterator = [(directory, [], files)]

    for dirpath, _, filenames in iterator:
        for fname in filenames:
            if os.path.splitext(fname)[1].lower() in IMAGE_EXTENSIONS:
                full_path = os.path.join(dirpath, fname)
                results.append(
                    ImageInfo(
                        path=os.path.abspath(full_path),
                        filename=fname,
                        size_bytes=os.path.getsize(full_path),
                    )
                )

    results.sort(key=lambda r: r.path)
    return results
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/visual/test_scan.py -v`
Expected: All 6 tests PASS

- [ ] **Step 6: Commit**

```bash
git add visual/src/pixelrag_visual/__init__.py visual/src/pixelrag_visual/scan.py tests/visual/test_scan.py
git commit -m "feat: add visual package scaffold and scan_images()"
```

---

### Task 2: embed_client.py — LM Studio Embedding Client

**Files:**
- Create: `visual/src/pixelrag_visual/embed_client.py`
- Test: `tests/visual/test_embed_client.py`

**Interfaces:**
- Produces: `get_lm_client(base_url: str | None = None) -> openai.OpenAI`
- Produces: `embed_text(client, text: str, model: str = "text-embedding-mxbai-embed-large-v1") -> numpy.ndarray`
  - Returns: L2-normalized float32 vector of shape `(D,)` where D is the model's embedding dimension
- Produces: `embed_batch(client, texts: list[str], model: str = "text-embedding-mxbai-embed-large-v1", batch_size: int = 16) -> numpy.ndarray`
  - Returns: stacked array of shape `(N, D)`

- [ ] **Step 1: Write the failing test**

```python
# tests/visual/test_embed_client.py
import numpy as np
from unittest.mock import MagicMock

from pixelrag_visual.embed_client import get_lm_client, embed_text, embed_batch


def test_get_lm_client_default_url():
    client = get_lm_client()
    assert str(client.base_url) == "http://172.16.0.203:1234/v1"


def test_get_lm_client_custom_url():
    client = get_lm_client("http://localhost:1234/v1")
    assert str(client.base_url) == "http://localhost:1234/v1"


def test_embed_text_returns_normalized_vector():
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[1.0, 2.0, 3.0])]
    )

    result = embed_text(mock_client, "test query", model="dummy-model")

    assert isinstance(result, np.ndarray)
    assert result.dtype == np.float32
    expected_norm = np.sqrt(np.sum(np.array([1.0, 2.0, 3.0]) ** 2))
    expected_normalized = np.array([1.0, 2.0, 3.0]) / expected_norm
    np.testing.assert_allclose(result, expected_normalized, atol=1e-6)


def test_embed_text_calls_correct_api():
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[0.1, 0.2])]
    )

    embed_text(mock_client, "hello world", model="test-model")

    mock_client.embeddings.create.assert_called_once()
    call_kwargs = mock_client.embeddings.create.call_args[1]
    assert call_kwargs["model"] == "test-model"
    assert call_kwargs["input"] == "hello world"


def test_embed_batch_stacks_vectors():
    mock_client = MagicMock()
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[1.0, 0.0]), MagicMock(embedding=[0.0, 1.0])]
    )

    result = embed_batch(mock_client, ["a", "b"], model="test-model")

    assert result.shape == (2, 2)
    np.testing.assert_array_equal(result[0], [1.0, 0.0])
    np.testing.assert_array_equal(result[1], [0.0, 1.0])


def test_embed_batch_normalizes_each_vector():
    mock_client = MagicMock()
    # Return unnormalized vectors [3, 4] and [5, 12]
    mock_client.embeddings.create.return_value = MagicMock(
        data=[MagicMock(embedding=[3.0, 4.0]), MagicMock(embedding=[5.0, 12.0])]
    )

    result = embed_batch(mock_client, ["a", "b"], model="test-model")

    # [3, 4] has norm 5 → normalized = [0.6, 0.8]
    # [5, 12] has norm 13 → normalized = [5/13, 12/13]
    np.testing.assert_allclose(result[0], [0.6, 0.8], atol=1e-6)
    np.testing.assert_allclose(result[1], [5 / 13, 12 / 13], atol=1e-6)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/visual/test_embed_client.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write minimal implementation**

```python
# visual/src/pixelrag_visual/embed_client.py
"""Embed text via LM Studio's OpenAI-compatible /v1/embeddings endpoint."""

import numpy as np
from openai import OpenAI


DEFAULT_LM_STUDIO_URL = "http://172.16.0.203:1234/v1"


def get_lm_client(base_url: str | None = None) -> OpenAI:
    """Create an OpenAI client configured for LM Studio.

    Args:
        base_url: LM Studio API base URL. Defaults to DEFAULT_LM_STUDIO_URL.

    Returns:
        Configured OpenAI client instance.
    """
    url = base_url or DEFAULT_LM_STUDIO_URL
    return OpenAI(base_url=url, api_key="not-needed")


def embed_text(
    client: OpenAI, text: str, model: str = "text-embedding-mxbai-embed-large-v1"
) -> np.ndarray:
    """Embed a single text string via LM Studio.

    Args:
        client: OpenAI-compatible client configured for LM Studio.
        text: Text to embed.
        model: Embedding model name (e.g. 'text-embedding-mxbai-embed-large-v1').

    Returns:
        L2-normalized float32 embedding vector.
    """
    response = client.embeddings.create(model=model, input=text)
    vec = np.array(response.data[0].embedding, dtype=np.float32)
    norm = np.linalg.norm(vec)
    if norm > 0:
        vec = vec / norm
    return vec


def embed_batch(
    client: OpenAI,
    texts: list[str],
    model: str = "text-embedding-mxbai-embed-large-v1",
    batch_size: int = 16,
) -> np.ndarray:
    """Embed multiple text strings via LM Studio.

    Args:
        client: OpenAI-compatible client configured for LM Studio.
        texts: List of text strings to embed.
        model: Embedding model name.
        batch_size: Number of texts per API call (LM Studio may have limits).

    Returns:
        Stacked float32 array of shape (N, D), L2-normalized.
    """
    all_vectors: list[np.ndarray] = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=model, input=batch)

        for item in response.data:
            vec = np.array(item.embedding, dtype=np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            all_vectors.append(vec)

    return np.stack(all_vectors, axis=0)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/visual/test_embed_client.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add visual/src/pixelrag_visual/embed_client.py tests/visual/test_embed_client.py
git commit -m "feat: add LM Studio embedding client for text vectors"
```

---

### Task 3: caption.py — VL Model Captioning via LM Studio

**Files:**
- Create: `visual/src/pixelrag_visual/caption.py`
- Test: `tests/visual/test_caption.py`

**Interfaces:**
- Produces: `encode_image_b64(image_path: str) -> str` — returns base64 data URL string
- Produces: `generate_caption(client, image_path, model="qwen/qwen3-vl-30b", system_prompt=None) -> str`
  - Sends image as base64 to LM Studio VL model via `/v1/chat/completions`
  - Returns caption string extracted from model response
- Produces: `generate_captions(client, image_paths, model, batch_size=4) -> dict[str, str]` — path → caption mapping

- [ ] **Step 1: Write the failing test**

```python
# tests/visual/test_caption.py
import base64
from unittest.mock import MagicMock, patch

from pixelrag_visual.caption import encode_image_b64, generate_caption


def test_encode_image_b64_returns_data_url():
    with patch("builtins.open") as mock_open:
        mock_file = MagicMock()
        mock_file.read.return_value = b"\x89PNG\r\n\x1a\n"
        mock_open.__enter__.return_value = mock_file

    result = encode_image_b64("/fake/photo.png")

    assert result.startswith("data:image/png;base64,")
    decoded = base64.b64decode(result.split(",", 1)[1])
    assert decoded == b"\x89PNG\r\n\x1a\n"


def test_encode_image_b64_jpeg_mime():
    with patch("builtins.open") as mock_open:
        mock_file = MagicMock()
        mock_file.read.return_value = b"\xff\xd8\xff\xe0fake"
        mock_open.__enter__.return_value = mock_file

    result = encode_image_b64("/fake/photo.jpg")

    assert result.startswith("data:image/jpeg;base64,")


def test_generate_caption_extracts_text_from_response():
    mock_client = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "A group of people smiling at the camera in a park."

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=mock_message)]
    mock_client.chat.completions.create.return_value = mock_response

    result = generate_caption(
        mock_client, "/fake/photo.png", model="test-vl-model"
    )

    assert result == "A group of people smiling at the camera in a park."
    mock_client.chat.completions.create.assert_called_once()


def test_generate_caption_uses_base64_image_format():
    mock_client = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "A red car."

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=mock_message)]
    mock_client.chat.completions.create.return_value = mock_response

    with patch("pixelrag_visual.caption.encode_image_b64") as mock_encode:
        mock_encode.return_value = "data:image/png;base64,SGVsbG8="
        generate_caption(mock_client, "/fake/photo.png", model="test-vl-model")

    call_args = mock_client.chat.completions.create.call_args[1]
    messages = call_args["messages"]

    user_msg = [m for m in messages if m["role"] == "user"][0]
    content = user_msg["content"]

    assert isinstance(content, list)
    assert any(part.get("type") == "text" for part in content)
    assert any(part.get("type") == "image_url" for part in content)


def test_generate_caption_default_system_prompt():
    mock_client = MagicMock()
    mock_message = MagicMock()
    mock_message.content = "A sunset."

    mock_response = MagicMock()
    mock_response.choices = [MagicMock(message=mock_message)]
    mock_client.chat.completions.create.return_value = mock_response

    with patch("pixelrag_visual.caption.encode_image_b64"):
        generate_caption(mock_client, "/fake/photo.png")

    call_args = mock_client.chat.completions.create.call_args[1]
    system_msg = [m for m in call_args["messages"] if m["role"] == "system"][0]

    assert "visual" in system_msg["content"].lower()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/visual/test_caption.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write minimal implementation**

```python
# visual/src/pixelrag_visual/caption.py
"""Generate visual notes/captions using a VL model from LM Studio."""

import base64
from pathlib import Path


DEFAULT_SYSTEM_PROMPT = (
    "You are a visual analyst. Describe the key visual elements in this image "
    "concisely (1-2 sentences). Focus on objects, people, actions, colors, and "
    "scene composition. Do NOT mention any text in the image."
)


def encode_image_b64(image_path: str) -> str:
    """Read an image file and return a base64 data URL.

    Args:
        image_path: Path to the image file.

    Returns:
        Base64 data URL string (e.g. 'data:image/png;base64,...').
    """
    ext = Path(image_path).suffix.lower()
    mime_type = "image/png" if ext == ".png" else "image/jpeg"

    with open(image_path, "rb") as f:
        encoded = base64.b64encode(f.read()).decode("utf-8")

    return f"data:{mime_type};base64,{encoded}"


def generate_caption(
    client,
    image_path: str,
    model: str = "qwen/qwen3-vl-30b",
    system_prompt: str | None = None,
) -> str:
    """Send image to LM Studio VL model and return caption.

    Args:
        client: OpenAI-compatible client configured for LM Studio.
        image_path: Path to the image file.
        model: VL model name (must support vision).
        system_prompt: Optional custom system prompt.

    Returns:
        Caption string from the model's response.
    """
    prompt = system_prompt or DEFAULT_SYSTEM_PROMPT
    image_url = encode_image_b64(image_path)

    messages = [
        {"role": "system", "content": prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe the visual content of this image."},
                {"type": "image_url", "image_url": {"url": image_url}},
            ],
        },
    ]

    response = client.chat.completions.create(model=model, messages=messages)
    return response.choices[0].message.content


def generate_captions(
    client,
    image_paths: list[str],
    model: str = "qwen/qwen3-vl-30b",
    batch_size: int = 4,
) -> dict[str, str]:
    """Generate captions for multiple images.

    Args:
        client: OpenAI-compatible client configured for LM Studio.
        image_paths: List of image file paths.
        model: VL model name.
        batch_size: Not used for chat API (kept for future batching).

    Returns:
        Mapping of image path to caption string.
    """
    from tqdm import tqdm

    captions: dict[str, str] = {}
    for path in tqdm(image_paths, desc="Generating captions"):
        try:
            captions[path] = generate_caption(client, path, model=model)
        except Exception as e:
            captions[path] = f"[ERROR: {e}]"

    return captions
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/visual/test_caption.py -v`
Expected: All 5 tests PASS

- [ ] **Step 5: Commit**

```bash
git add visual/src/pixelrag_visual/caption.py tests/visual/test_caption.py
git commit -m "feat: add VL model captioning via LM Studio chat completions"
```

---

### Task 4: index_builder.py — FAISS Index Build, Load, Search

**Files:**
- Create: `visual/src/pixelrag_visual/index_builder.py`
- Test: `tests/visual/test_index_builder.py`

**Interfaces:**
- Consumes: `numpy.ndarray` (embeddings), `list[str]` (paths), `list[str]` (captions)
- Produces: `build_index(embeddings, paths, captions, output_dir, nlist=1024) -> dict`
  - Saves `index.faiss`, `metadata.json`, `config.json` to output_dir
  - Returns summary dict: `{"total_vectors": int, "dimension": int, "index_path": str, "metadata_path": str}`
- Produces: `load_index(index_dir) -> tuple[faiss.Index, dict]` — returns FAISS index + dict with paths/captions
- Produces: `search_index(index, metadata, query_vec, k=10) -> list[dict]`
  - Returns: `[{"score": float, "path": str, "caption": str}, ...]` sorted by descending score

- [ ] **Step 1: Write the failing test**

```python
# tests/visual/test_index_builder.py
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
        embeddings = _make_dummy_embeddings(100, dim=4)
        paths = [f"/img/{i}.png" for i in range(100)]
        captions = [f"C{i}" for i in range(100)]

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/visual/test_index_builder.py -v`
Expected: FAIL with "ModuleNotFoundError"

- [ ] **Step 3: Write minimal implementation**

```python
# visual/src/pixelrag_visual/index_builder.py
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
        index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_IP)
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
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/visual/test_index_builder.py -v`
Expected: All 6 tests PASS

- [ ] **Step 5: Commit**

```bash
git add visual/src/pixelrag_visual/index_builder.py tests/visual/test_index_builder.py
git commit -m "feat: add FAISS index builder with auto FlatIP/IVF selection"
```

---

### Task 5: cli.py — Build + Search Subcommands

**Files:**
- Create: `visual/src/pixelrag_visual/cli.py`

**Interfaces:**
- Consumes: all modules above (`scan`, `caption`, `embed_client`, `index_builder`)
- Produces: CLI with two subcommands:
  - `pixelrag visual build --input-dir DIR --output-dir DIR` — full pipeline
  - `pixelrag visual search --index-dir DIR --query "text"` — search

- [ ] **Step 1: Write the implementation**

```python
# visual/src/pixelrag_visual/cli.py
"""CLI for visual image search: build and query a FAISS index."""

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI

from pixelrag_visual.scan import scan_images
from pixelrag_visual.caption import generate_captions
from pixelrag_visual.embed_client import get_lm_client, embed_batch
from pixelrag_visual.index_builder import build_index, load_index, search_index


def cmd_build(args) -> None:
    """Build a visual search index from an image directory.

    Pipeline: scan -> caption -> embed -> build FAISS index.
    """
    input_dir = os.path.expanduser(args.input_dir)
    output_dir = args.output_dir

    if not os.path.isdir(input_dir):
        print(f"Error: input directory '{input_dir}' does not exist")
        return

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

    # Step 3: Embed captions
    print("Embedding captions...")
    caption_texts = [captions.get(p, "") for p in image_paths]
    embeddings = embed_batch(client, caption_texts, model=args.embed_model)
    print(f"Embedding dimension: {embeddings.shape[1]}")

    # Step 4: Build index
    print("Building FAISS index...")
    result = build_index(
        embeddings, image_paths, list(captions.values()), output_dir, nlist=args.nlist
    )

    print(f"\nIndex built successfully!")
    print(f"  Vectors: {result['total_vectors']}")
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


def cmd_search(args) -> None:
    """Search the visual index by text query."""
    index_dir = args.index_dir

    # Step 1: Load index
    print(f"Loading index from {index_dir}...")
    index, metadata = load_index(index_dir)

    # Load config for nprobe
    config_path = Path(index_dir) / "config.json"
    if config_path.exists():
        with open(config_path) as f:
            config = json.load(f)
    else:
        config = {"nprobe": 16}

    # Step 2: Embed query
    print("Embedding query...")
    client = get_lm_client(args.lm_studio_url)
    query_vec = embed_batch(client, [args.query], model=args.embed_model)[0]

    # Step 3: Search
    print(f"Searching for '{args.query}'...")
    results = search_index(index, metadata, query_vec, k=args.k)

    # Step 4: Display results
    print(f"\nTop {len(results)} results:\n")
    for i, r in enumerate(results, 1):
        print(f"  {i}. [{r['score']:.4f}] {r['path']}")
        print(f"     {r['caption']}\n")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="pixelrag visual",
        description=(
            "Visual image search: build and query a FAISS index from personal photos."
        ),
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # build subparser
    p_build = subparsers.add_parser(
        "build", help="Build visual search index from an image directory"
    )
    p_build.add_argument(
        "--input-dir", required=True, help="Directory containing images (PNG/JPG)"
    )
    p_build.add_argument(
        "--output-dir", required=True, help="Output directory for the index"
    )
    p_build.add_argument(
        "--lm-studio-url",
        default=None,
        help=f"LM Studio API URL (default: {get_lm_client().base_url})",
    )
    p_build.add_argument(
        "--caption-model",
        default="qwen/qwen3-vl-30b",
        help="VL model for captions",
    )
    p_build.add_argument(
        "--embed-model",
        default="text-embedding-mxbai-embed-large-v1",
        help="Embedding model",
    )
    p_build.add_argument(
        "--nlist", type=int, default=1024, help="FAISS IVF clusters (auto for small datasets)"
    )
    p_build.add_argument(
        "--recursive", action="store_true", help="Recurse into subdirectories"
    )

    # search subparser
    p_search = subparsers.add_parser(
        "search", help="Search visual index by text query"
    )
    p_search.add_argument(
        "--index-dir", required=True, help="Directory containing built index"
    )
    p_search.add_argument(
        "--query", "-q", required=True, help="Text query to search for"
    )
    p_search.add_argument(
        "--k", type=int, default=10, help="Number of results to return"
    )
    p_search.add_argument(
        "--lm-studio-url", default=None, help="LM Studio API URL"
    )
    p_search.add_argument(
        "--embed-model", default="text-embedding-mxbai-embed-large-v1", help="Embedding model"
    )

    return parser.parse_args()


def main():
    args = parse_args()
    if args.command == "build":
        cmd_build(args)
    elif args.command == "search":
        cmd_search(args)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Verify CLI parses correctly**

Run: `uv run python -m pixelrag_visual.cli --help`
Expected: Shows help with build and search subcommands

Run: `uv run python -m pixelrag_visual.cli build --help`
Expected: Shows build subcommand help with all arguments

- [ ] **Step 3: Commit**

```bash
git add visual/src/pixelrag_visual/cli.py
git commit -m "feat: add pixelrag visual CLI with build and search subcommands"
```

---

### Task 6: Register in PixelRAG CLI + pyproject.toml

**Files:**
- Modify: `src/pixelrag/cli.py` — add `"visual"` to STAGES dict
- Modify: `pyproject.toml` — add `visual` extra, update build packages

**Interfaces:**
- Consumes: nothing new
- Produces: `uv run pixelrag visual build --input-dir ...` works as a top-level command

- [ ] **Step 1: Register in STAGES dict**

In `src/pixelrag/cli.py`, find the `STAGES` dict (line 15-22) and add:
```python
"visual": ("pixelrag_visual.cli", "main", "pixelrag-visual", "visual"),
```

- [ ] **Step 2: Add visual extra to pyproject.toml**

Add to `[project.optional-dependencies]`:
```toml
visual = [
    "openai>=1.0",
    "faiss-cpu>=1.9.0",
]
```

Add to `[tool.hatch.build.targets.wheel].packages`:
```toml
"visual/src/pixelrag_visual",
```

Add to `[tool.pyright].extraPaths`:
```toml
"visual/src",
```

- [ ] **Step 3: Verify installation**

Run: `uv sync --extra visual`
Expected: Installs openai and faiss-cpu (already present from other extras)

Run: `uv run pixelrag visual --help`
Expected: Shows help for the new subcommand

- [ ] **Step 4: Commit**

```bash
git add src/pixelrag/cli.py pyproject.toml
git commit -m "feat: register pixelrag visual subcommand in CLI and pyproject.toml"
```

---

## Verification

1. **Unit tests pass:** `uv run pytest tests/visual/ -v` — all 23 tests PASS
2. **CLI help works:** `uv run pixelrag visual --help` — shows build/search subcommands
3. **End-to-end (small dataset):** Create 5-10 test images, run:
   ```bash
   uv run pixelrag visual build --input-dir ~/Desktop/image_dataset --output-dir ./visual_index
   uv run pixelrag visual search --index-dir ./visual_index --query "smiling people"
   ```
