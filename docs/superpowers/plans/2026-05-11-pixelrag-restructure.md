# PixelRAG 5-Package Restructure Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restructure ~/pixelrag/ from the current messy first-pass merge into 5 clean packages: ingest, embed, index, serve, train.

**Architecture:** Five independent uv workspace packages. ingest renders documents to tiles, embed provides orchestrator-free chunk/embed/build tools, index orchestrates full pipelines, serve provides the search API, train handles model fine-tuning. Source repos at ~/pixelrag-src/ are read-only.

**Tech Stack:** Python 3.12+, uv workspaces, FastAPI, FAISS, torch, Playwright, Chromium CDP

---

## File Structure

```
~/pixelrag/
├── pyproject.toml                          # workspace root
├── uv.lock
├── LICENSE
├── README.md
├── .gitignore
├── packages/
│   ├── ingest/
│   │   ├── pyproject.toml
│   │   └── src/pixelrag_render/
│   │       ├── __init__.py
│   │       ├── render.py                   # Public API dispatch
│   │       ├── backends/
│   │       │   ├── __init__.py
│   │       │   ├── cdp.py                  # Lean CDP capture (default)
│   │       │   ├── playwright.py           # Full Playwright (compat)
│   │       │   └── pdf.py                  # PDF rendering
│   │       └── bench/
│   │           ├── benchmark.py
│   │           ├── benchmark_optimizations.py
│   │           ├── benchmark_fullpage.py
│   │           └── benchmark_longtail_matrix.py
│   │
│   ├── embed/
│   │   ├── pyproject.toml
│   │   └── src/pixelrag_embed/
│   │       ├── __init__.py
│   │       ├── chunk.py                    # Tile → 1024px strips
│   │       ├── embed.py                    # Images → vectors
│   │       └── index.py                    # Vectors → FAISS
│   │
│   ├── index/
│   │   ├── pyproject.toml
│   │   └── src/pixelrag_index/
│   │       ├── __init__.py
│   │       ├── config.py                   # pixelrag.yaml parser
│   │       ├── pipelines.py                # End-to-end orchestration
│   │       ├── distributed.py              # S3ShardCoordinator (optional)
│   │       ├── monitor.py                  # Progress dashboard
│   │       └── sources/
│   │           ├── __init__.py
│   │           ├── base.py                 # Source ABC
│   │           ├── kiwix.py                # Wikipedia ZIM
│   │           ├── web.py                  # URLs + download (generalized news)
│   │           ├── pdf.py                  # PDF directory
│   │           └── local.py                # Auto-detect mixed files
│   │
│   ├── serve/
│   │   ├── pyproject.toml
│   │   └── src/pixelrag_serve/
│   │       ├── __init__.py
│   │       └── api.py                      # Unified search API
│   │
│   └── train/
│       ├── pyproject.toml
│       └── src/pixelrag_train/
│           ├── __init__.py
│           ├── models/
│           │   ├── __init__.py
│           │   └── biqwen3.py
│           ├── contrastive.py
│           └── mine.py
│
└── eval/
    ├── run_naive_simpleqa.py
    └── simpleqa/
```

---

### Task 1: Clean workspace and create scaffold

**Files:**
- Modify: `~/pixelrag/pyproject.toml`
- Create: all package directories and `__init__.py` files

- [ ] **Step 1: Remove old packages directory**

```bash
cd ~/pixelrag
rm -rf packages/
```

- [ ] **Step 2: Create new package skeleton**

```bash
cd ~/pixelrag

# ingest
mkdir -p packages/render/src/pixelrag_render/backends
mkdir -p packages/render/src/pixelrag_render/bench

# embed
mkdir -p packages/embed/src/pixelrag_embed

# index
mkdir -p packages/index/src/pixelrag_index/sources

# serve
mkdir -p packages/serve/src/pixelrag_serve

# train
mkdir -p packages/train/src/pixelrag_train/models

# __init__.py for all packages
for pkg in \
    packages/render/src/pixelrag_render \
    packages/render/src/pixelrag_render/backends \
    packages/embed/src/pixelrag_embed \
    packages/index/src/pixelrag_index \
    packages/index/src/pixelrag_index/sources \
    packages/serve/src/pixelrag_serve \
    packages/train/src/pixelrag_train \
    packages/train/src/pixelrag_train/models; do
    touch "$pkg/__init__.py"
done
```

- [ ] **Step 3: Update workspace root pyproject.toml**

Write `~/pixelrag/pyproject.toml`:
```toml
[project]
name = "pixelrag"
version = "0.1.0"
description = "Visual Retrieval-Augmented Generation — render, embed, index, search, train"
requires-python = ">=3.12"

[tool.uv.workspace]
members = ["packages/*"]

[tool.uv]
override-dependencies = ["nvidia-cudnn-cu12==9.20.0.48"]
environments = ["sys_platform == 'linux'"]

[[tool.uv.index]]
name = "pytorch-cu129"
url = "https://download.pytorch.org/whl/cu129"
explicit = true
```

- [ ] **Step 4: Commit scaffold**

```bash
cd ~/pixelrag
git add -A
git commit -m "scaffold: 5-package workspace (ingest, embed, index, serve, train)"
```

---

### Task 2: pixelrag-render package

**Files:**
- Create: `packages/render/pyproject.toml`
- Create: `packages/render/src/pixelrag_render/render.py`
- Create: `packages/render/src/pixelrag_render/backends/cdp.py`
- Copy+strip: `packages/render/src/pixelrag_render/backends/playwright.py`
- Create: `packages/render/src/pixelrag_render/backends/pdf.py`
- Copy: `packages/render/src/pixelrag_render/bench/*.py`

- [ ] **Step 1: Create pyproject.toml**

Write `~/pixelrag/packages/render/pyproject.toml`:
```toml
[project]
name = "pixelrag-render"
version = "0.1.0"
description = "Document → image tiles. Renders web pages, PDFs, and local files as tiled screenshots."
requires-python = ">=3.12"
dependencies = [
    "playwright>=1.40.0",
    "pillow>=10.0.0",
    "aiohttp>=3.9.0",
]

[project.optional-dependencies]
pdf = ["pdf2image>=1.16.0"]
dev = ["pytest>=7.0.0", "pytest-asyncio>=0.21.0"]

[project.scripts]
pixelrag-render = "pixelrag_render.render:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/pixelrag_render"]
```

- [ ] **Step 2: Create render.py — public API**

Write `~/pixelrag/packages/render/src/pixelrag_render/render.py`:
```python
"""Public API for rendering documents to image tiles.

Usage:
    from pixelrag_render import render_url, render_pdf, render_file

    tiles = render_url("https://en.wikipedia.org/wiki/Python", "./output")
    tiles = render_pdf("paper.pdf", "./output")
    tiles = render_file("doc.html", "./output")  # auto-detect type
"""

import argparse
import os
from pathlib import Path


def render_url(
    url: str,
    output_dir: str,
    backend: str = "cdp",
    *,
    tile_height: int = 8192,
    quality: int = 85,
    viewport_width: int = 875,
    workers: int = 1,
    **backend_kwargs,
) -> list[Path]:
    """Render a URL to tiled screenshots.

    Args:
        url: Web page URL to render.
        output_dir: Directory for output tiles.
        backend: "cdp" (default, fastest) or "playwright" (more options).
        tile_height: Height of each tile in pixels.
        quality: JPEG quality (1-100).
        viewport_width: Browser viewport width.
        workers: Number of browser workers for batch rendering.

    Returns:
        List of tile file paths.
    """
    if backend == "cdp":
        from .backends.cdp import render_urls
        return render_urls(
            [url], output_dir,
            tile_height=tile_height, quality=quality,
            viewport_width=viewport_width, workers=workers,
            **backend_kwargs,
        )
    elif backend == "playwright":
        from .backends.playwright import render_urls
        return render_urls(
            [url], output_dir,
            tile_height=tile_height, quality=quality,
            viewport_width=viewport_width,
            **backend_kwargs,
        )
    else:
        raise ValueError(f"Unknown backend: {backend!r}. Use 'cdp' or 'playwright'.")


def render_pdf(
    path: str,
    output_dir: str,
    *,
    dpi: int = 200,
    pages: str | None = None,
) -> list[Path]:
    """Render a PDF to tiled page images.

    Args:
        path: Path to PDF file.
        output_dir: Directory for output tiles.
        dpi: Rendering resolution.
        pages: Page range (e.g. "1-10"). None = all pages.

    Returns:
        List of tile file paths.
    """
    from .backends.pdf import render_pdf as _render_pdf
    return _render_pdf(path, output_dir, dpi=dpi, pages=pages)


def render_file(
    path: str,
    output_dir: str,
    backend: str = "cdp",
    **kwargs,
) -> list[Path]:
    """Auto-detect file type and render to tiles.

    Supports: .pdf, .html, .png/.jpg (direct copy), URLs (if starts with http).
    """
    p = str(path)
    if p.startswith("http://") or p.startswith("https://"):
        return render_url(p, output_dir, backend=backend, **kwargs)
    ext = os.path.splitext(p)[1].lower()
    if ext == ".pdf":
        return render_pdf(p, output_dir, **kwargs)
    elif ext in (".html", ".htm"):
        file_url = f"file://{os.path.abspath(p)}"
        return render_url(file_url, output_dir, backend=backend, **kwargs)
    elif ext in (".png", ".jpg", ".jpeg", ".webp"):
        # Image files: copy directly as a single tile
        os.makedirs(output_dir, exist_ok=True)
        import shutil
        dest = Path(output_dir) / Path(p).name
        shutil.copy2(p, dest)
        return [dest]
    else:
        raise ValueError(f"Unsupported file type: {ext}")


def main():
    parser = argparse.ArgumentParser(description="Render documents to image tiles")
    parser.add_argument("inputs", nargs="+", help="URLs, file paths, or directories")
    parser.add_argument("--output", "-o", default="./tiles", help="Output directory")
    parser.add_argument("--backend", default="cdp", choices=["cdp", "playwright"])
    parser.add_argument("--tile-height", type=int, default=8192)
    parser.add_argument("--quality", type=int, default=85)
    parser.add_argument("--viewport-width", type=int, default=875)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--dpi", type=int, default=200, help="PDF rendering DPI")
    args = parser.parse_args()

    all_tiles = []
    for inp in args.inputs:
        tiles = render_file(
            inp, args.output,
            backend=args.backend,
            tile_height=args.tile_height,
            quality=args.quality,
            viewport_width=args.viewport_width,
            workers=args.workers,
            dpi=args.dpi,
        )
        all_tiles.extend(tiles)
        print(f"{inp} → {len(tiles)} tiles")

    print(f"\nTotal: {len(all_tiles)} tiles in {args.output}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create cdp.py — lean CDP backend**

Copy from source and generalize:
```bash
cp ~/pixelrag-src/wiki-screenshot/scripts/render_news_pages.py \
   ~/pixelrag/packages/render/src/pixelrag_render/backends/cdp.py
```

Then apply these transformations to `cdp.py`:
1. Remove news-specific imports (`from wiki_screenshot.news.db import NewsDB`, `from wiki_screenshot.news.metrics import start_metrics_server`)
2. Remove `NewsDB` usage in `worker()` and `run_batch()` — replace with simple success/fail counters
3. Remove `check_nginx()` preflight — not general
4. Remove `main()` function (the CLI with `--db-path`, `--pages-dir` etc.) — replaced by `render.py`
5. Rename `capture_article()` to a general name
6. Export a `render_urls(urls, output_dir, ...)` function that `render.py` calls
7. Remove hardcoded paths (`/opt/dlami/nvme/`)
8. Keep: `_launch_browser()`, `BROWSER_ARGS`, the CDP capture logic, multi-browser worker architecture, JPEG tile output

- [ ] **Step 4: Copy and strip playwright.py**

```bash
cp ~/pixelrag-src/wiki-screenshot/src/wiki_screenshot/tools/playwright_tool.py \
   ~/pixelrag/packages/render/src/pixelrag_render/backends/playwright.py
```

Transformations to `playwright.py`:
1. Remove imports of `streaming_capture`, `raw_pixels`, `temp_dirs`
2. Remove the `use_streaming` code path and all streaming-related params
3. Remove unused experimental options (keep only: `use_cdp_screenshot`, `cdp_optimize_for_speed`, `segmented_save_tiles`, `segment_height`, `enable_gpu`, `device_scale_factor`, `image_format`, `quality`, `width`, `max_height`)
4. Remove `_cdp_sessions` management if not needed for the retained CDP path
5. Export a `render_urls(urls, output_dir, ...)` function matching cdp.py's interface
6. Keep: CDP screenshot mode, segmented tile capture, GPU rasterization flags, core `_capture_page()` logic
7. Target: strip from ~2388 lines to ~500-800 lines of production-relevant code

- [ ] **Step 5: Create pdf.py — PDF backend**

Write `~/pixelrag/packages/render/src/pixelrag_render/backends/pdf.py`:
```python
"""PDF rendering backend: PDF pages → tile images."""

import json
import os
from pathlib import Path


def render_pdf(
    path: str,
    output_dir: str,
    *,
    dpi: int = 200,
    pages: str | None = None,
) -> list[Path]:
    """Render PDF pages as tile images.

    Args:
        path: Path to PDF file.
        output_dir: Output directory for tiles.
        dpi: Rendering resolution.
        pages: Page range string (e.g. "1-10", "3,5,7"). None = all.

    Returns:
        List of tile image paths.
    """
    try:
        from pdf2image import convert_from_path
    except ImportError:
        raise ImportError(
            "pdf2image is required for PDF rendering. "
            "Install with: pip install pixelrag-render[pdf]"
        )

    pdf_path = Path(path)
    doc_id = pdf_path.stem
    tile_dir = Path(output_dir) / f"{doc_id}.tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)

    # Parse page range
    kwargs = {"dpi": dpi}
    if pages:
        if "-" in pages:
            first, last = pages.split("-", 1)
            kwargs["first_page"] = int(first)
            kwargs["last_page"] = int(last)
        else:
            page_nums = [int(p.strip()) for p in pages.split(",")]
            kwargs["first_page"] = min(page_nums)
            kwargs["last_page"] = max(page_nums)

    images = convert_from_path(str(pdf_path), **kwargs)

    tile_paths = []
    for i, img in enumerate(images):
        tile_path = tile_dir / f"tile_{i:04d}.jpg"
        img.save(tile_path, "JPEG", quality=85)
        tile_paths.append(tile_path)

    # Write manifest
    manifest = {
        "source": str(pdf_path),
        "dpi": dpi,
        "pages": len(images),
        "tiles": [p.name for p in tile_paths],
        "complete": True,
    }
    with open(tile_dir / "tiles.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return tile_paths
```

- [ ] **Step 6: Copy bench/**

```bash
cp ~/pixelrag-src/wiki-screenshot/bench/benchmark.py \
   ~/pixelrag/packages/render/src/pixelrag_render/bench/
cp ~/pixelrag-src/wiki-screenshot/bench/benchmark_optimizations.py \
   ~/pixelrag/packages/render/src/pixelrag_render/bench/
cp ~/pixelrag-src/wiki-screenshot/bench/benchmark_fullpage.py \
   ~/pixelrag/packages/render/src/pixelrag_render/bench/
cp ~/pixelrag-src/wiki-screenshot/bench/benchmark_longtail_matrix.py \
   ~/pixelrag/packages/render/src/pixelrag_render/bench/
```

Fix imports in bench files: replace `wiki_screenshot` → `pixelrag_render`:
```bash
find ~/pixelrag/packages/render/src/pixelrag_render/bench -name '*.py' -exec sed -i \
    -e 's/from wiki_screenshot/from pixelrag_render/g' \
    -e 's/import wiki_screenshot/import pixelrag_render/g' \
    {} +
```

- [ ] **Step 7: Verify ingest package imports**

```bash
cd ~/pixelrag
uv sync --package pixelrag-render 2>&1 | tail -3
uv run --package pixelrag-render python -c "from pixelrag_render.render import render_url, render_pdf, render_file; print('OK')"
```

- [ ] **Step 8: Commit**

```bash
cd ~/pixelrag
git add packages/render/
git commit -m "feat: add pixelrag-render package (CDP/Playwright/PDF backends)"
```

---

### Task 3: pixelrag-embed package

**Files:**
- Create: `packages/embed/pyproject.toml`
- Copy: `packages/embed/src/pixelrag_embed/chunk.py`
- Copy: `packages/embed/src/pixelrag_embed/embed.py`
- Copy: `packages/embed/src/pixelrag_embed/index.py`

- [ ] **Step 1: Create pyproject.toml**

Write `~/pixelrag/packages/embed/pyproject.toml`:
```toml
[project]
name = "pixelrag-embed"
version = "0.1.0"
description = "Image tiles → vectors → FAISS index. Three independent CLI tools."
requires-python = ">=3.12"
dependencies = [
    "torch>=2.9.0",
    "transformers>=4.57.0",
    "faiss-cpu>=1.9.0",
    "pillow>=10.0.0",
    "numpy>=1.26.0",
    "tqdm>=4.60.0",
]

[project.optional-dependencies]
gpu = ["faiss-gpu-cu12>=1.13.2"]

[project.scripts]
pixelrag-chunk = "pixelrag_embed.chunk:main"
pixelrag-embed = "pixelrag_embed.embed:main"
pixelrag-build-index = "pixelrag_embed.index:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/pixelrag_embed"]

[tool.uv]
override-dependencies = ["nvidia-cudnn-cu12==9.20.0.48"]

[tool.uv.sources]
torch = [{ index = "pytorch-cu129" }]
```

- [ ] **Step 2: Copy chunk.py**

```bash
cp ~/pixelrag-src/wiki-screenshot/embedding/chunk_tiles.py \
   ~/pixelrag/packages/embed/src/pixelrag_embed/chunk.py
```

No import renames needed — `chunk_tiles.py` only uses stdlib + PIL.

- [ ] **Step 3: Copy embed.py**

```bash
cp ~/pixelrag-src/wiki-screenshot/embedding/embed_tiles.py \
   ~/pixelrag/packages/embed/src/pixelrag_embed/embed.py
```

No import renames needed — `embed_tiles.py` only uses stdlib + numpy/PIL/tqdm + subprocess for vLLM/sglang.

- [ ] **Step 4: Copy index.py**

```bash
cp ~/pixelrag-src/wiki-screenshot/indexing/build_index.py \
   ~/pixelrag/packages/embed/src/pixelrag_embed/index.py
```

No import renames needed — `build_index.py` only uses stdlib + numpy + faiss.

- [ ] **Step 5: Clean hardcoded paths in all three files**

```bash
find ~/pixelrag/packages/embed/src -name '*.py' -exec sed -i \
    -e 's|/opt/dlami/nvme/[^ "'"'"'\\)]*|./data|g' \
    -e 's|/home/user/[^ "'"'"'\\)]*|./|g' \
    -e 's|/home/ubuntu/[^ "'"'"'\\)]*|./|g' \
    -e 's|/home/andy/[^ "'"'"'\\)]*|./|g' \
    {} +
```

- [ ] **Step 6: Verify embed package imports**

```bash
cd ~/pixelrag
uv sync --package pixelrag-embed 2>&1 | tail -3
uv run --package pixelrag-embed python -c "from pixelrag_embed import chunk, embed, index; print('OK')"
```

- [ ] **Step 7: Commit**

```bash
cd ~/pixelrag
git add packages/embed/
git commit -m "feat: add pixelrag-embed package (chunk, embed, build-index)"
```

---

### Task 4: pixelrag-serve package

**Files:**
- Create: `packages/serve/pyproject.toml`
- Create: `packages/serve/src/pixelrag_serve/api.py` (merged from 3 APIs)

- [ ] **Step 1: Create pyproject.toml**

Write `~/pixelrag/packages/serve/pyproject.toml`:
```toml
[project]
name = "pixelrag-serve"
version = "0.1.0"
description = "FAISS-based visual search API. Serves any pre-built index."
requires-python = ">=3.12"
dependencies = [
    "fastapi>=0.115.0",
    "uvicorn>=0.30.0",
    "numpy>=1.26.0",
    "faiss-cpu>=1.9.0",
    "transformers>=4.57.0",
    "torch>=2.9.0",
    "qwen-vl-utils",
    "pillow>=10.0.0",
    "pydantic>=2.0.0",
]

[project.optional-dependencies]
gpu = ["faiss-gpu-cu12>=1.13.2"]

[project.scripts]
pixelrag-serve = "pixelrag_serve.api:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/pixelrag_serve"]

[tool.uv.sources]
torch = [{ index = "pytorch-cu129" }]
```

- [ ] **Step 2: Create unified api.py**

Start from the existing adapted search_api.py (which already has CPU support):
```bash
cp ~/pixelrag-src/wiki-screenshot/serving/search_api.py \
   ~/pixelrag/packages/serve/src/pixelrag_serve/api.py
```

Apply transformations to `api.py`:
1. Remove vllm backend (keep only direct transformers inference)
2. Remove `torch.compile()` call
3. Add `--device cpu|cuda` arg (default cpu), use `torch.float32` on CPU
4. Replace hardcoded `/opt/dlami/nvme/` paths with env var defaults (`PIXELRAG_INDEX_DIR`, `PIXELRAG_ARTICLES_JSON`)
5. Replace `torch_dtype` with `dtype` in `from_pretrained()` to fix deprecation warning
6. This is the unified API — it serves any FAISS index (wiki, news, text, any). No separate news_search_api or text_search_api needed if the index format is consistent.

The existing api.py from the first-pass merge (at `~/pixelrag/packages/serving/src/pixelrag_serving/search_api.py`) already has most of these changes. Use that as the starting point instead:
```bash
# Actually use the already-adapted version
cp ~/pixelrag/packages/serving/src/pixelrag_serving/search_api.py \
   ~/pixelrag/packages/serve/src/pixelrag_serve/api.py 2>/dev/null || \
cp ~/pixelrag-src/wiki-screenshot/serving/search_api.py \
   ~/pixelrag/packages/serve/src/pixelrag_serve/api.py
```

If using the source version, apply the CPU adaptations from the spec (remove vllm, add --device, fix torch_dtype, replace paths).

- [ ] **Step 3: Verify serve package**

```bash
cd ~/pixelrag
uv sync --package pixelrag-serve 2>&1 | tail -3
uv run --package pixelrag-serve python -c "from pixelrag_serve import api; print('OK')"
```

- [ ] **Step 4: Smoke test with existing downloaded index**

```bash
PIXELRAG_INDEX_DIR=/home/yichuan/pixelrag-data/text_search_index_1024 \
PIXELRAG_ARTICLES_JSON=/home/yichuan/pixelrag-data/articles.json \
uv run --package pixelrag-serve python -m pixelrag_serve.api --port 31001 &
sleep 120  # wait for index + model loading
curl -s http://localhost:31001/health
curl -s -X POST http://localhost:31001/search \
    -H "Content-Type: application/json" \
    -d '{"queries": [{"text": "capital of France"}], "n_docs": 3}' | python3 -m json.tool | head -20
kill %1
```

- [ ] **Step 5: Commit**

```bash
cd ~/pixelrag
git add packages/serve/
git commit -m "feat: add pixelrag-serve package (unified FAISS search API)"
```

---

### Task 5: pixelrag-train package

**Files:**
- Create: `packages/train/pyproject.toml`
- Copy: `packages/train/src/pixelrag_train/models/biqwen3.py`
- Copy+rename: `packages/train/src/pixelrag_train/contrastive.py`
- Create: `packages/train/src/pixelrag_train/mine.py` (merged from 2 scripts)
- Copy: tests

- [ ] **Step 1: Create pyproject.toml**

Write `~/pixelrag/packages/train/pyproject.toml`:
```toml
[project]
name = "pixelrag-train"
version = "0.1.0"
description = "LoRA/DoRA contrastive fine-tuning for visual document retrieval embeddings"
requires-python = ">=3.12"
dependencies = [
    "torch==2.9.1",
    "torchvision",
    "transformers==4.57.1",
    "nvidia-cudnn-cu12==9.20.0.48",
    "peft>=0.15.0",
    "accelerate>=1.0.0",
    "Pillow",
    "tqdm",
    "numpy",
    "faiss-cpu",
    "wandb",
    "safetensors",
    "huggingface-hub",
    "qwen-vl-utils",
    "datasets",
]

[project.scripts]
pixelrag-train = "pixelrag_train.contrastive:main"
pixelrag-mine = "pixelrag_train.mine:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/pixelrag_train"]

[tool.uv]
override-dependencies = ["nvidia-cudnn-cu12==9.20.0.48"]

[tool.uv.sources]
torch = [{ index = "pytorch-cu129" }]
torchvision = [{ index = "pytorch-cu129" }]
```

- [ ] **Step 2: Copy model and training code**

```bash
SRC=~/pixelrag-src/wiki-screenshot-training
DST=~/pixelrag/packages/train

# Model
cp $SRC/models/__init__.py $DST/src/pixelrag_train/models/
cp $SRC/models/biqwen3.py $DST/src/pixelrag_train/models/

# Training script
cp $SRC/train_contrastors.py $DST/src/pixelrag_train/contrastive.py
```

- [ ] **Step 3: Create merged mine.py**

Copy the image mining script as base, then merge text mining functionality:
```bash
cp ~/pixelrag-src/wiki-screenshot-training/mine_hard_negatives.py \
   ~/pixelrag/packages/train/src/pixelrag_train/mine.py
```

Add to the `mine.py` argparse a `--mode image|text` flag. The image mode calls the image search API (original `mine_hard_negatives.py` behavior). The text mode calls the text search API (original `mine_text_hard_negatives.py` behavior). Read both source files to understand the differences and merge them.

Key differences between the two scripts:
- `mine_hard_negatives.py` queries `:30888/search` with image results (returns `chunk_path`)
- `mine_text_hard_negatives.py` queries `:30889/search` with text results (returns `article_id`, `chunk_index`, `text`)
- Both share: query loading, JSONL I/O, concurrent API calls, dedup logic

- [ ] **Step 4: Copy tests**

```bash
cp ~/pixelrag-src/wiki-screenshot-training/tests/test_grad_equivalence.py \
   ~/pixelrag/packages/train/tests/
cp ~/pixelrag-src/wiki-screenshot-training/tests/test_grad_multi_gpu.py \
   ~/pixelrag/packages/train/tests/
mkdir -p ~/pixelrag/packages/train/tests
```

- [ ] **Step 5: Clean hardcoded paths**

```bash
find ~/pixelrag/packages/train -name '*.py' -exec sed -i \
    -e 's|/opt/dlami/nvme/[^ "'"'"'\\)]*|./data|g' \
    -e 's|/home/user/[^ "'"'"'\\)]*|./|g' \
    -e 's|/home/ubuntu/[^ "'"'"'\\)]*|./|g' \
    {} +
```

- [ ] **Step 6: Commit**

```bash
cd ~/pixelrag
git add packages/train/
git commit -m "feat: add pixelrag-train package (contrastive training + mining)"
```

---

### Task 6: pixelrag-index package

**Files:**
- Create: `packages/index/pyproject.toml`
- Create: `packages/index/src/pixelrag_index/config.py`
- Create: `packages/index/src/pixelrag_index/pipelines.py`
- Copy+refactor: `packages/index/src/pixelrag_index/distributed.py`
- Copy+refactor: `packages/index/src/pixelrag_index/monitor.py`
- Copy+refactor: `packages/index/src/pixelrag_index/sources/kiwix.py`
- Create: `packages/index/src/pixelrag_index/sources/web.py`
- Create: `packages/index/src/pixelrag_index/sources/pdf.py`
- Create: `packages/index/src/pixelrag_index/sources/local.py`
- Create: `packages/index/src/pixelrag_index/sources/base.py`

- [ ] **Step 1: Create pyproject.toml**

Write `~/pixelrag/packages/index/pyproject.toml`:
```toml
[project]
name = "pixelrag-index"
version = "0.1.0"
description = "Build searchable FAISS indexes from any document source"
requires-python = ">=3.12"
dependencies = [
    "pixelrag-render",
    "pixelrag-embed",
    "pyyaml>=6.0",
    "tqdm>=4.60.0",
]

[project.optional-dependencies]
distributed = ["boto3>=1.42.0"]

[project.scripts]
pixelrag-index = "pixelrag_index.pipelines:main"
pixelrag-monitor = "pixelrag_index.monitor:main"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.hatch.build.targets.wheel]
packages = ["src/pixelrag_index"]
```

- [ ] **Step 2: Create sources/base.py**

Write `~/pixelrag/packages/index/src/pixelrag_index/sources/base.py`:
```python
"""Base class for document sources."""

from dataclasses import dataclass
from typing import Iterator


@dataclass
class Document:
    """A document to be rendered and indexed."""
    id: str
    url: str | None = None
    path: str | None = None
    metadata: dict | None = None


class Source:
    """Base class for document sources. Subclasses yield Documents."""

    def __iter__(self) -> Iterator[Document]:
        raise NotImplementedError

    def __len__(self) -> int:
        raise NotImplementedError
```

- [ ] **Step 3: Create config.py**

Write `~/pixelrag/packages/index/src/pixelrag_index/config.py`:
```python
"""Parse pixelrag.yaml configuration with parameter forwarding."""

import os
from pathlib import Path

import yaml

from .sources import SOURCES


DEFAULT_CONFIG = {
    "ingest": {"backend": "cdp", "quality": 85, "tile_height": 8192},
    "embed": {"model": "text-embedding-mxbai-embed-large-v1", "device": "cuda"},
    "output": "./index",
}


def load_config(path: str | None = None) -> dict:
    """Load config from pixelrag.yaml or defaults.

    Looks for pixelrag.yaml in: explicit path > cwd > ~/.config/pixelrag/
    """
    if path is None:
        candidates = [
            Path("pixelrag.yaml"),
            Path("pixelrag.yml"),
            Path.home() / ".config" / "pixelrag" / "pixelrag.yaml",
        ]
        for c in candidates:
            if c.exists():
                path = str(c)
                break

    if path and os.path.exists(path):
        with open(path) as f:
            config = yaml.safe_load(f)
    else:
        config = {}

    # Merge with defaults
    result = {**DEFAULT_CONFIG, **config}
    return result


def make_source(config: dict):
    """Create a Source instance from config["source"] with parameter forwarding."""
    source_config = dict(config.get("source", {}))
    source_type = source_config.pop("type", "local")

    if source_type not in SOURCES:
        raise ValueError(
            f"Unknown source type: {source_type!r}. "
            f"Available: {', '.join(SOURCES.keys())}"
        )

    return SOURCES[source_type](**source_config)
```

- [ ] **Step 4: Create sources/kiwix.py**

Copy from source and refactor to use the new Source interface:
```bash
cp ~/pixelrag-src/wiki-screenshot/src/wiki_screenshot/datasources/kiwix.py \
   ~/pixelrag/packages/index/src/pixelrag_index/sources/kiwix.py
```

Refactor: make `KiwixSource` extend `Source`, yield `Document` objects instead of `Article` objects. Remove imports of `wiki_screenshot`. Keep the core article iteration logic (fetch from kiwix-serve, cache articles.json).

- [ ] **Step 5: Create sources/web.py**

Copy news-related code and generalize:
```bash
# Start from the news datasource as the iteration layer
cp ~/pixelrag-src/wiki-screenshot/src/wiki_screenshot/datasources/news.py \
   ~/pixelrag/packages/index/src/pixelrag_index/sources/web.py
```

Then integrate download logic from `news/download.py` and `news/db.py` as internal implementation. Rename news-specific classes/functions to general names. Add `preset` parameter with `"news"` preset containing BBC/CNN/AP domain limits and cookie banner CSS.

Key transformations:
1. `NewsDataSource` → `WebSource(Source)`
2. Import and embed `NewsDownloader` logic from `news/download.py` (or import it as a submodule)
3. Import `NewsDB` from `news/db.py` as `WebDB` (SQLite state tracking)
4. Add `PRESETS` dict with `"news"` key containing domain limits, cookie CSS
5. Yield `Document` objects instead of `Article`

- [ ] **Step 6: Create sources/pdf.py and sources/local.py**

Write `~/pixelrag/packages/index/src/pixelrag_index/sources/pdf.py`:
```python
"""PDF directory source — iterates PDF files for rendering."""

import os
from pathlib import Path
from typing import Iterator

from .base import Document, Source


class PDFSource(Source):
    def __init__(self, path: str, **kwargs):
        self.path = Path(path)
        self.kwargs = kwargs
        self._files = sorted(self.path.glob("**/*.pdf"))

    def __iter__(self) -> Iterator[Document]:
        for pdf in self._files:
            yield Document(
                id=pdf.stem,
                path=str(pdf),
                metadata={"type": "pdf", **self.kwargs},
            )

    def __len__(self) -> int:
        return len(self._files)
```

Write `~/pixelrag/packages/index/src/pixelrag_index/sources/local.py`:
```python
"""Local directory source — auto-detects file types and routes."""

import os
from pathlib import Path
from typing import Iterator

from .base import Document, Source

SUPPORTED_EXTENSIONS = {
    ".pdf": "pdf",
    ".html": "web",
    ".htm": "web",
    ".png": "image",
    ".jpg": "image",
    ".jpeg": "image",
    ".webp": "image",
}


class LocalSource(Source):
    def __init__(self, path: str, **kwargs):
        self.path = Path(path)
        self.kwargs = kwargs
        self._files = []
        for f in sorted(self.path.rglob("*")):
            if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
                self._files.append(f)

    def __iter__(self) -> Iterator[Document]:
        for f in self._files:
            ext = f.suffix.lower()
            file_type = SUPPORTED_EXTENSIONS.get(ext, "unknown")
            if file_type == "web":
                url = f"file://{f.resolve()}"
                yield Document(id=f.stem, url=url, metadata={"type": file_type})
            else:
                yield Document(id=f.stem, path=str(f), metadata={"type": file_type})

    def __len__(self) -> int:
        return len(self._files)
```

- [ ] **Step 7: Create sources/__init__.py registry**

Write `~/pixelrag/packages/index/src/pixelrag_index/sources/__init__.py`:
```python
"""Document source registry."""

from .base import Document, Source
from .kiwix import KiwixSource
from .local import LocalSource
from .pdf import PDFSource
from .web import WebSource

SOURCES = {
    "kiwix": KiwixSource,
    "web": WebSource,
    "pdf": PDFSource,
    "local": LocalSource,
}

__all__ = ["Document", "Source", "SOURCES", "KiwixSource", "WebSource", "PDFSource", "LocalSource"]
```

- [ ] **Step 8: Copy and refactor distributed.py**

```bash
cp ~/pixelrag-src/wiki-screenshot/src/wiki_screenshot/coordinator.py \
   ~/pixelrag/packages/index/src/pixelrag_index/distributed.py
```

Rename: `wiki_screenshot` imports → none needed (coordinator.py only uses stdlib + boto3).
Replace hardcoded paths.

- [ ] **Step 9: Copy and refactor monitor.py**

```bash
cp ~/pixelrag-src/wiki-screenshot/scripts/monitor_global.py \
   ~/pixelrag/packages/index/src/pixelrag_index/monitor.py
```

Replace `from pixelrag_capture.coordinator import S3ShardCoordinator` → `from .distributed import S3ShardCoordinator`.

- [ ] **Step 10: Create pipelines.py — orchestration**

Write `~/pixelrag/packages/index/src/pixelrag_index/pipelines.py`:
```python
"""End-to-end pipeline: source → ingest → chunk → embed → build index."""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from .config import load_config, make_source

logger = logging.getLogger("pixelrag-index")


def build(config: dict) -> Path:
    """Build a searchable index from a document source.

    Chains: source → pixelrag-render (render) → pixelrag-chunk → pixelrag-embed → pixelrag-build-index
    """
    source = make_source(config)
    output_dir = Path(config.get("output", "./index"))
    tiles_dir = output_dir / "tiles"
    chunks_dir = output_dir / "chunks"
    embeddings_dir = output_dir / "embeddings"
    index_dir = output_dir

    ingest_config = config.get("ingest", {})
    embed_config = config.get("embed", {})

    os.makedirs(tiles_dir, exist_ok=True)
    os.makedirs(chunks_dir, exist_ok=True)
    os.makedirs(embeddings_dir, exist_ok=True)

    logger.info("Source: %s (%d documents)", type(source).__name__, len(source))

    # Stage 1: Render documents to tiles
    from pixelrag_render.render import render_url, render_pdf, render_file

    logger.info("Stage 1: Rendering %d documents...", len(source))
    for doc in source:
        doc_tiles_dir = str(tiles_dir / f"{doc.id}.tiles")
        if doc.url:
            render_url(doc.url, doc_tiles_dir, **ingest_config)
        elif doc.path:
            render_file(doc.path, doc_tiles_dir, **ingest_config)
        logger.info("  Rendered: %s", doc.id)

    # Stage 2: Chunk tiles
    logger.info("Stage 2: Chunking tiles...")
    subprocess.run([
        sys.executable, "-m", "pixelrag_embed.chunk",
        "--tiles-dir", str(tiles_dir),
    ], check=True)

    # Stage 3: Embed chunks
    logger.info("Stage 3: Embedding chunks...")
    embed_cmd = [
        sys.executable, "-m", "pixelrag_embed.embed",
        "--shard-dir", str(tiles_dir),
        "--output-dir", str(embeddings_dir),
    ]
    if "gpu_ids" in embed_config:
        embed_cmd.extend(["--gpu-ids", ",".join(str(g) for g in embed_config["gpu_ids"])])
    if "model" in embed_config:
        embed_cmd.extend(["--model", embed_config["model"]])
    if "backend" in embed_config:
        embed_cmd.extend(["--backend", embed_config["backend"]])
    subprocess.run(embed_cmd, check=True)

    # Stage 4: Build FAISS index
    logger.info("Stage 4: Building FAISS index...")
    subprocess.run([
        sys.executable, "-m", "pixelrag_embed.index",
        "build",
        "--embeddings-dir", str(embeddings_dir),
        "--output-dir", str(index_dir),
    ], check=True)

    logger.info("Index built at: %s", index_dir)
    return index_dir


def main():
    parser = argparse.ArgumentParser(description="Build a visual search index")
    parser.add_argument("command", choices=["build"], help="Command to run")
    parser.add_argument("--config", "-c", default=None, help="Path to pixelrag.yaml")
    parser.add_argument("--source", "-s", default=None, help="Source path (overrides config)")
    parser.add_argument("--output", "-o", default=None, help="Output directory")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(message)s")

    config = load_config(args.config)
    if args.source:
        config.setdefault("source", {})["path"] = args.source
    if args.output:
        config["output"] = args.output

    if args.command == "build":
        build(config)


if __name__ == "__main__":
    main()
```

- [ ] **Step 11: Commit**

```bash
cd ~/pixelrag
git add packages/index/
git commit -m "feat: add pixelrag-index package (orchestration, sources, distributed)"
```

---

### Task 7: Update eval/, README, and cleanup

**Files:**
- Modify: `eval/` (fix imports if needed)
- Modify: `README.md`
- Remove: old `packages/` remnants

- [ ] **Step 1: Verify eval/ still works**

eval/ should be unchanged. Check for broken imports:
```bash
grep -rn 'pixelrag_capture\|pixelrag_serving\|pixelrag_training\|wiki_screenshot' ~/pixelrag/eval/ --include='*.py'
```

If any found, fix them. eval/ scripts talk to the search API over HTTP, so they shouldn't import from other packages.

- [ ] **Step 2: Update README.md**

Rewrite to reflect the new 5-package architecture, user personas, and quick-start examples for each user type.

- [ ] **Step 3: Clean up arxiv/ directory**

The `arxiv/` directory appeared — add to `.gitignore` if it shouldn't be tracked, or remove from git.

- [ ] **Step 4: Final sweep**

```bash
cd ~/pixelrag
# No secrets
grep -rn 'hf_[A-Za-z0-9]\{20,\}' --include='*.py' --include='*.sh' . | grep -v .git/
# No Tsinghua mirror
grep -rn 'tsinghua' --include='*.toml' . | grep -v .git/
# No hardcoded machine paths
grep -rn '/opt/dlami\|/home/user/\|/home/ubuntu/\|/home/andy/' --include='*.py' . | grep -v .git/ | head -10
# No large files
find . -size +1M -type f | grep -v '.git/' | grep -v '.venv/'
```

- [ ] **Step 5: Commit**

```bash
cd ~/pixelrag
git add -A
git commit -m "cleanup: update README, eval, remove old package remnants"
```

---

### Task 8: Workspace verification

- [ ] **Step 1: Resolve workspace dependencies**

```bash
cd ~/pixelrag
rm -f uv.lock
uv sync 2>&1 | tail -5
```

- [ ] **Step 2: Verify each package imports**

```bash
uv run --package pixelrag-render python -c "from pixelrag_render.render import render_url; print('ingest OK')"
uv run --package pixelrag-embed python -c "from pixelrag_embed import chunk, embed, index; print('embed OK')"
uv run --package pixelrag-serve python -c "from pixelrag_serve import api; print('serve OK')"
uv run --package pixelrag-train python -c "from pixelrag_train.models.biqwen3 import BiQwen3; print('train OK')"
uv run --package pixelrag-index python -c "from pixelrag_index.config import load_config; print('index OK')"
```

- [ ] **Step 3: Verify serving still works**

```bash
PIXELRAG_INDEX_DIR=/home/yichuan/pixelrag-data/text_search_index_1024 \
PIXELRAG_ARTICLES_JSON=/home/yichuan/pixelrag-data/articles.json \
uv run --package pixelrag-serve pixelrag-serve --port 31001 &
# Wait for loading, then test
sleep 120
curl -s http://localhost:31001/health
curl -s -X POST http://localhost:31001/search \
    -H "Content-Type: application/json" \
    -d '{"queries": [{"text": "Apollo 11"}], "n_docs": 3}'
kill %1
```

- [ ] **Step 4: Commit lock file**

```bash
cd ~/pixelrag
git add uv.lock
git commit -m "chore: regenerate uv.lock for 5-package workspace"
```
