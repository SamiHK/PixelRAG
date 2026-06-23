# PixelRAG Unirepo Restructure — Design Spec

**Date:** 2026-05-11
**Status:** Draft

## Context

PixelRAG is a visual document retrieval framework: any document (web page, PDF, image) → visual rendering → embedding → FAISS index → search API. Three private repos (yichuan-w/Vis-RAG, andylizf/wiki-screenshot, andylizf/wiki-screenshot-training) are being merged into a single public repo with a clean architecture.

The current repo at ~/pixelrag/ has a messy first-pass merge. This spec defines the target architecture.

## Users

**A — Framework user (primary):** "I have documents (web pages, PDFs, local files) and want to build a visual retrieval system." Needs the full pipeline. Cares about generality — not a Wikipedia-specific tool.

**B — Paper reproducer:** "I want to reproduce PixelRAG results." Downloads pre-built indexes, starts the search API, runs eval. Should exist but not the design focus.

**C — Agent developer:** Uses screenshot capture and visual search as agent skills/tools. Needs callable APIs: give URL → get screenshot, give query → get results. Will demo agent integration.

**D — Model trainer:** Trains visual embedding models. Contrastive learning + hard negative mining via search API. Should exist but not the design focus.

## Package Architecture

Five packages, single-direction dependencies:

```
ingest ←── index ──→ embed

serve (independent)

train → serve (API calls for mining)
```

### Package 1: pixelrag-render

**"Document → image tiles."** Standalone rendering tool. Agents call it directly; index calls it for batch jobs.

```
src/pixelrag_render/
├── render.py              # Public API:
│                          #   render_url(url, output_dir, backend="cdp") → list[Path]
│                          #   render_pdf(path, output_dir) → list[Path]
│                          #   render_file(path, output_dir) → list[Path]  (auto-detect)
├── backends/
│   ├── cdp.py             # Lean CDP capture — default, fastest
│   │                      # Direct Page.captureScreenshot, multi-browser workers
│   │                      # JPEG q85, DPR 1, fromSurface=False, optimizeForSpeed=True
│   │                      # Based on render_news_pages.py (23.9s/50 articles benchmark)
│   ├── playwright.py      # Full Playwright — more options, experimental/compat
│   │                      # Stripped to production-useful config only
│   │                      # Keeps: CDP screenshot mode, segmented tiles, GPU rasterization
│   │                      # Removes: unused experimental options
│   └── pdf.py             # PDF → page images (pdf2image or PyMuPDF)
└── bench/                 # Rendering benchmarks
    ├── benchmark.py               # Config sweep (workers, batch size, concurrency)
    ├── benchmark_optimizations.py # GPU accel, PNG compression, tile sizes
    ├── benchmark_fullpage.py      # Screenshot strategy comparison
    └── benchmark_longtail_matrix.py  # Long pages × tile size × concurrency
```

**Dependencies:** playwright, pillow, aiohttp (lightweight — no torch)

**CLI entry points:**
- `pixelrag-render` → `pixelrag_render.render:main` (render URLs/files to tiles)

**Source of code:**
- `cdp.py` ← `scripts/render_news_pages.py` capture_article + worker + multi-browser setup (generalized, news-specific parts removed)
- `playwright.py` ← `tools/playwright_tool.py` (stripped from 2388L to production-relevant config)
- `bench/` ← `bench/` directory (kept as-is)
- `render.py` — new, thin API layer that dispatches to backends

### Package 2: pixelrag-embed

**"Image tiles → vectors → FAISS index."** Three independent CLI tools, orchestrator-free. Each has its own `main()`, no imports between them.

```
src/pixelrag_embed/
├── chunk.py     # Large image → 1024px strips
│                # Input: tile directory. Output: chunk PNGs + chunks.json
│                # Pure PIL, no torch. ~380 lines.
├── embed.py     # Images → embedding vectors
│                # Input: chunk directory. Output: shard_NNN.npz
│                # vLLM/sglang backend, multi-GPU. ~2400 lines.
└── index.py     # Vectors → FAISS IVFFlat index
                 # Input: embedding .npz shards. Output: index.faiss + metadata.npz
                 # ~330 lines.
```

**Dependencies:** torch, transformers, faiss-cpu, pillow, numpy, tqdm

**CLI entry points:**
- `pixelrag-chunk` → `pixelrag_embed.chunk:main`
- `pixelrag-embed` → `pixelrag_embed.embed:main`
- `pixelrag-build-index` → `pixelrag_embed.index:main`

**Source of code:**
- `chunk.py` ← `embedding/chunk_tiles.py`
- `embed.py` ← `embedding/embed_tiles.py`
- `index.py` ← `indexing/build_index.py`

### Package 3: pixelrag-index

**"Data source → complete searchable index."** Orchestration layer. Knows how to chain ingest + embed for different data sources. Two modes: single-machine (default, no S3) and distributed (S3 coordination for multi-machine).

```
src/pixelrag_index/
├── config.py          # pixelrag.yaml parser
│                      # Defines: source type, paths, embed model, output location
├── sources/           # Data source iterators (yield items for ingest to render)
│   ├── kiwix.py       #   Wikipedia ZIM → iterate articles → call ingest per article
│   ├── web.py         #   URL list/sitemap → download HTML+assets → call ingest
│   │                  #   Download + SQLite state are internal to this source
│   │                  #   Includes presets (e.g. "news") with per-domain rate limits,
│   │                  #   cookie banner CSS, source-specific HTML handling (BBC/CNN/AP)
│   │                  #   Usage: --source web --preset news
│   ├── pdf.py         #   PDF directory → iterate files → call ingest per file
│   └── local.py       #   Scan directory → auto-detect file types → route to above
├── pipelines.py       # End-to-end: source → ingest → chunk → embed → build
│                      # Chains the stages, handles checkpointing between stages
├── distributed.py     # S3ShardCoordinator + claim-loop worker (optional)
│                      # Only used with --distributed flag
│                      # Used by both capture and embedding distributed runs
└── monitor.py         # Cross-machine progress dashboard (reads S3 claims)
│                      # Only relevant in distributed mode
```

**Two orchestration modes:**
- `pixelrag-index build --source ./my_docs` — single machine, iterate locally, no S3
- `pixelrag-index build --source kiwix --distributed --bucket my-bucket` — multi-machine, S3 coordination

**Dependencies:** pixelrag-render, pixelrag-embed, boto3 (optional, only for distributed), tqdm

**CLI entry points:**
- `pixelrag-index` → `pixelrag_index.pipelines:main` (build index from source)
- `pixelrag-monitor` → `pixelrag_index.monitor:main` (progress dashboard)

**pixelrag.yaml — parameter forwarding pattern:**

Each section's parameters are forwarded directly to the corresponding package. Index only manages orchestration order, not parameter details.

```python
# index/config.py — forwarding logic
source_type = config["source"].pop("type")
source = SOURCES[source_type](**config["source"])   # forward all source params
# ingest params forwarded to render calls
# embed params forwarded to chunk/embed/build calls
```

```yaml
# Example: local files (User A)
source:
  type: local
  path: ./my_docs

ingest:
  backend: cdp
  quality: 85

embed:
  model: text-embedding-mxbai-embed-large-v1
  device: cuda
  gpu_ids: [0, 1, 2, 3]
  batch_size: 128

output: ./my_index
```

```yaml
# Example: web URLs with news preset
source:
  type: web
  urls: ./urls.txt
  preset: news
  concurrency: 200

ingest:
  backend: cdp

embed:
  model: text-embedding-mxbai-embed-large-v1
  gpu_ids: [0, 1]

output: ./news_index
```

```yaml
# Example: PDF collection
source:
  type: pdf
  path: ./papers/
  dpi: 300
  pages: "1-10"

embed:
  model: text-embedding-mxbai-embed-large-v1
  device: cpu

output: ./paper_index
```

```yaml
# Example: Wikipedia (distributed)
source:
  type: kiwix
  zim: ./wikipedia.zim
  serve_url: http://localhost:9454

distributed:
  bucket: my-bucket
  prefix: kiwix

embed:
  model: text-embedding-mxbai-embed-large-v1
  gpu_ids: [0, 1, 2, 3, 4, 5, 6, 7]
  backend: sglang

output: s3://my-bucket/index
```

**Source of code:**
- `distributed.py` ← `coordinator.py` (S3ShardCoordinator) + claim loop from `coordinator_worker.py` and `embedding_worker.py`
- `sources/kiwix.py` ← `datasources/kiwix.py` (article iteration logic)
- `sources/web.py` ← `datasources/news.py` + `news/download.py` + `news/db.py` (generalized, news-specific naming removed; download + SQLite state are internal implementation details of this source)
- `sources/local.py` — new
- `sources/pdf.py` — new (thin, delegates rendering to ingest)
- `pipelines.py` ← new, chains stages
- `monitor.py` ← `scripts/monitor_global.py`
- `config.py` — new

### Package 4: pixelrag-serve

**"FAISS index → search API."** One unified FastAPI server that serves any index.

```
src/pixelrag_serve/
└── api.py        # Unified search API
                  # POST /search — text/image/embedding queries → top-k results
                  # GET /health, GET /status
                  # Configurable via CLI args or env vars
                  # Supports CPU and CUDA for query embedding
```

**Dependencies:** fastapi, uvicorn, faiss-cpu, torch, transformers, pillow, numpy

**CLI entry points:**
- `pixelrag-serve` → `pixelrag_serve.api:main`

**Source of code:**
- `api.py` ← merge of `search_api.py` + `text_search_api.py` + `news_search_api.py` into one unified API. Hex ID mapping handled at index build time (in pixelrag-embed), not at serve time.

### Package 5: pixelrag-train

**"Train visual embedding models."**

```
src/pixelrag_train/
├── models/
│   └── biqwen3.py       # BiQwen3: Qwen3VLModel + last-token pooling + L2 norm
├── contrastive.py       # GradCache contrastive training with LoRA/DoRA
└── mine.py              # Hard negative mining (calls serve API)
                         # Unified: image mining (:30888) + text mining (:30889)
```

**Dependencies:** torch, transformers, peft, accelerate, wandb, faiss-cpu

**CLI entry points:**
- `pixelrag-train` → `pixelrag_train.contrastive:main`
- `pixelrag-mine` → `pixelrag_train.mine:main`

**Source of code:**
- `biqwen3.py` ← `models/biqwen3.py` (unchanged)
- `contrastive.py` ← `train_contrastors.py` (renamed)
- `mine.py` ← merge of `mine_hard_negatives.py` + `mine_text_hard_negatives.py`

### eval/

Not a package. Script directory for paper reproduction (User B).

```
eval/
├── run_naive_simpleqa.py    # Main eval runner
└── simpleqa/                # Support library (data, llm, retrieval, etc.)
```

Source: kept from current repo, unchanged.

## Dependency Graph

```
pixelrag-index
├── pixelrag-render (calls render_url/render_pdf for capture stage)
├── pixelrag-embed (calls chunk/embed/index tools)
└── boto3 (S3 coordination)

pixelrag-serve (independent — no deps on other pixelrag packages)

pixelrag-train
└── calls pixelrag-serve API over HTTP (not a Python dependency)

pixelrag-render (independent)
pixelrag-embed (independent)
```

## Data Flow

```
User A: "Build me a visual search index"

  pixelrag-index build --source ./my_docs
       │
       ├─ sources/local.py scans directory, classifies files
       │
       ├─ For each document:
       │    pixelrag-render.render_url() or render_pdf()
       │    → tiles/{doc_id}.tiles/tile_0000.jpg, tile_0001.jpg, ...
       │
       ├─ pixelrag-embed.chunk
       │    → chunks/{doc_id}.tiles/chunk_0000_00.png, ...
       │
       ├─ pixelrag-embed.embed (GPU)
       │    → embeddings/shard_NNN.npz
       │
       └─ pixelrag-embed.index
            → output/index.faiss + metadata.npz

  pixelrag-serve --index-dir ./output --port 30888
       → POST /search {"queries": [{"text": "..."}]} → top-k results
```

## What Gets Cut

From the current ~/pixelrag/ repo:
- `packages/capture/` → replaced by `packages/render/` (new structure)
- `packages/serving/` → replaced by `packages/serve/` (unified API)
- `packages/training/` → replaced by `packages/train/` (renamed files)
- `packages/embed/` — new package (from loose scripts)
- `packages/index/` — new package (from loose scripts + new code)
- `eval/` — kept

From source repos (~/pixelrag-src/), code NOT carried forward:
- `executors/base.py`, `executors/skypilot.py` — executor ABC and cloud-specific code
- `proxy/` — proxy rotation (not needed for offline rendering)
- `lead_images/` — lead image extraction (hardcoded paths)
- `datasources/enterprise.py`, `datasources/wikimedia.py` — paid API / superseded
- `tools/streaming_capture.py` — superseded by lean CDP backend
- `tools/raw_pixels.py`, `tools/temp_dirs.py` — helpers for old PlaywrightTool
- Most of PlaywrightTool's 2388 lines — stripped to production config
- `run.py`, `monitor.py` (top-level) — replaced by index CLI
- `scripts/run_embeddings.py` — thin wrapper, redundant with embed CLI
- `scripts/status.py` — replaced by monitor

## Migration Strategy

1. Create new package directories under ~/pixelrag/packages/
2. Copy + transform code from ~/pixelrag-src/ (source repos are read-only)
3. For each package: create pyproject.toml, rename imports, add CLI entry points
4. Verify `uv sync --package <name>` works for each
5. Verify existing endpoint (port 30001) still works with new pixelrag-serve
6. Commit as clean restructure
