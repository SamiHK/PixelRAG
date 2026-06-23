# Reproducing Paper Results

> **Paper**: *PixelRAG: Retrieval and Generation in Pixel Space over Millions of Web Screenshots*
>
> This document maps every table and figure in the paper to the exact commands needed to reproduce the numbers.

## Prerequisites

### Infrastructure

| Component | Description | Where |
|-----------|-------------|-------|
| **Wikipedia tile index (base)** | 28M vectors, text-embedding-mxbai-embed-large-v1 (pretrained) | `pixelrag-data/search_index/` (215 GB FAISS IVF, dim=2048) |
| **Wikipedia tile index (fine-tuned)** | 26M vectors, LoRA checkpoint-200 | `pixelrag-data/search_index_lora_vit_ckpt200_v2/` (202 GB) |
| **Wikipedia text index** | 15.7M text chunks (1024 tokens, Trafilatura) | `pixelrag-data/text_search_index_1024/` (121 GB) |
| **Article metadata** | URL↔tile mapping for 7.1M articles | `pixelrag-data/articles.json` (199 MB) |
| **Tile images** | ~30M PNG tiles (1024×1024) | Remote NFS or local SSD (~5.6 TB) |
| **News tile index** | 3.6M tiles (BBC/AP/CNN) for LiveVQA | S3: `s3://wiki-screenshot-tiles-backup/kiwix_tiles/news_image_search_index/` |
| **News text index** | 866K text chunks for news | S3: `s3://wiki-screenshot-tiles-backup/kiwix_tiles/news_text_search_index/` |
| **News tiles** | Raw PNG tiles for news articles | S3: `s3://wiki-screenshot-tiles-backup/kiwix_tiles/news_tiles/` |
| **LoRA adapter** | Fine-tuned embedding LoRA weights | S3: `s3://wiki-screenshot-tiles-backup/kiwix_tiles/adapters/lora_vit_ckpt200/` |
| **Kiwix ZIM** | Offline Wikipedia for HTML baselines | S3: `s3://wiki-screenshot-tiles-backup/kiwix_tiles/zim/` |

All S3 paths use AWS profile `leann` (`aws s3 --profile leann ...`).

### Services to Start

```bash
# 1. Screenshot search API (port 30888) — serves the pixel tile index
pixelrag-serve \
    --index-dir pixelrag-data/search_index \           # or search_index_lora_vit_ckpt200_v2
    --tiles-dir /path/to/wikipedia_tiles \
    --articles-json pixelrag-data/articles.json \
    --model text-embedding-mxbai-embed-large-v1 \
    --device cuda --port 30888

# 2. Text search API (port 30889) — serves the text chunk index
pixelrag-serve \
    --index-dir pixelrag-data/text_search_index_1024 \
    --tiles-dir /path/to/text_chunks \
    --articles-json pixelrag-data/articles.json \
    --model text-embedding-mxbai-embed-large-v1 \
    --device cuda --port 30889

# 3. Reader model (port 8000) — vLLM serving Qwen3.5-4B (default reader)
vllm serve Qwen/Qwen3.5-4B-Instruct \
    --port 8000 --tensor-parallel-size 1 \
    --max-model-len 32768
```

### Environment

```bash
cd ~/pixelrag/eval

# Install eval dependencies (one-time)
uv pip install pandas tqdm trafilatura openai aiohttp datasets huggingface-hub

# For grading
export OPENAI_API_KEY=sk-...   # GPT-4.1 judge
```

---

## Table 1: Main Results (6 Benchmarks × 4 Methods)

**Reader**: Qwen3.5-4B, **k=3**, **Grader**: GPT-4.1 judge (except LiveVQA = exact match)

### No Retrieval (baseline)

```bash
# SimpleQA — no retrieval
python run_bench.py \
    --task simpleqa --model Qwen/Qwen3.5-4B-Instruct \
    --num-examples 1000 --no-think

# NQ — no retrieval
python run_bench.py \
    --task nq --model Qwen/Qwen3.5-4B-Instruct \
    --num-examples 1000 --no-think

# NQ-Tables — no retrieval
python run_bench.py \
    --task nq_tables --model Qwen/Qwen3.5-4B-Instruct \
    --num-examples 1000 --no-think

# MMSearch — no retrieval (300 examples)
python run_bench.py \
    --task mmsearch --model Qwen/Qwen3.5-4B-Instruct \
    --num-examples 300 --no-think

# EVQA — no retrieval (landmarks, automatic only, n=749)
python run_bench.py \
    --task encyclopedic_vqa --model Qwen/Qwen3.5-4B-Instruct \
    --evqa-dataset-filter landmarks --evqa-question-type-filter automatic \
    --num-examples 749 --no-think

# LiveVQA — see "LiveVQA Separate Pipeline" section below
```

### Text Retrieval — Trafilatura (Text → Text)

Requires: text search API on port 30889 with Trafilatura-parsed text chunks.

```bash
# SimpleQA — Trafilatura text retrieval
python run_bench.py \
    --task simpleqa --model Qwen/Qwen3.5-4B-Instruct \
    --text-api --text-api-url http://localhost:30889/search \
    --retrieval-top-k 3 --num-examples 1000 --no-think

# NQ — Trafilatura text retrieval
python run_bench.py \
    --task nq --model Qwen/Qwen3.5-4B-Instruct \
    --text-api --text-api-url http://localhost:30889/search \
    --retrieval-top-k 3 --num-examples 1000 --no-think

# NQ-Tables
python run_bench.py \
    --task nq_tables --model Qwen/Qwen3.5-4B-Instruct \
    --text-api --text-api-url http://localhost:30889/search \
    --retrieval-top-k 3 --num-examples 1000 --no-think

# MMSearch (multimodal query: text + image → text index)
python run_bench.py \
    --task mmsearch --model Qwen/Qwen3.5-4B-Instruct \
    --text-api --text-api-url http://localhost:30889/search \
    --retrieval-top-k 3 --num-examples 300 --no-think

# EVQA
python run_bench.py \
    --task encyclopedic_vqa --model Qwen/Qwen3.5-4B-Instruct \
    --text-api --text-api-url http://localhost:30889/search \
    --evqa-dataset-filter landmarks --evqa-question-type-filter automatic \
    --retrieval-top-k 3 --num-examples 749 --no-think

# LiveVQA — see "LiveVQA Separate Pipeline" section below
```

### Text Retrieval — mwparserfromhell

Same as Trafilatura but requires a separate text index built with mwparserfromhell parser.
The text API must be started pointing to that index.

```bash
# Same commands as Trafilatura above, but --text-api-url points to
# the mwparserfromhell text index API (different port or index-dir).
# The parser choice is baked into the index at build time, not a runtime flag.
```

### PixelRAG (base) — Screenshot → Screenshot

Requires: screenshot search API on port 30888 with base (pretrained) embedding index.

```bash
# SimpleQA — pixel retrieval (base)
python run_bench.py \
    --task simpleqa --model Qwen/Qwen3.5-4B-Instruct \
    --local-api --local-api-url http://localhost:30888/search \
    --retrieval-top-k 3 --num-examples 1000 --no-think

# NQ — pixel retrieval (base)
python run_bench.py \
    --task nq --model Qwen/Qwen3.5-4B-Instruct \
    --local-api --local-api-url http://localhost:30888/search \
    --retrieval-top-k 3 --num-examples 1000 --no-think

# NQ-Tables
python run_bench.py \
    --task nq_tables --model Qwen/Qwen3.5-4B-Instruct \
    --local-api --local-api-url http://localhost:30888/search \
    --retrieval-top-k 3 --num-examples 1000 --no-think

# MMSearch (multimodal: query image sent alongside text)
python run_bench.py \
    --task mmsearch --model Qwen/Qwen3.5-4B-Instruct \
    --local-api --local-api-url http://localhost:30888/search \
    --retrieval-top-k 3 --num-examples 300 --no-think

# EVQA (multimodal: landmark photo + question text)
python run_bench.py \
    --task encyclopedic_vqa --model Qwen/Qwen3.5-4B-Instruct \
    --local-api --local-api-url http://localhost:30888/search \
    --evqa-dataset-filter landmarks --evqa-question-type-filter automatic \
    --retrieval-top-k 3 --num-examples 749 --no-think

# LiveVQA — see "LiveVQA Separate Pipeline" section below
```

### PixelRAG (fine-tuned) — Screenshot → Screenshot with LoRA embedding

Same commands as PixelRAG (base), but the search API must be started with the fine-tuned index:

```bash
# Start search API with fine-tuned index
pixelrag-serve \
    --index-dir pixelrag-data/search_index_lora_vit_ckpt200_v2 \
    --tiles-dir /path/to/wikipedia_tiles \
    --articles-json pixelrag-data/articles.json \
    --model text-embedding-mxbai-embed-large-v1 \
    --peft-adapter /path/to/lora_checkpoint_200 \
    --device cuda --port 30888
```

Then run the same `--local-api` commands above.

### Grading

```bash
cd ~/pixelrag/eval

# Grade with GPT-4.1 judge (Wikipedia QA tasks)
python grade.py simpleqa eval_output/simpleqa_*.jsonl
python grade.py encyclopedic_vqa eval_output/encyclopedic_vqa_*.jsonl
python grade.py mmsearch eval_output/mmsearch_*.jsonl

# For NQ/NQ-Tables (with LLM judge for paper numbers)
python grade.py nq eval_output/nq_*.jsonl --llm-judge
python grade.py nq_tables eval_output/nq_tables_*.jsonl --llm-judge

# For LiveVQA (exact letter match — handled by the LiveVQA pipeline scripts)
```

---

## Table 3: Retrieval–Reader Modality Ablation

**Task**: SimpleQA (1000) + LiveVQA (6632), **Reader**: Qwen3.5-4B, **k=3**,
**Embedding**: text-embedding-mxbai-embed-large-v1 (base, no LoRA)

| Row | Retrieval | Reader Input | Flags |
|-----|-----------|-------------|-------|
| Screenshot → Screenshot | Pixel index | Raw tile images | `--local-api` |
| Screenshot → OCR text | Pixel index | OCR'd text from tiles | `--local-api --read-as-text-ocr` |
| Text → Rendered image | Text index | Text chunks rendered as PNG | `--text-api --render-as-image` |
| Text → Text | Text index | Raw text chunks | `--text-api` |
| Text → HTML | Text index | Raw HTML from kiwix | `--text-api --html-dom-lookup` |

```bash
# Screenshot → Screenshot (same as main results PixelRAG base)
python run_bench.py \
    --task simpleqa --model Qwen/Qwen3.5-4B-Instruct \
    --local-api --local-api-url http://localhost:30888/search \
    --retrieval-top-k 3 --num-examples 1000 --no-think

# Screenshot → OCR text
python run_bench.py \
    --task simpleqa --model Qwen/Qwen3.5-4B-Instruct \
    --local-api --local-api-url http://localhost:30888/search \
    --read-as-text-ocr --ocr-url http://localhost:8202/v1 \
    --retrieval-top-k 3 --num-examples 1000 --no-think

# Text → Rendered image
python run_bench.py \
    --task simpleqa --model Qwen/Qwen3.5-4B-Instruct \
    --text-api --text-api-url http://localhost:30889/search \
    --render-as-image \
    --retrieval-top-k 3 --num-examples 1000 --no-think

# Text → Text (same as main results Trafilatura)
python run_bench.py \
    --task simpleqa --model Qwen/Qwen3.5-4B-Instruct \
    --text-api --text-api-url http://localhost:30889/search \
    --retrieval-top-k 3 --num-examples 1000 --no-think

# Text → HTML (DOM lookup)
python run_bench.py \
    --task simpleqa --model Qwen/Qwen3.5-4B-Instruct \
    --text-api --text-api-url http://localhost:30889/search \
    --html-dom-lookup \
    --retrieval-top-k 3 --num-examples 1000 --no-think
```

For LiveVQA, use the separate pipeline (see "LiveVQA Separate Pipeline" section) with the corresponding ablation scripts.

---

## Table 4: Embedding Training Recipe Ablation

**Evaluated on mini-datastore** (400 queries, 7426 tiles).

This ablation uses `--prebuilt-tiles-dir` pointing to the pre-built mini-datastore, with different embedding checkpoints. Each row corresponds to a different embedding training recipe:

```bash
# Base model (no fine-tuning)
python run_bench.py \
    --task simpleqa --model Qwen/Qwen3.5-4B-Instruct \
    --use-tiled-retrieval --use-qwen3vl-embedding \
    --qwen3vl-model text-embedding-mxbai-embed-large-v1 \
    --embedding-backend hf \
    --prebuilt-tiles-dir tiles-hard-mini/ \
    --retrieval-top-k 3 --num-examples 400 --no-think

# With LoRA checkpoint (dynamic hard negatives + ViT unfrozen)
python run_bench.py \
    --task simpleqa --model Qwen/Qwen3.5-4B-Instruct \
    --use-tiled-retrieval --use-qwen3vl-embedding \
    --qwen3vl-model text-embedding-mxbai-embed-large-v1 \
    --embedding-backend biqwen3 \
    --peft-adapter /path/to/checkpoint-200 \
    --prebuilt-tiles-dir tiles-hard-mini/ \
    --retrieval-top-k 3 --num-examples 400 --no-think
```

The intermediate checkpoints (in-batch negatives, naive hard negatives, dynamic hard negatives frozen) each have their own PEFT adapter path.

---

## Figure 2: Token Efficiency (SimpleQA, k=1,2,3, 4 readers)

**Task**: SimpleQA (1000), **Readers**: Qwen3.5-4B, Qwen3.5-9B, Qwen3.5-27B, Qwen3.6-35B-A3B

For each reader × k × retrieval method, run:

```bash
# Example: Qwen3.5-4B, k=1, PixelRAG (fine-tuned)
python run_bench.py \
    --task simpleqa --model Qwen/Qwen3.5-4B-Instruct \
    --local-api --local-api-url http://localhost:30888/search \
    --retrieval-top-k 3 --reader-top-k 1 \
    --num-examples 1000 --no-think

# Example: Qwen3.5-4B, k=2, PixelRAG (fine-tuned)
python run_bench.py \
    --task simpleqa --model Qwen/Qwen3.5-4B-Instruct \
    --local-api --local-api-url http://localhost:30888/search \
    --retrieval-top-k 3 --reader-top-k 2 \
    --num-examples 1000 --no-think

# Example: Qwen3.5-4B, k=3, PixelRAG (fine-tuned)
python run_bench.py \
    --task simpleqa --model Qwen/Qwen3.5-4B-Instruct \
    --local-api --local-api-url http://localhost:30888/search \
    --retrieval-top-k 3 \
    --num-examples 1000 --no-think
```

> **Optimization**: Use `--retrieval-top-k 3 --reader-top-k N` to retrieve once at k=3 and evaluate at k=1,2,3 from the same JSONL (the full retrieved set is stored in `retrieved_images`).

For each reader, change `--model` and start the appropriate vLLM server.
Repeat for text retrieval (Trafilatura: `--text-api`) and PixelRAG base (base index).

The plot script is at `arxiv/figures/plot_token_efficiency.py`.

---

## Figure 3: Agentic Multi-Hop QA (MoNaCo)

**Task**: MoNaCo (1315 questions), **Agent**: GPT-5 ReAct, **k=5 per search**

Uses `eval/run_monaco.py` — a ReAct agent that issues search tool calls.

```bash
cd ~/pixelrag/eval

# PixelRAG backend
python run_monaco.py \
    --reader gpt-5 \
    --retrieval pixel \
    --pixel-api http://localhost:30888/search \
    --default-top-k 5

# Text retrieval backend (Trafilatura)
python run_monaco.py \
    --reader gpt-5 \
    --retrieval text \
    --text-api http://localhost:30889/search \
    --default-top-k 5

# Grade (token F1 computed inline; add --judge for LLM judge F1)
python run_monaco.py \
    --reader gpt-5 \
    --retrieval pixel \
    --judge --judge-model gpt-4.1-2025-04-14

# Or grade existing predictions:
python grade.py monaco eval_output/monaco/<run_tag>
```

The dataset (`monaco_version_1_release.jsonl`) should be placed at
`eval/data/monaco/` or passed via `--data-path`.

---

## Figure 4: Image Compression Curve

**Task**: SimpleQA (1000), **Reader**: Qwen3.5-4B (base + SFT), k=1..5, compression c=1×/2×/3×

```bash
# No compression (c=1×), k=3
python run_bench.py \
    --task simpleqa --model Qwen/Qwen3.5-4B-Instruct \
    --local-api --local-api-url http://localhost:30888/search \
    --retrieval-top-k 5 --reader-top-k 3 \
    --num-examples 1000 --no-think

# 2× compression, k=3
python run_bench.py \
    --task simpleqa --model Qwen/Qwen3.5-4B-Instruct \
    --local-api --local-api-url http://localhost:30888/search \
    --retrieval-top-k 5 --reader-top-k 3 \
    --pixel-compress-ratio 2.0 \
    --num-examples 1000 --no-think

# 3× compression, k=3
python run_bench.py \
    --task simpleqa --model Qwen/Qwen3.5-4B-Instruct \
    --local-api --local-api-url http://localhost:30888/search \
    --retrieval-top-k 5 --reader-top-k 3 \
    --pixel-compress-ratio 3.0 \
    --num-examples 1000 --no-think
```

For the SFT reader, replace `--model` with the SFT checkpoint path and serve it via vLLM.

The plot script is at `arxiv/figures/plot_sft_compression_curve.py`.

---

## Table 8: Full Reader-Model Sweep (31 VLMs)

**Task**: SimpleQA (1000), **k=3**, pixel retrieval (base) vs text retrieval (Trafilatura)

For each of the 31 reader models, run two jobs:

```bash
# Pixel retrieval
python run_bench.py \
    --task simpleqa --model <MODEL_NAME> \
    --local-api --local-api-url http://localhost:30888/search \
    --retrieval-top-k 3 --num-examples 1000 --no-think

# Text retrieval
python run_bench.py \
    --task simpleqa --model <MODEL_NAME> \
    --text-api --text-api-url http://localhost:30889/search \
    --retrieval-top-k 3 --num-examples 1000 --no-think
```

where `<MODEL_NAME>` is one of:
- `liuhaotian/llava-v1.5-7b`
- `meta-llama/Llama-3.2-11B-Vision-Instruct` (k=1 for pixel due to architecture limit)
- `meta-llama/Llama-3.2-90B-Vision-Instruct` (k=1 for pixel)
- `meta-llama/Llama-4-Scout-17B-16E-Instruct`
- `meta-llama/Llama-4-Maverick-17B-128E-Instruct`
- `Qwen/Qwen2-VL-2B-Instruct` through `Qwen/Qwen2-VL-72B-Instruct`
- `Qwen/Qwen2.5-VL-3B-Instruct` through `Qwen/Qwen2.5-VL-72B-Instruct`
- `Qwen/Qwen3-VL-2B` through `Qwen/Qwen3-VL-235B-A22B`
- `Qwen/Qwen3.5-0.8B` through `Qwen/Qwen3.5-35B-A3B`
- `Qwen/Qwen3.6-27B`, `Qwen/Qwen3.6-35B-A3B`

For reasoning-mode models, omit `--no-think`.

Each model requires its own vLLM instance (or OpenRouter/Commonstack for API models).

---

## LiveVQA (Table 1 + Table 3)

LiveVQA uses `eval/run_livevqa.py` — a dedicated script for the news corpus.

**Requires**: News pixel search API (port 30890), news text search API (port 30892),
LiveVQA v4 JSON dataset, vLLM reader.

```bash
cd ~/pixelrag/eval

# No retrieval
python run_livevqa.py --mode naive \
    --model Qwen/Qwen3.5-4B-Instruct \
    --output eval_output/livevqa_naive.jsonl

# PixelRAG (screenshot → screenshot)
python run_livevqa.py --mode pixel \
    --pixel-api http://localhost:30890/search \
    --model Qwen/Qwen3.5-4B-Instruct \
    --output eval_output/livevqa_pixel.jsonl

# Text retrieval (Trafilatura)
python run_livevqa.py --mode text \
    --text-api http://localhost:30892/search \
    --model Qwen/Qwen3.5-4B-Instruct \
    --output eval_output/livevqa_text.jsonl

# Hybrid (pixel + text)
python run_livevqa.py --mode hybrid \
    --pixel-api http://localhost:30890/search \
    --text-api http://localhost:30892/search \
    --model Qwen/Qwen3.5-4B-Instruct \
    --output eval_output/livevqa_hybrid.jsonl
```

Grading is automatic (5-option MC exact letter match) — printed at the end of each run.

---

## Known Issues (Blockers for Reproduction)

### ~~0. Missing simpleqa modules~~ (FIXED)

`screenshot.py` and `pixel_query.py` have been copied into `eval/lib/`.
Selenium import is deferred so it doesn't block `--local-api` users.

### ~~1. `dr_agent` not importable~~ (FIXED)

Dataset loaders extracted into `eval/lib/benchmarks.py`. The `run_bench.py`
import now reads from `simpleqa.datasets_loader` instead of `dr_agent`.

### ~~2. Grading script not in this repo~~ (FIXED)

`eval/grade.py` implements GPT-4.1 3-way grading (CORRECT/INCORRECT/NOT_ATTEMPTED) using
the same prompt template as the paper. No dependency on the old repo's evaluation framework.

For the legacy full evaluation framework (per-example HTML reports, etc.), the original
is still at `~/pixelrag-src/Vis-RAG/agent/scripts/evaluate.py`.

### 3. Hardcoded paths in retrieval.py

`eval/lib/retrieval.py` lines 84–88 have placeholder paths (`/path/to/project`, `/path/to/data`) for the local kiwix tile store. These are only used by `LocalWikiTiledScreenshotRetriever` (ground-truth screenshot mode), not by the production `--local-api` mode.

### ~~4. LiveVQA uses separate pipeline~~ (FIXED)

`eval/run_livevqa.py` handles all LiveVQA modes (naive, pixel, text, hybrid).

### ~~5. MoNaCo runs from old repo~~ (FIXED)

`eval/run_monaco.py` implements the full ReAct agent loop with pixel/text retrieval backends.

### 6. mwparserfromhell text index

The paper's second text baseline uses mwparserfromhell parser. The text index must be built separately with this parser — the parser choice is embedded at index build time, not at query time. The build pipeline for this variant needs to be documented.

### 7. News corpus indexes

LiveVQA requires separate tile and text indexes built over the news corpus (BBC/AP/CNN). These indexes are on a different machine/path and need their own `pixelrag-serve` instances.

---

## Grading Protocol Summary

| Benchmark | Metric | Grader |
|-----------|--------|--------|
| SimpleQA | CORRECT/INCORRECT/NOT_ATTEMPTED → accuracy | GPT-4.1 (temp=0, seed=42) |
| NQ | Same 3-way judge | GPT-4.1 (temp=0, seed=42) |
| NQ-Tables | Same 3-way judge (up to 10 gold aliases joined with OR) | GPT-4.1 |
| MMSearch | Same 3-way judge | GPT-4.1 |
| EVQA | Same 3-way judge (reference_list → "Any of: ref1 \| ref2") | GPT-4.1 |
| LiveVQA | 5-option multiple-choice exact letter match | No LLM |
| MoNaCo | Token-level F1 (primary), LLM judge F1 (secondary) | GPT-4.1 |

---

## Quick Smoke Test (Verify Pipeline Works)

Run a single example end-to-end before committing to full runs:

```bash
# 1. Verify search API is responding
curl -s http://localhost:30888/status | python -m json.tool

# 2. Run 5 examples, no retrieval
python run_bench.py \
    --task simpleqa --model Qwen/Qwen3.5-4B-Instruct \
    --num-examples 5 --no-think --force

# 3. Run 5 examples, pixel retrieval
python run_bench.py \
    --task simpleqa --model Qwen/Qwen3.5-4B-Instruct \
    --local-api --local-api-url http://localhost:30888/search \
    --retrieval-top-k 3 --num-examples 5 --no-think --force

# 4. Grade
cd ~/pixelrag-src/Vis-RAG/agent
python scripts/evaluate.py simpleqa ~/pixelrag/eval/eval_output/<output>.jsonl
```

---

## Output File Convention

All outputs go to `eval_output/` with auto-generated filenames:

```
eval_output/{task}_{mode}_{model_safe}_{n}.jsonl
```

Examples:
- `eval_output/simpleqa_naive_qwen_qwen3.5_4b_instruct_1000.jsonl`
- `eval_output/simpleqa_local_api_qwen_qwen3.5_4b_instruct_1000.jsonl`
- `eval_output/nq_text_api_qwen_qwen3.5_4b_instruct_1000.jsonl`

Grading results are saved alongside as `*_eval_results.json`.
