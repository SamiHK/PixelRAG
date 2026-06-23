---
name: pixelrag
description: Visual search over documents. Use when the user wants to capture screenshots of web pages, search visual content, or build visual retrieval indexes. Triggers on: "screenshot this URL", "search Wikipedia visually", "find documents about X", "capture this page", "build a visual index".
---

# PixelRAG — Visual Retrieval-Augmented Generation

You have access to a visual document retrieval system. Use it when the user needs to:
- **Capture** a web page or document as tiled screenshot images
- **Search** for visually relevant content in pre-built indexes (Wikipedia, news, custom)
- **Build** a searchable visual index from documents

## Available Tools

### 1. Capture a URL

Render any web page to tiled JPEG screenshots:

```bash
cd ~/pixelrag
uv run pixelshot <URL> --output ./tiles
```

Or from Python:
```python
from pixelrag_render import render_url
tiles = render_url("https://en.wikipedia.org/wiki/Python", "./tiles")
```

Output: `{output_dir}/{stem}.png.tiles/tile_NNNN.jpg` + `tiles.json` manifest.

### 2. Search an Index

Query the running search API (must be started first):

```bash
curl -s -X POST http://localhost:30001/search \
    -H "Content-Type: application/json" \
    -d '{"queries": [{"text": "YOUR QUERY"}], "n_docs": 5}'
```

The API returns JSON with hits:
```json
{
  "results": [{
    "hits": [
      {"score": 0.73, "url": "https://en.wikipedia.org/wiki/...", "article_id": 123, ...}
    ]
  }]
}
```

Available endpoints (if running):
- `:30001` — Wikipedia text chunks (15.7M vectors)
- `:30002` — Wikipedia pixel screenshots (28M vectors)
- `:30003` — Wikipedia LoRA+ViT pixel (28M vectors)

### 3. Build an Index

Create a searchable visual index from any document source:

```bash
cd ~/pixelrag

# Create pixelrag.yaml
cat > pixelrag.yaml << 'EOF'
source:
  type: local        # or: kiwix, web, pdf
  path: ./my_docs

embed:
  model: text-embedding-mxbai-embed-large-v1
  device: cpu         # or: cuda

output: ./my_index
EOF

uv run pixelrag index build --config pixelrag.yaml --limit 100
```

Then serve it:
```bash
PIXELRAG_INDEX_DIR=./my_index PIXELRAG_ARTICLES_JSON=./my_index/articles.json \
uv run pixelrag serve --port 31337
```

### 4. Start/Check Serving

```bash
# Check if search API is running
curl -s http://localhost:30001/health

# Start serving a pre-built index
PIXELRAG_INDEX_DIR=/home/yichuan/pixelrag-data/text_search_index_1024 \
PIXELRAG_ARTICLES_JSON=/home/yichuan/pixelrag-data/articles.json \
uv run pixelrag serve --port 30001 &
```

## When to Use

- User asks to **find information** about a topic → search the index
- User shares a **URL** and wants to see/capture it → use ingest
- User has **documents** and wants them searchable → build an index
- User asks about **Wikipedia** content → search the pre-built Wikipedia index
- User wants to **compare** visual vs text retrieval → search both `:30001` (text) and `:30002` (pixel)

## Tips

- The search API embeds queries on CPU (~1-2s per query). For faster queries, use GPU.
- Pre-built Wikipedia indexes are at `/home/yichuan/pixelrag-data/`.
- The ingest CDP backend is fastest (~1s per page). Playwright backend has more options.
- For large-scale embedding, use GPU machines with `pixelrag embed` (vLLM/sglang backend).
