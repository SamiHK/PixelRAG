# PixelRAG Frontend Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a modern Next.js frontend for the PixelRAG visual retrieval engine — search page with image grid, API docs, and index dashboard.

**Architecture:** Standalone Next.js 15 app in `web/` directory. Communicates with existing FastAPI backend via API proxy rewrites in dev, CORS in production. Dark theme, indigo accent, image-first design.

**Tech Stack:** Next.js 15 (App Router), Tailwind CSS 4, shadcn/ui, Framer Motion, TypeScript

**Spec:** `docs/superpowers/specs/2026-05-25-pixelrag-frontend-design.md`

---

## File Map

### New files (all under `web/`)

| File | Responsibility |
|------|---------------|
| `web/src/lib/types.ts` | TypeScript types mirroring FastAPI Pydantic models |
| `web/src/lib/api.ts` | Typed fetch wrapper for all backend endpoints |
| `web/src/app/layout.tsx` | Root layout: nav bar, fonts, theme |
| `web/src/app/page.tsx` | Search home page: SearchBar + results |
| `web/src/app/status/page.tsx` | Index dashboard |
| `web/src/app/docs/page.tsx` | API documentation |
| `web/src/components/SearchBar.tsx` | Text + image input with drag-drop |
| `web/src/components/TileCard.tsx` | Single tile result card |
| `web/src/components/ResultGroup.tsx` | Article group with horizontal tile row |
| `web/src/components/Lightbox.tsx` | Fullscreen tile viewer with pan/zoom/nav |
| `web/src/components/ComparePanel.tsx` | Side-by-side tile comparison |
| `web/src/components/ApiPlayground.tsx` | Try-it-live widget for docs page |
| `web/src/components/StatusCard.tsx` | Metric display card |

### Modified files

| File | Change |
|------|--------|
| `serve/src/pixelrag_serve/api.py` | Add CORS middleware (3 lines) |

---

## Phase 1: Core Search (MVP)

### Task 1: Project Scaffolding

**Files:**
- Create: `web/` (entire Next.js project via CLI)
- Modify: `web/src/app/globals.css` (custom theme tokens)
- Modify: `web/next.config.ts` (API proxy rewrites)
- Modify: `web/postcss.config.mjs` (verify Tailwind v4 plugin)

- [ ] **Step 1: Scaffold Next.js project with shadcn/ui**

```bash
cd /home/yichuan/pixelrag
npx shadcn@latest init -t next web
```

When prompted, accept defaults. This creates a Next.js 15 + Tailwind CSS 4 + shadcn/ui project in `web/`.

- [ ] **Step 2: Verify the scaffold built correctly**

```bash
cd /home/yichuan/pixelrag/web && npm run build
```

Expected: Build succeeds with no errors.

- [ ] **Step 3: Install additional dependencies**

```bash
cd /home/yichuan/pixelrag/web && npm install framer-motion
```

- [ ] **Step 4: Add shadcn/ui components we'll need**

```bash
cd /home/yichuan/pixelrag/web
npx shadcn@latest add button input badge collapsible dialog slider
```

- [ ] **Step 5: Configure API proxy rewrites**

Replace `web/next.config.ts` with:

```ts
import type { NextConfig } from "next";

const nextConfig: NextConfig = {
  async rewrites() {
    return [
      {
        source: "/api/:path*",
        destination: "http://localhost:30001/:path*",
      },
    ];
  },
};

export default nextConfig;
```

- [ ] **Step 6: Set up custom theme in globals.css**

Replace the `@theme` block in `web/src/app/globals.css` with the PixelRAG color palette. Keep the existing `@import "tailwindcss"` and shadcn layers. Add the custom theme tokens:

```css
@import "tailwindcss";

@theme inline {
  --color-background: #0c0c0c;
  --color-surface: #1a1a1a;
  --color-border: #222222;
  --color-foreground: #ffffff;
  --color-muted: #888888;
  --color-muted-foreground: #555555;
  --color-accent: #6366f1;
  --color-accent-light: #8b5cf6;
  --color-score: #6366f1;
  --color-method-get: #3b82f6;
  --color-method-post: #22c55e;

  --font-sans: "Inter", ui-sans-serif, system-ui, sans-serif;
  --font-display: "Crimson Pro", ui-serif, Georgia, serif;
  --font-mono: "JetBrains Mono", ui-monospace, monospace;
}

/* shadcn overrides for dark theme */
:root {
  color-scheme: dark;
}

body {
  background: var(--color-background);
  color: var(--color-foreground);
  font-family: var(--font-sans);
}
```

Note: The exact format depends on what the shadcn init generated. Preserve any existing shadcn CSS variables and layer imports. The key additions are the custom color tokens and font families.

- [ ] **Step 7: Add Google Fonts**

In `web/src/app/layout.tsx`, add font imports. The shadcn scaffold creates a layout with a font already — modify it to use Inter + Crimson Pro + JetBrains Mono:

```tsx
import { Inter, Crimson_Pro, JetBrains_Mono } from "next/font/google";

const inter = Inter({ subsets: ["latin"], variable: "--font-sans" });
const crimsonPro = Crimson_Pro({ subsets: ["latin"], variable: "--font-display" });
const jetbrainsMono = JetBrains_Mono({ subsets: ["latin"], variable: "--font-mono" });

// In the <body> tag:
<body className={`${inter.variable} ${crimsonPro.variable} ${jetbrainsMono.variable} antialiased`}>
```

- [ ] **Step 8: Verify dev server starts**

```bash
cd /home/yichuan/pixelrag/web && npm run dev &
sleep 3
curl -s http://localhost:3000 | head -20
kill %1
```

Expected: HTML response from Next.js dev server.

- [ ] **Step 9: Commit**

```bash
cd /home/yichuan/pixelrag
git add web/
git commit -m "feat(web): scaffold Next.js 15 + Tailwind 4 + shadcn/ui project"
```

---

### Task 2: Backend CORS Middleware

**Files:**
- Modify: `serve/src/pixelrag_serve/api.py` (lines 46-54)

- [ ] **Step 1: Add CORS middleware to FastAPI**

In `serve/src/pixelrag_serve/api.py`, add after the `app = FastAPI(...)` line (line 54):

```python
from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

The import `CORSMiddleware` should be added to the existing import block at the top. The `from fastapi.middleware.cors import CORSMiddleware` line goes near line 47 with the other fastapi imports.

- [ ] **Step 2: Commit**

```bash
cd /home/yichuan/pixelrag
git add serve/src/pixelrag_serve/api.py
git commit -m "feat(serve): add CORS middleware for frontend dev server"
```

---

### Task 3: TypeScript Types + API Client

**Files:**
- Create: `web/src/lib/types.ts`
- Create: `web/src/lib/api.ts`

- [ ] **Step 1: Create TypeScript types mirroring the Pydantic models**

Create `web/src/lib/types.ts`:

```ts
export interface Query {
  text?: string;
  image?: string; // base64-encoded
  embedding?: number[];
}

export interface SearchRequest {
  queries: Query[];
  n_docs?: number;
  nprobe?: number;
  min_tile_height?: number;
  instruction?: string;
}

export interface Hit {
  score: number;
  vector_id: number;
  article_id: number;
  tile_index: number;
  chunk_index: number;
  y_offset: number;
  tile_height: number;
  path: string;
  url: string;
}

export interface QueryResult {
  hits: Hit[];
}

export interface SearchResponse {
  results: QueryResult[];
}

export interface StatusResponse {
  total_vectors: number;
  dimension: number;
  nlist: number;
  nprobe: number;
  model: string;
  index_dir: string;
  tiles_dir: string;
  index_built_at: string;
  index_size_bytes: number;
  metadata_size_bytes: number;
}

export interface ArticleGroup {
  article_id: number;
  title: string;
  url: string;
  hits: (Hit & { rank: number })[];
}
```

- [ ] **Step 2: Create API client**

Create `web/src/lib/api.ts`:

```ts
import type { SearchRequest, SearchResponse, StatusResponse } from "./types";

const API_BASE = "/api";

async function fetchApi<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, init);
  if (!res.ok) {
    const body = await res.text();
    throw new Error(`API ${res.status}: ${body}`);
  }
  return res.json();
}

export async function search(req: SearchRequest): Promise<SearchResponse> {
  return fetchApi<SearchResponse>("/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(req),
  });
}

export async function getStatus(): Promise<StatusResponse> {
  return fetchApi<StatusResponse>("/status");
}

export async function getHealth(): Promise<{ status: string }> {
  return fetchApi<{ status: string }>("/health");
}

export function tileUrl(path: string): string {
  return `${API_BASE}/tile?path=${encodeURIComponent(path)}`;
}

export async function reconstruct(
  vectorIds: number[]
): Promise<{ embeddings: number[][] }> {
  return fetchApi<{ embeddings: number[][] }>("/reconstruct", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ vector_ids: vectorIds }),
  });
}
```

- [ ] **Step 3: Verify types compile**

```bash
cd /home/yichuan/pixelrag/web && npx tsc --noEmit
```

Expected: No errors.

- [ ] **Step 4: Commit**

```bash
cd /home/yichuan/pixelrag
git add web/src/lib/types.ts web/src/lib/api.ts
git commit -m "feat(web): add TypeScript types and API client"
```

---

### Task 4: Navigation Shell (Layout)

**Files:**
- Modify: `web/src/app/layout.tsx`

- [ ] **Step 1: Build the root layout with nav bar**

Replace `web/src/app/layout.tsx` with the full layout. Keep the font setup from Task 1 Step 7. The nav bar has: logo (left), page links (right: Search, Docs, Status).

```tsx
import type { Metadata } from "next";
import { Inter, Crimson_Pro, JetBrains_Mono } from "next/font/google";
import Link from "next/link";
import "./globals.css";

const inter = Inter({ subsets: ["latin"], variable: "--font-sans" });
const crimsonPro = Crimson_Pro({
  subsets: ["latin"],
  variable: "--font-display",
});
const jetbrainsMono = JetBrains_Mono({
  subsets: ["latin"],
  variable: "--font-mono",
});

export const metadata: Metadata = {
  title: "PixelRAG",
  description: "Visual retrieval over Wikipedia screenshot tiles",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en" className="dark">
      <body
        className={`${inter.variable} ${crimsonPro.variable} ${jetbrainsMono.variable} font-sans antialiased bg-background text-foreground min-h-screen`}
      >
        <nav className="border-b border-border/50 sticky top-0 z-50 bg-background/80 backdrop-blur-sm">
          <div className="max-w-6xl mx-auto px-6 h-14 flex items-center justify-between">
            <Link href="/" className="flex items-center gap-2">
              <span className="font-display text-xl font-semibold tracking-tight">
                Vis<span className="text-accent">RAG</span>
              </span>
            </Link>
            <div className="flex items-center gap-6 text-sm text-muted">
              <Link
                href="/"
                className="hover:text-foreground transition-colors"
              >
                Search
              </Link>
              <Link
                href="/docs"
                className="hover:text-foreground transition-colors"
              >
                API Docs
              </Link>
              <Link
                href="/status"
                className="hover:text-foreground transition-colors"
              >
                Status
              </Link>
            </div>
          </div>
        </nav>
        <main>{children}</main>
      </body>
    </html>
  );
}
```

- [ ] **Step 2: Verify layout renders**

```bash
cd /home/yichuan/pixelrag/web && npm run build
```

Expected: Build succeeds.

- [ ] **Step 3: Commit**

```bash
cd /home/yichuan/pixelrag
git add web/src/app/layout.tsx
git commit -m "feat(web): add navigation shell with header"
```

---

### Task 5: TileCard Component

**Files:**
- Create: `web/src/components/TileCard.tsx`

- [ ] **Step 1: Build the TileCard component**

Create `web/src/components/TileCard.tsx`:

```tsx
"use client";

import Image from "next/image";
import { useState } from "react";
import type { Hit } from "@/lib/types";
import { tileUrl } from "@/lib/api";

interface TileCardProps {
  hit: Hit;
  rank: number;
  selected?: boolean;
  onSelect?: (hit: Hit) => void;
  onClick?: (hit: Hit) => void;
}

export function TileCard({
  hit,
  rank,
  selected,
  onSelect,
  onClick,
}: TileCardProps) {
  const [imgError, setImgError] = useState(false);

  return (
    <div
      className={`relative group min-w-[200px] max-w-[240px] flex-shrink-0 rounded-lg border overflow-hidden cursor-pointer transition-all
        ${selected ? "border-accent ring-1 ring-accent" : "border-border hover:border-border/80"}
        bg-surface`}
      onClick={() => onClick?.(hit)}
    >
      {/* Tile image */}
      <div className="relative w-full aspect-[875/600] bg-background">
        {imgError ? (
          <div className="absolute inset-0 flex items-center justify-center text-muted-foreground text-xs">
            tile {hit.tile_index}:{hit.chunk_index}
          </div>
        ) : (
          <img
            src={tileUrl(hit.path)}
            alt={`tile ${hit.tile_index}:${hit.chunk_index}`}
            className="w-full h-full object-cover object-top"
            loading="lazy"
            onError={() => setImgError(true)}
          />
        )}
      </div>

      {/* Rank badge */}
      <div className="absolute top-1.5 left-1.5 bg-black/70 text-white text-[10px] font-bold px-1.5 py-0.5 rounded">
        #{rank}
      </div>

      {/* Select checkbox (visible on hover or when selected) */}
      {onSelect && (
        <div
          className={`absolute top-1.5 right-1.5 w-5 h-5 rounded border flex items-center justify-center text-xs transition-opacity
            ${selected ? "opacity-100 bg-accent border-accent text-white" : "opacity-0 group-hover:opacity-100 border-white/50 bg-black/50"}`}
          onClick={(e) => {
            e.stopPropagation();
            onSelect(hit);
          }}
        >
          {selected && "✓"}
        </div>
      )}

      {/* Metadata footer */}
      <div className="px-2.5 py-2 flex items-center justify-between">
        <span className="text-xs font-semibold text-score">
          {hit.score.toFixed(3)}
        </span>
        <span className="text-[10px] text-muted-foreground">
          {hit.tile_height}px
        </span>
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Verify it compiles**

```bash
cd /home/yichuan/pixelrag/web && npx tsc --noEmit
```

Expected: No errors.

- [ ] **Step 3: Commit**

```bash
cd /home/yichuan/pixelrag
git add web/src/components/TileCard.tsx
git commit -m "feat(web): add TileCard component"
```

---

### Task 6: ResultGroup Component

**Files:**
- Create: `web/src/components/ResultGroup.tsx`

- [ ] **Step 1: Build the ResultGroup component**

Create `web/src/components/ResultGroup.tsx`:

```tsx
"use client";

import { ExternalLink } from "lucide-react";
import type { Hit, ArticleGroup } from "@/lib/types";
import { TileCard } from "./TileCard";

interface ResultGroupProps {
  group: ArticleGroup;
  selectedHits: Set<number>;
  onSelectHit: (hit: Hit) => void;
  onClickHit: (hit: Hit) => void;
}

export function ResultGroup({
  group,
  selectedHits,
  onSelectHit,
  onClickHit,
}: ResultGroupProps) {
  return (
    <div className="mb-6">
      {/* Article header */}
      <div className="flex items-center gap-2 mb-2.5">
        <h3 className="text-sm font-medium text-foreground">{group.title}</h3>
        {group.url && (
          <a
            href={group.url}
            target="_blank"
            rel="noopener noreferrer"
            className="text-[11px] text-accent hover:underline flex items-center gap-0.5"
          >
            {new URL(group.url).hostname}
            <ExternalLink className="w-3 h-3" />
          </a>
        )}
        <span className="text-[10px] text-muted-foreground bg-surface px-2 py-0.5 rounded-full">
          {group.hits.length} tile{group.hits.length !== 1 && "s"}
        </span>
      </div>

      {/* Horizontal scrollable tile row */}
      <div className="flex gap-2.5 overflow-x-auto pb-2 scrollbar-thin scrollbar-thumb-border scrollbar-track-transparent">
        {group.hits.map((hit) => (
          <TileCard
            key={hit.vector_id}
            hit={hit}
            rank={hit.rank}
            selected={selectedHits.has(hit.vector_id)}
            onSelect={onSelectHit}
            onClick={onClickHit}
          />
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Add a utility to group hits by article**

Add to the bottom of `web/src/lib/types.ts`:

```ts
export function groupHitsByArticle(hits: Hit[]): ArticleGroup[] {
  const map = new Map<number, ArticleGroup>();
  hits.forEach((hit, index) => {
    const ranked = { ...hit, rank: index + 1 };
    let group = map.get(hit.article_id);
    if (!group) {
      const slug = hit.url.split("/wiki/").pop() ?? "";
      const title = decodeURIComponent(slug).replace(/_/g, " ") || `Article #${hit.article_id}`;
      group = { article_id: hit.article_id, title, url: hit.url, hits: [] };
      map.set(hit.article_id, group);
    }
    group.hits.push(ranked);
  });
  return Array.from(map.values());
}
```

- [ ] **Step 3: Verify compilation**

```bash
cd /home/yichuan/pixelrag/web && npx tsc --noEmit
```

Expected: No errors.

- [ ] **Step 4: Commit**

```bash
cd /home/yichuan/pixelrag
git add web/src/components/ResultGroup.tsx web/src/lib/types.ts
git commit -m "feat(web): add ResultGroup component with article grouping"
```

---

### Task 7: SearchBar Component (Text Only)

**Files:**
- Create: `web/src/components/SearchBar.tsx`

- [ ] **Step 1: Build the SearchBar component**

Create `web/src/components/SearchBar.tsx`:

```tsx
"use client";

import { useState, useRef, useCallback } from "react";
import { Search, X, ImagePlus } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

interface SearchBarProps {
  onSearch: (query: string, image?: string) => void;
  isLoading: boolean;
}

export function SearchBar({ onSearch, isLoading }: SearchBarProps) {
  const [query, setQuery] = useState("");
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [imageBase64, setImageBase64] = useState<string | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleSubmit = useCallback(() => {
    if (!query.trim() && !imageBase64) return;
    onSearch(query.trim(), imageBase64 ?? undefined);
  }, [query, imageBase64, onSearch]);

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent) => {
      if (e.key === "Enter") handleSubmit();
    },
    [handleSubmit]
  );

  const handleImageUpload = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) return;
    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target?.result as string;
      setImagePreview(dataUrl);
      setImageBase64(dataUrl.split(",")[1]);
    };
    reader.readAsDataURL(file);
  }, []);

  const handleDrop = useCallback(
    (e: React.DragEvent) => {
      e.preventDefault();
      const file = e.dataTransfer.files[0];
      if (file) handleImageUpload(file);
    },
    [handleImageUpload]
  );

  const clearImage = useCallback(() => {
    setImagePreview(null);
    setImageBase64(null);
    if (fileInputRef.current) fileInputRef.current.value = "";
  }, []);

  return (
    <div className="w-full max-w-2xl mx-auto">
      <div
        className="flex gap-2 items-center"
        onDrop={handleDrop}
        onDragOver={(e) => e.preventDefault()}
      >
        {/* Image preview thumbnail */}
        {imagePreview && (
          <div className="relative w-10 h-10 rounded-md overflow-hidden flex-shrink-0 border border-border">
            <img
              src={imagePreview}
              alt="Query image"
              className="w-full h-full object-cover"
            />
            <button
              onClick={clearImage}
              className="absolute -top-1 -right-1 w-4 h-4 bg-background border border-border rounded-full flex items-center justify-center"
            >
              <X className="w-2.5 h-2.5" />
            </button>
          </div>
        )}

        {/* Text input */}
        <div className="flex-1 relative">
          <Input
            value={query}
            onChange={(e) => setQuery(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Search Wikipedia visually..."
            className="bg-surface border-border text-foreground placeholder:text-muted-foreground h-11 pr-10"
          />
          <button
            onClick={() => fileInputRef.current?.click()}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-muted-foreground hover:text-foreground transition-colors"
            title="Upload image"
          >
            <ImagePlus className="w-4 h-4" />
          </button>
        </div>

        <input
          ref={fileInputRef}
          type="file"
          accept="image/*"
          className="hidden"
          onChange={(e) => {
            const file = e.target.files?.[0];
            if (file) handleImageUpload(file);
          }}
        />

        {/* Search button */}
        <Button
          onClick={handleSubmit}
          disabled={isLoading || (!query.trim() && !imageBase64)}
          className="h-11 px-5 bg-accent hover:bg-accent/90 text-white"
        >
          {isLoading ? (
            <span className="animate-pulse">Searching...</span>
          ) : (
            <>
              <Search className="w-4 h-4 mr-1.5" />
              Search
            </>
          )}
        </Button>
      </div>

      {/* Mode chips */}
      <div className="flex gap-2 mt-3 justify-center">
        {["Text query", "Image upload", "Drag & drop"].map((label) => (
          <span
            key={label}
            className="text-[11px] text-muted-foreground bg-surface px-2.5 py-1 rounded-full"
          >
            {label}
          </span>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 2: Verify compilation**

```bash
cd /home/yichuan/pixelrag/web && npx tsc --noEmit
```

- [ ] **Step 3: Commit**

```bash
cd /home/yichuan/pixelrag
git add web/src/components/SearchBar.tsx
git commit -m "feat(web): add SearchBar component with text and image input"
```

---

### Task 8: Search Page

**Files:**
- Modify: `web/src/app/page.tsx`

- [ ] **Step 1: Build the search home page**

Replace `web/src/app/page.tsx`:

```tsx
"use client";

import { useState, useCallback } from "react";
import { SearchBar } from "@/components/SearchBar";
import { ResultGroup } from "@/components/ResultGroup";
import { Lightbox } from "@/components/Lightbox";
import { search } from "@/lib/api";
import type { Hit, ArticleGroup } from "@/lib/types";
import { groupHitsByArticle } from "@/lib/types";

export default function SearchPage() {
  const [groups, setGroups] = useState<ArticleGroup[]>([]);
  const [allHits, setAllHits] = useState<Hit[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [resultMeta, setResultMeta] = useState<{
    count: number;
    timeMs: number;
  } | null>(null);
  const [selectedHits, setSelectedHits] = useState<Set<number>>(new Set());
  const [lightboxHit, setLightboxHit] = useState<Hit | null>(null);
  const [hasSearched, setHasSearched] = useState(false);

  const handleSearch = useCallback(
    async (query: string, image?: string) => {
      setIsLoading(true);
      setError(null);
      setSelectedHits(new Set());
      const t0 = performance.now();

      try {
        const queryObj: { text?: string; image?: string } = {};
        if (query) queryObj.text = query;
        if (image) queryObj.image = image;

        const res = await search({
          queries: [queryObj],
          n_docs: 20,
        });
        const elapsed = performance.now() - t0;
        const hits = res.results[0]?.hits ?? [];
        setAllHits(hits);
        setGroups(groupHitsByArticle(hits));
        setResultMeta({ count: hits.length, timeMs: elapsed });
        setHasSearched(true);
      } catch (err) {
        setError(err instanceof Error ? err.message : "Search failed");
        setGroups([]);
        setAllHits([]);
      } finally {
        setIsLoading(false);
      }
    },
    []
  );

  const handleSelectHit = useCallback((hit: Hit) => {
    setSelectedHits((prev) => {
      const next = new Set(prev);
      if (next.has(hit.vector_id)) {
        next.delete(hit.vector_id);
      } else {
        next.add(hit.vector_id);
      }
      return next;
    });
  }, []);

  const handleClickHit = useCallback((hit: Hit) => {
    setLightboxHit(hit);
  }, []);

  return (
    <div className="max-w-6xl mx-auto px-6 py-12">
      {/* Hero */}
      <div className="text-center mb-8">
        <h1 className="font-display text-4xl font-semibold tracking-tight mb-1">
          Vis<span className="text-accent">RAG</span>
        </h1>
        <p className="text-sm text-muted">
          Visual retrieval over 15.7M Wikipedia screenshot tiles
        </p>
      </div>

      {/* Search */}
      <SearchBar onSearch={handleSearch} isLoading={isLoading} />

      {/* Status bar */}
      {resultMeta && (
        <div className="text-xs text-muted mt-6 mb-4 text-center">
          {resultMeta.count} results in{" "}
          <span className="text-accent">
            {(resultMeta.timeMs / 1000).toFixed(2)}s
          </span>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="mt-6 p-4 border border-red-500/30 rounded-lg bg-red-500/5 text-red-400 text-sm">
          {error}
        </div>
      )}

      {/* Results */}
      {groups.length > 0 && (
        <div className="mt-6">
          {groups.map((group) => (
            <ResultGroup
              key={group.article_id}
              group={group}
              selectedHits={selectedHits}
              onSelectHit={handleSelectHit}
              onClickHit={handleClickHit}
            />
          ))}
        </div>
      )}

      {/* Empty state */}
      {hasSearched && groups.length === 0 && !error && !isLoading && (
        <div className="text-center text-muted-foreground mt-12">
          No results found
        </div>
      )}

      {/* Lightbox */}
      {lightboxHit && (
        <Lightbox
          hit={lightboxHit}
          allHits={allHits}
          onClose={() => setLightboxHit(null)}
          onNavigate={setLightboxHit}
        />
      )}

      {/* Compare floating bar */}
      {selectedHits.size >= 2 && (
        <div className="fixed bottom-6 left-1/2 -translate-x-1/2 bg-surface border border-border rounded-full px-5 py-2.5 flex items-center gap-3 shadow-lg z-40">
          <span className="text-sm text-muted">
            {selectedHits.size} tiles selected
          </span>
          <button
            className="text-sm text-accent font-medium hover:underline"
            onClick={() => {
              /* ComparePanel handled in Phase 2 */
            }}
          >
            Compare
          </button>
          <button
            className="text-sm text-muted-foreground hover:text-foreground"
            onClick={() => setSelectedHits(new Set())}
          >
            Clear
          </button>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Verify build**

```bash
cd /home/yichuan/pixelrag/web && npm run build
```

Expected: Build succeeds. (Lightbox component not yet created — create a stub first, see Task 9.)

Note: Before building, create a minimal Lightbox stub so the import doesn't fail:

```bash
mkdir -p /home/yichuan/pixelrag/web/src/components
```

Create `web/src/components/Lightbox.tsx` with a minimal stub:

```tsx
"use client";

import type { Hit } from "@/lib/types";

interface LightboxProps {
  hit: Hit;
  allHits: Hit[];
  onClose: () => void;
  onNavigate: (hit: Hit) => void;
}

export function Lightbox({ onClose }: LightboxProps) {
  return (
    <div
      className="fixed inset-0 z-50 bg-black/90 flex items-center justify-center"
      onClick={onClose}
    >
      <p className="text-white">Lightbox placeholder</p>
    </div>
  );
}
```

- [ ] **Step 3: Build and verify**

```bash
cd /home/yichuan/pixelrag/web && npm run build
```

Expected: Build succeeds.

- [ ] **Step 4: Commit**

```bash
cd /home/yichuan/pixelrag
git add web/src/app/page.tsx web/src/components/Lightbox.tsx
git commit -m "feat(web): add search home page with result grouping"
```

---

### Task 9: Tile Lightbox

**Files:**
- Modify: `web/src/components/Lightbox.tsx` (replace stub)

- [ ] **Step 1: Implement the full Lightbox component**

Replace `web/src/components/Lightbox.tsx`:

```tsx
"use client";

import { useEffect, useCallback, useState, useRef } from "react";
import { motion, AnimatePresence } from "framer-motion";
import { X, ChevronLeft, ChevronRight, ExternalLink } from "lucide-react";
import type { Hit } from "@/lib/types";
import { tileUrl } from "@/lib/api";

interface LightboxProps {
  hit: Hit;
  allHits: Hit[];
  onClose: () => void;
  onNavigate: (hit: Hit) => void;
}

export function Lightbox({ hit, allHits, onClose, onNavigate }: LightboxProps) {
  const [scale, setScale] = useState(1);
  const [position, setPosition] = useState({ x: 0, y: 0 });
  const [dragging, setDragging] = useState(false);
  const dragStart = useRef({ x: 0, y: 0 });
  const posStart = useRef({ x: 0, y: 0 });

  const currentIndex = allHits.findIndex(
    (h) => h.vector_id === hit.vector_id
  );
  const hasPrev = currentIndex > 0;
  const hasNext = currentIndex < allHits.length - 1;

  const resetView = useCallback(() => {
    setScale(1);
    setPosition({ x: 0, y: 0 });
  }, []);

  const goPrev = useCallback(() => {
    if (hasPrev) {
      resetView();
      onNavigate(allHits[currentIndex - 1]);
    }
  }, [hasPrev, currentIndex, allHits, onNavigate, resetView]);

  const goNext = useCallback(() => {
    if (hasNext) {
      resetView();
      onNavigate(allHits[currentIndex + 1]);
    }
  }, [hasNext, currentIndex, allHits, onNavigate, resetView]);

  useEffect(() => {
    const handleKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") onClose();
      if (e.key === "ArrowLeft") goPrev();
      if (e.key === "ArrowRight") goNext();
    };
    window.addEventListener("keydown", handleKey);
    return () => window.removeEventListener("keydown", handleKey);
  }, [onClose, goPrev, goNext]);

  const handleWheel = useCallback((e: React.WheelEvent) => {
    e.preventDefault();
    setScale((prev) => Math.max(0.5, Math.min(5, prev - e.deltaY * 0.002)));
  }, []);

  const handleMouseDown = useCallback(
    (e: React.MouseEvent) => {
      if (scale <= 1) return;
      setDragging(true);
      dragStart.current = { x: e.clientX, y: e.clientY };
      posStart.current = { ...position };
    },
    [scale, position]
  );

  const handleMouseMove = useCallback(
    (e: React.MouseEvent) => {
      if (!dragging) return;
      setPosition({
        x: posStart.current.x + (e.clientX - dragStart.current.x),
        y: posStart.current.y + (e.clientY - dragStart.current.y),
      });
    },
    [dragging]
  );

  const handleMouseUp = useCallback(() => {
    setDragging(false);
  }, []);

  const slug = hit.url.split("/wiki/").pop() ?? "";
  const title =
    decodeURIComponent(slug).replace(/_/g, " ") ||
    `Article #${hit.article_id}`;

  return (
    <AnimatePresence>
      <motion.div
        initial={{ opacity: 0 }}
        animate={{ opacity: 1 }}
        exit={{ opacity: 0 }}
        className="fixed inset-0 z-50 bg-black/95 flex"
        onClick={(e) => {
          if (e.target === e.currentTarget) onClose();
        }}
      >
        {/* Image area */}
        <div
          className="flex-1 flex items-center justify-center overflow-hidden cursor-grab active:cursor-grabbing"
          onWheel={handleWheel}
          onMouseDown={handleMouseDown}
          onMouseMove={handleMouseMove}
          onMouseUp={handleMouseUp}
          onMouseLeave={handleMouseUp}
        >
          <img
            src={tileUrl(hit.path)}
            alt={`tile ${hit.tile_index}:${hit.chunk_index}`}
            className="max-w-full max-h-full object-contain select-none"
            style={{
              transform: `translate(${position.x}px, ${position.y}px) scale(${scale})`,
              transition: dragging ? "none" : "transform 0.15s ease-out",
            }}
            draggable={false}
          />
        </div>

        {/* Metadata sidebar */}
        <motion.div
          initial={{ x: 80, opacity: 0 }}
          animate={{ x: 0, opacity: 1 }}
          className="w-72 bg-surface border-l border-border p-5 flex flex-col gap-4 overflow-y-auto"
        >
          <h3 className="text-base font-medium">{title}</h3>
          {hit.url && (
            <a
              href={hit.url}
              target="_blank"
              rel="noopener noreferrer"
              className="text-xs text-accent hover:underline flex items-center gap-1"
            >
              Open article <ExternalLink className="w-3 h-3" />
            </a>
          )}

          <div className="space-y-3 text-xs">
            <div>
              <div className="text-muted-foreground mb-0.5">Score</div>
              <div className="text-accent font-semibold text-lg">
                {hit.score.toFixed(4)}
              </div>
            </div>
            <div>
              <div className="text-muted-foreground mb-0.5">Rank</div>
              <div>#{currentIndex + 1} of {allHits.length}</div>
            </div>
            <div>
              <div className="text-muted-foreground mb-0.5">Position</div>
              <div>
                tile {hit.tile_index} : chunk {hit.chunk_index}
              </div>
            </div>
            <div>
              <div className="text-muted-foreground mb-0.5">Tile Height</div>
              <div>{hit.tile_height}px</div>
            </div>
            <div>
              <div className="text-muted-foreground mb-0.5">Y Offset</div>
              <div>{hit.y_offset}px</div>
            </div>
            <div>
              <div className="text-muted-foreground mb-0.5">Vector ID</div>
              <div className="font-mono">{hit.vector_id}</div>
            </div>
          </div>

          <div className="mt-auto text-[10px] text-muted-foreground">
            Scroll to zoom · Drag to pan · Arrow keys to navigate
          </div>
        </motion.div>

        {/* Close button */}
        <button
          onClick={onClose}
          className="absolute top-4 right-4 text-muted hover:text-foreground transition-colors z-10"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Navigation arrows */}
        {hasPrev && (
          <button
            onClick={goPrev}
            className="absolute left-4 top-1/2 -translate-y-1/2 text-muted hover:text-foreground transition-colors"
          >
            <ChevronLeft className="w-8 h-8" />
          </button>
        )}
        {hasNext && (
          <button
            onClick={goNext}
            className="absolute right-80 top-1/2 -translate-y-1/2 text-muted hover:text-foreground transition-colors"
          >
            <ChevronRight className="w-8 h-8" />
          </button>
        )}
      </motion.div>
    </AnimatePresence>
  );
}
```

- [ ] **Step 2: Verify build**

```bash
cd /home/yichuan/pixelrag/web && npm run build
```

Expected: Build succeeds.

- [ ] **Step 3: Commit**

```bash
cd /home/yichuan/pixelrag
git add web/src/components/Lightbox.tsx
git commit -m "feat(web): add Lightbox component with pan/zoom/navigation"
```

---

## Phase 2: Full Features

### Task 10: Advanced Search Controls

**Files:**
- Create: `web/src/components/SearchControls.tsx`
- Modify: `web/src/app/page.tsx`

- [ ] **Step 1: Create SearchControls component**

Create `web/src/components/SearchControls.tsx`:

```tsx
"use client";

import { useState } from "react";
import { ChevronDown, ChevronUp } from "lucide-react";
import { Input } from "@/components/ui/input";

export interface SearchOptions {
  n_docs: number;
  nprobe?: number;
  min_tile_height?: number;
  instruction?: string;
}

interface SearchControlsProps {
  options: SearchOptions;
  onChange: (options: SearchOptions) => void;
}

export function SearchControls({ options, onChange }: SearchControlsProps) {
  const [open, setOpen] = useState(false);

  return (
    <div className="w-full max-w-2xl mx-auto mt-3">
      <button
        onClick={() => setOpen(!open)}
        className="flex items-center gap-1 text-[11px] text-muted-foreground hover:text-muted mx-auto"
      >
        Advanced
        {open ? (
          <ChevronUp className="w-3 h-3" />
        ) : (
          <ChevronDown className="w-3 h-3" />
        )}
      </button>

      {open && (
        <div className="mt-3 p-4 bg-surface border border-border rounded-lg grid grid-cols-2 gap-3">
          <div>
            <label className="text-[10px] text-muted-foreground uppercase tracking-wider">
              Results
            </label>
            <Input
              type="number"
              value={options.n_docs}
              onChange={(e) =>
                onChange({ ...options, n_docs: parseInt(e.target.value) || 10 })
              }
              min={1}
              max={100}
              className="mt-1 h-8 text-xs bg-background border-border"
            />
          </div>
          <div>
            <label className="text-[10px] text-muted-foreground uppercase tracking-wider">
              nprobe
            </label>
            <Input
              type="number"
              value={options.nprobe ?? ""}
              onChange={(e) =>
                onChange({
                  ...options,
                  nprobe: e.target.value ? parseInt(e.target.value) : undefined,
                })
              }
              placeholder="default"
              className="mt-1 h-8 text-xs bg-background border-border"
            />
          </div>
          <div>
            <label className="text-[10px] text-muted-foreground uppercase tracking-wider">
              Min tile height
            </label>
            <Input
              type="number"
              value={options.min_tile_height ?? ""}
              onChange={(e) =>
                onChange({
                  ...options,
                  min_tile_height: e.target.value
                    ? parseInt(e.target.value)
                    : undefined,
                })
              }
              placeholder="none"
              className="mt-1 h-8 text-xs bg-background border-border"
            />
          </div>
          <div>
            <label className="text-[10px] text-muted-foreground uppercase tracking-wider">
              Instruction
            </label>
            <Input
              value={options.instruction ?? ""}
              onChange={(e) =>
                onChange({
                  ...options,
                  instruction: e.target.value || undefined,
                })
              }
              placeholder="default"
              className="mt-1 h-8 text-xs bg-background border-border"
            />
          </div>
        </div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Wire SearchControls into the search page**

In `web/src/app/page.tsx`, add state and pass options to the search call:

1. Add import: `import { SearchControls, type SearchOptions } from "@/components/SearchControls";`
2. Add state: `const [searchOptions, setSearchOptions] = useState<SearchOptions>({ n_docs: 20 });`
3. In `handleSearch`, change the `search()` call to use `searchOptions`:
   ```ts
   const res = await search({
     queries: [queryObj],
     n_docs: searchOptions.n_docs,
     nprobe: searchOptions.nprobe,
     min_tile_height: searchOptions.min_tile_height,
     instruction: searchOptions.instruction,
   });
   ```
4. Add `<SearchControls options={searchOptions} onChange={setSearchOptions} />` right after `<SearchBar />`.

- [ ] **Step 3: Verify build**

```bash
cd /home/yichuan/pixelrag/web && npm run build
```

- [ ] **Step 4: Commit**

```bash
cd /home/yichuan/pixelrag
git add web/src/components/SearchControls.tsx web/src/app/page.tsx
git commit -m "feat(web): add advanced search controls (nprobe, min_tile_height, instruction)"
```

---

### Task 11: Side-by-Side Compare Panel

**Files:**
- Create: `web/src/components/ComparePanel.tsx`
- Modify: `web/src/app/page.tsx`

- [ ] **Step 1: Create ComparePanel component**

Create `web/src/components/ComparePanel.tsx`:

```tsx
"use client";

import { motion } from "framer-motion";
import { X } from "lucide-react";
import type { Hit } from "@/lib/types";
import { tileUrl } from "@/lib/api";

interface ComparePanelProps {
  hits: Hit[];
  allHits: Hit[];
  onClose: () => void;
}

export function ComparePanel({ hits, allHits, onClose }: ComparePanelProps) {
  return (
    <motion.div
      initial={{ y: "100%" }}
      animate={{ y: 0 }}
      exit={{ y: "100%" }}
      transition={{ type: "spring", damping: 25, stiffness: 300 }}
      className="fixed bottom-0 left-0 right-0 z-40 bg-surface border-t border-border max-h-[60vh] overflow-y-auto"
    >
      <div className="max-w-6xl mx-auto px-6 py-4">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-sm font-medium">
            Comparing {hits.length} tiles
          </h3>
          <button
            onClick={onClose}
            className="text-muted-foreground hover:text-foreground"
          >
            <X className="w-4 h-4" />
          </button>
        </div>

        <div className="flex gap-4 overflow-x-auto pb-4">
          {hits.map((hit) => {
            const rank =
              allHits.findIndex((h) => h.vector_id === hit.vector_id) + 1;
            const slug = hit.url.split("/wiki/").pop() ?? "";
            const title =
              decodeURIComponent(slug).replace(/_/g, " ") ||
              `Article #${hit.article_id}`;

            return (
              <div
                key={hit.vector_id}
                className="flex-shrink-0 w-80 bg-background border border-border rounded-lg overflow-hidden"
              >
                <img
                  src={tileUrl(hit.path)}
                  alt={`tile ${hit.tile_index}:${hit.chunk_index}`}
                  className="w-full h-48 object-cover object-top"
                />
                <div className="p-3 space-y-1">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-accent">
                      {hit.score.toFixed(4)}
                    </span>
                    <span className="text-[10px] text-muted-foreground">
                      Rank #{rank}
                    </span>
                  </div>
                  <div className="text-xs text-muted truncate">{title}</div>
                  <div className="text-[10px] text-muted-foreground">
                    tile {hit.tile_index}:{hit.chunk_index} ·{" "}
                    {hit.tile_height}px
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </motion.div>
  );
}
```

- [ ] **Step 2: Wire ComparePanel into search page**

In `web/src/app/page.tsx`:

1. Add import: `import { ComparePanel } from "@/components/ComparePanel";`
2. Add state: `const [showCompare, setShowCompare] = useState(false);`
3. Replace the `Compare` button onClick with: `onClick={() => setShowCompare(true)}`
4. Add ComparePanel below the floating bar (inside AnimatePresence):

```tsx
{showCompare && selectedHits.size >= 2 && (
  <ComparePanel
    hits={allHits.filter((h) => selectedHits.has(h.vector_id))}
    allHits={allHits}
    onClose={() => setShowCompare(false)}
  />
)}
```

- [ ] **Step 3: Verify build**

```bash
cd /home/yichuan/pixelrag/web && npm run build
```

- [ ] **Step 4: Commit**

```bash
cd /home/yichuan/pixelrag
git add web/src/components/ComparePanel.tsx web/src/app/page.tsx
git commit -m "feat(web): add side-by-side tile comparison panel"
```

---

### Task 12: Index Status Dashboard

**Files:**
- Create: `web/src/components/StatusCard.tsx`
- Create: `web/src/app/status/page.tsx`

- [ ] **Step 1: Create StatusCard component**

Create `web/src/components/StatusCard.tsx`:

```tsx
interface StatusCardProps {
  label: string;
  value: string;
  sub?: string;
}

export function StatusCard({ label, value, sub }: StatusCardProps) {
  return (
    <div className="bg-surface border border-border rounded-lg p-5">
      <div className="text-[10px] text-muted-foreground uppercase tracking-wider mb-1">
        {label}
      </div>
      <div className="text-2xl font-semibold text-foreground">{value}</div>
      {sub && (
        <div className="text-xs text-muted mt-1">{sub}</div>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Create status page**

Create `web/src/app/status/page.tsx`:

```tsx
"use client";

import { useEffect, useState } from "react";
import { getStatus } from "@/lib/api";
import type { StatusResponse } from "@/lib/types";
import { StatusCard } from "@/components/StatusCard";

function formatBytes(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 ** 2) return `${(bytes / 1024).toFixed(1)} KB`;
  if (bytes < 1024 ** 3) return `${(bytes / 1024 ** 2).toFixed(1)} MB`;
  return `${(bytes / 1024 ** 3).toFixed(2)} GB`;
}

function formatVectors(n: number): string {
  if (n >= 1_000_000) return `${(n / 1_000_000).toFixed(1)}M`;
  if (n >= 1_000) return `${(n / 1_000).toFixed(1)}K`;
  return `${n}`;
}

export default function StatusPage() {
  const [status, setStatus] = useState<StatusResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    getStatus()
      .then(setStatus)
      .catch((err) =>
        setError(err instanceof Error ? err.message : "Failed to load status")
      );
  }, []);

  if (error) {
    return (
      <div className="max-w-4xl mx-auto px-6 py-12">
        <h1 className="font-display text-2xl font-semibold mb-6">
          Index Status
        </h1>
        <div className="p-4 border border-red-500/30 rounded-lg bg-red-500/5 text-red-400 text-sm">
          {error}
        </div>
      </div>
    );
  }

  if (!status) {
    return (
      <div className="max-w-4xl mx-auto px-6 py-12">
        <h1 className="font-display text-2xl font-semibold mb-6">
          Index Status
        </h1>
        <div className="grid grid-cols-2 gap-4">
          {[1, 2, 3, 4].map((i) => (
            <div
              key={i}
              className="bg-surface border border-border rounded-lg p-5 animate-pulse h-24"
            />
          ))}
        </div>
      </div>
    );
  }

  return (
    <div className="max-w-4xl mx-auto px-6 py-12">
      <h1 className="font-display text-2xl font-semibold mb-6">
        Index Status
      </h1>

      <div className="grid grid-cols-2 gap-4 mb-8">
        <StatusCard
          label="Total Vectors"
          value={formatVectors(status.total_vectors)}
          sub={`${status.total_vectors.toLocaleString()} exact`}
        />
        <StatusCard
          label="Dimension"
          value={`${status.dimension}`}
        />
        <StatusCard
          label="Model"
          value={status.model.split("/").pop() ?? status.model}
          sub={status.model}
        />
        <StatusCard
          label="Index Size"
          value={formatBytes(status.index_size_bytes)}
          sub={`metadata: ${formatBytes(status.metadata_size_bytes)}`}
        />
      </div>

      <h2 className="text-sm font-medium text-muted mb-3">Configuration</h2>
      <div className="bg-surface border border-border rounded-lg divide-y divide-border text-sm">
        {[
          ["nlist", `${status.nlist}`],
          ["nprobe", `${status.nprobe}`],
          ["Built at", new Date(status.index_built_at).toLocaleString()],
          ["Index dir", status.index_dir],
          ["Tiles dir", status.tiles_dir],
        ].map(([label, value]) => (
          <div key={label} className="flex px-4 py-2.5">
            <span className="w-32 text-muted-foreground flex-shrink-0">
              {label}
            </span>
            <span className="font-mono text-xs">{value}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Verify build**

```bash
cd /home/yichuan/pixelrag/web && npm run build
```

- [ ] **Step 4: Commit**

```bash
cd /home/yichuan/pixelrag
git add web/src/components/StatusCard.tsx web/src/app/status/page.tsx
git commit -m "feat(web): add index status dashboard page"
```

---

### Task 13: API Documentation Page

**Files:**
- Create: `web/src/components/ApiPlayground.tsx`
- Create: `web/src/app/docs/page.tsx`

- [ ] **Step 1: Create ApiPlayground component**

Create `web/src/components/ApiPlayground.tsx`:

```tsx
"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";

interface ApiPlaygroundProps {
  method: "GET" | "POST";
  path: string;
  defaultBody?: string;
}

export function ApiPlayground({
  method,
  path,
  defaultBody,
}: ApiPlaygroundProps) {
  const [body, setBody] = useState(defaultBody ?? "");
  const [response, setResponse] = useState<string | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleSend = async () => {
    setIsLoading(true);
    setError(null);
    setResponse(null);

    try {
      const url = `/api${path}`;
      const init: RequestInit =
        method === "POST"
          ? {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body,
            }
          : {};
      const res = await fetch(url, init);
      const text = await res.text();
      try {
        setResponse(JSON.stringify(JSON.parse(text), null, 2));
      } catch {
        setResponse(text);
      }
    } catch (err) {
      setError(err instanceof Error ? err.message : "Request failed");
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="bg-surface border border-border rounded-lg p-4 mt-3">
      <div className="text-[10px] text-muted-foreground uppercase tracking-wider mb-2">
        Try it
      </div>
      {method === "POST" && (
        <textarea
          value={body}
          onChange={(e) => setBody(e.target.value)}
          rows={4}
          className="w-full bg-background border border-border rounded-md p-3 font-mono text-xs text-foreground resize-y mb-2 focus:outline-none focus:border-accent"
          spellCheck={false}
        />
      )}
      <Button
        onClick={handleSend}
        disabled={isLoading}
        size="sm"
        className="bg-accent hover:bg-accent/90 text-white text-xs"
      >
        {isLoading ? "Sending..." : `Send ${method}`}
      </Button>

      {error && (
        <div className="mt-3 text-xs text-red-400 bg-red-500/5 border border-red-500/20 rounded p-2">
          {error}
        </div>
      )}
      {response && (
        <pre className="mt-3 bg-background border border-border rounded-md p-3 font-mono text-[11px] text-muted overflow-x-auto max-h-64 overflow-y-auto">
          {response}
        </pre>
      )}
    </div>
  );
}
```

- [ ] **Step 2: Create the docs page**

Create `web/src/app/docs/page.tsx`:

```tsx
"use client";

import { useState } from "react";
import { ApiPlayground } from "@/components/ApiPlayground";

interface Endpoint {
  method: "GET" | "POST";
  path: string;
  description: string;
  requestBody?: string;
  responseSchema: string;
  curl: string;
}

const endpoints: Endpoint[] = [
  {
    method: "POST",
    path: "/search",
    description:
      "Search the visual index with text queries, images, or pre-computed embeddings.",
    requestBody: JSON.stringify(
      { queries: [{ text: "Nikola Tesla" }], n_docs: 10 },
      null,
      2
    ),
    responseSchema: `{
  "results": [{
    "hits": [{
      "score": 0.847,
      "vector_id": 12345,
      "article_id": 42,
      "tile_index": 0,
      "chunk_index": 0,
      "y_offset": 0,
      "tile_height": 8192,
      "path": "/path/to/tile.png",
      "url": "https://en.wikipedia.org/wiki/..."
    }]
  }]
}`,
    curl: `curl -X POST http://localhost:30001/search \\
  -H "Content-Type: application/json" \\
  -d '{"queries": [{"text": "Nikola Tesla"}], "n_docs": 10}'`,
  },
  {
    method: "GET",
    path: "/status",
    description: "Get index metadata and statistics.",
    responseSchema: `{
  "total_vectors": 15700000,
  "dimension": 2048,
  "nlist": 4096,
  "nprobe": 64,
  "model": "text-embedding-mxbai-embed-large-v1",
  "index_built_at": "2026-05-20T00:00:00Z",
  "index_size_bytes": 13312000000,
  "metadata_size_bytes": 512000000
}`,
    curl: "curl http://localhost:30001/status",
  },
  {
    method: "GET",
    path: "/tile",
    description:
      "Serve a tile image by its local path. Path must be under the tiles directory.",
    responseSchema: "(PNG image binary)",
    curl: `curl "http://localhost:30001/tile?path=/path/to/chunk_0000_00.png" -o tile.png`,
  },
  {
    method: "GET",
    path: "/health",
    description: "Health check endpoint.",
    responseSchema: '{"status": "ok"}',
    curl: "curl http://localhost:30001/health",
  },
  {
    method: "POST",
    path: "/reconstruct",
    description:
      "Reconstruct stored embeddings by vector_id for alignment debugging.",
    requestBody: JSON.stringify({ vector_ids: [0, 1, 2] }, null, 2),
    responseSchema: `{
  "embeddings": [[0.012, -0.034, ...], ...]
}`,
    curl: `curl -X POST http://localhost:30001/reconstruct \\
  -H "Content-Type: application/json" \\
  -d '{"vector_ids": [0, 1, 2]}'`,
  },
];

export default function DocsPage() {
  const [activeEndpoint, setActiveEndpoint] = useState(endpoints[0].path);
  const active = endpoints.find((e) => e.path === activeEndpoint)!;

  return (
    <div className="max-w-6xl mx-auto px-6 py-12">
      <h1 className="font-display text-2xl font-semibold mb-8">
        API Reference
      </h1>

      <div className="flex gap-8">
        {/* Sidebar */}
        <div className="w-48 flex-shrink-0">
          <div className="text-[10px] text-muted-foreground uppercase tracking-wider mb-3">
            Endpoints
          </div>
          <div className="space-y-0.5">
            {endpoints.map((ep) => (
              <button
                key={ep.path}
                onClick={() => setActiveEndpoint(ep.path)}
                className={`w-full text-left px-3 py-1.5 rounded text-xs transition-colors ${
                  activeEndpoint === ep.path
                    ? "bg-accent/10 text-foreground border-l-2 border-accent"
                    : "text-muted hover:text-foreground"
                }`}
              >
                <span
                  className={`text-[10px] font-semibold mr-1.5 ${
                    ep.method === "POST"
                      ? "text-method-post"
                      : "text-method-get"
                  }`}
                >
                  {ep.method}
                </span>
                {ep.path}
              </button>
            ))}
          </div>
        </div>

        {/* Main content */}
        <div className="flex-1 min-w-0">
          <div className="mb-6">
            <div className="flex items-center gap-2 mb-2">
              <span
                className={`text-xs font-bold px-2 py-0.5 rounded ${
                  active.method === "POST"
                    ? "bg-method-post/10 text-method-post"
                    : "bg-method-get/10 text-method-get"
                }`}
              >
                {active.method}
              </span>
              <span className="font-mono text-sm">{active.path}</span>
            </div>
            <p className="text-sm text-muted">{active.description}</p>
          </div>

          {/* Request body */}
          {active.requestBody && (
            <div className="mb-6">
              <div className="text-[10px] text-muted-foreground uppercase tracking-wider mb-2">
                Request Body
              </div>
              <pre className="bg-surface border border-border rounded-lg p-4 font-mono text-xs text-muted overflow-x-auto">
                {active.requestBody}
              </pre>
            </div>
          )}

          {/* Response schema */}
          <div className="mb-6">
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider mb-2">
              Response
            </div>
            <pre className="bg-surface border border-border rounded-lg p-4 font-mono text-xs text-muted overflow-x-auto">
              {active.responseSchema}
            </pre>
          </div>

          {/* curl example */}
          <div className="mb-6">
            <div className="text-[10px] text-muted-foreground uppercase tracking-wider mb-2">
              curl
            </div>
            <pre className="bg-surface border border-border rounded-lg p-4 font-mono text-[11px] text-muted overflow-x-auto">
              {active.curl}
            </pre>
          </div>

          {/* Playground */}
          <ApiPlayground
            method={active.method}
            path={active.path}
            defaultBody={active.requestBody}
          />
        </div>
      </div>
    </div>
  );
}
```

- [ ] **Step 3: Verify build**

```bash
cd /home/yichuan/pixelrag/web && npm run build
```

- [ ] **Step 4: Commit**

```bash
cd /home/yichuan/pixelrag
git add web/src/components/ApiPlayground.tsx web/src/app/docs/page.tsx
git commit -m "feat(web): add API documentation page with live playground"
```

---

## Phase 3: Polish

### Task 14: Loading States and Error Handling

**Files:**
- Modify: `web/src/app/page.tsx`
- Modify: `web/src/components/TileCard.tsx`

- [ ] **Step 1: Add skeleton loading state to search page**

In `web/src/app/page.tsx`, add a skeleton component rendered when `isLoading` is true:

```tsx
{isLoading && (
  <div className="mt-6 space-y-6">
    {[1, 2, 3].map((i) => (
      <div key={i}>
        <div className="h-4 w-48 bg-surface rounded animate-pulse mb-3" />
        <div className="flex gap-2.5">
          {[1, 2, 3].map((j) => (
            <div
              key={j}
              className="min-w-[200px] h-44 bg-surface rounded-lg animate-pulse"
            />
          ))}
        </div>
      </div>
    ))}
  </div>
)}
```

Place this right after the status bar section, and wrap the existing results section in `{!isLoading && groups.length > 0 && (...)}`

- [ ] **Step 2: Add loading shimmer to TileCard image**

In `web/src/components/TileCard.tsx`, add a loading state to the image area. Before the `<img>` tag, add:

```tsx
const [imgLoaded, setImgLoaded] = useState(false);
```

And in the image container:

```tsx
{!imgLoaded && !imgError && (
  <div className="absolute inset-0 bg-surface animate-pulse" />
)}
<img
  // ...existing props...
  onLoad={() => setImgLoaded(true)}
  className={`w-full h-full object-cover object-top transition-opacity ${imgLoaded ? "opacity-100" : "opacity-0"}`}
/>
```

- [ ] **Step 3: Verify build**

```bash
cd /home/yichuan/pixelrag/web && npm run build
```

- [ ] **Step 4: Commit**

```bash
cd /home/yichuan/pixelrag
git add web/src/app/page.tsx web/src/components/TileCard.tsx
git commit -m "feat(web): add loading skeletons and image shimmer"
```

---

### Task 15: Shareable Search URLs

**Files:**
- Modify: `web/src/app/page.tsx`

- [ ] **Step 1: Sync search state with URL query params**

In `web/src/app/page.tsx`:

1. Add imports:
   ```tsx
   import { useSearchParams, useRouter } from "next/navigation";
   ```

2. Read initial query from URL params:
   ```tsx
   const searchParams = useSearchParams();
   const router = useRouter();
   const initialQuery = searchParams.get("q") ?? "";
   ```

3. Pass `initialQuery` to `SearchBar` as a `defaultValue` prop. Update `SearchBar` to accept and use it.

4. After a successful search, update the URL:
   ```tsx
   const params = new URLSearchParams();
   if (query) params.set("q", query);
   if (searchOptions.n_docs !== 20) params.set("n_docs", String(searchOptions.n_docs));
   router.replace(`?${params.toString()}`, { scroll: false });
   ```

5. Trigger a search on mount if `initialQuery` exists:
   ```tsx
   useEffect(() => {
     if (initialQuery) {
       handleSearch(initialQuery);
     }
   }, []); // eslint-disable-line react-hooks/exhaustive-deps
   ```

- [ ] **Step 2: Update SearchBar to accept defaultValue**

In `web/src/components/SearchBar.tsx`, add to the props:

```tsx
interface SearchBarProps {
  onSearch: (query: string, image?: string) => void;
  isLoading: boolean;
  defaultValue?: string;
}
```

And change the initial state:

```tsx
const [query, setQuery] = useState(defaultValue ?? "");
```

- [ ] **Step 3: Wrap page content in Suspense**

Since `useSearchParams()` requires a Suspense boundary in Next.js App Router, wrap the page export:

```tsx
import { Suspense } from "react";

function SearchPageContent() {
  // ... all existing page content
}

export default function SearchPage() {
  return (
    <Suspense>
      <SearchPageContent />
    </Suspense>
  );
}
```

- [ ] **Step 4: Verify build**

```bash
cd /home/yichuan/pixelrag/web && npm run build
```

- [ ] **Step 5: Commit**

```bash
cd /home/yichuan/pixelrag
git add web/src/app/page.tsx web/src/components/SearchBar.tsx
git commit -m "feat(web): add shareable search URLs via query params"
```

---

### Task 16: Keyboard Navigation

**Files:**
- Modify: `web/src/app/page.tsx`

- [ ] **Step 1: Add Cmd+K / Ctrl+K shortcut to focus search bar**

In `web/src/components/SearchBar.tsx`:

1. Add a ref to the input: `const inputRef = useRef<HTMLInputElement>(null);`
2. Expose it via a forwardRef or add a global keyboard listener:

```tsx
useEffect(() => {
  const handleKey = (e: KeyboardEvent) => {
    if ((e.metaKey || e.ctrlKey) && e.key === "k") {
      e.preventDefault();
      inputRef.current?.focus();
    }
  };
  window.addEventListener("keydown", handleKey);
  return () => window.removeEventListener("keydown", handleKey);
}, []);
```

3. Add a hint to the input placeholder or nearby: show `⌘K` badge.

In the search bar, next to the input, add:

```tsx
<kbd className="absolute right-10 top-1/2 -translate-y-1/2 text-[10px] text-muted-foreground bg-background border border-border rounded px-1 py-0.5 pointer-events-none hidden sm:block">
  ⌘K
</kbd>
```

Adjust the image upload button position to not overlap with the kbd hint.

- [ ] **Step 2: Verify build**

```bash
cd /home/yichuan/pixelrag/web && npm run build
```

- [ ] **Step 3: Commit**

```bash
cd /home/yichuan/pixelrag
git add web/src/components/SearchBar.tsx
git commit -m "feat(web): add Cmd+K keyboard shortcut to focus search"
```

---

### Task 17: Final Integration Test

- [ ] **Step 1: Build production bundle**

```bash
cd /home/yichuan/pixelrag/web && npm run build
```

Expected: Build succeeds with no errors.

- [ ] **Step 2: Type check**

```bash
cd /home/yichuan/pixelrag/web && npx tsc --noEmit
```

Expected: No type errors.

- [ ] **Step 3: Start dev server and manually verify**

```bash
cd /home/yichuan/pixelrag/web && npm run dev
```

Open http://localhost:3000 in a browser and verify:
- Nav bar renders with logo and links
- Search input is focused, Cmd+K works
- All three pages load: `/`, `/docs`, `/status`
- Dark theme with indigo accent throughout
- Fonts load correctly (Inter for body, Crimson Pro for headings)

Note: Search results require the FastAPI backend running on port 30001. Without it, the search will show an error state (which is also worth verifying looks correct).

- [ ] **Step 4: Final commit**

```bash
cd /home/yichuan/pixelrag
git add -A web/
git commit -m "feat(web): PixelRAG frontend — complete implementation"
```
