"""Streamlit UI for the visual image index — a mymind-style search board.

Launched by `pixelrag visual web --base-dir <dir>`, which runs `streamlit run` on this file.
It shells out to the existing `pixelrag visual` CLI for search and build, and reads each
index's metadata.json directly; it imports no faiss/torch/openai itself.
"""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import streamlit as st

from pixelrag_visual.tags import build_tag_counts, matches_tags, parse_caption

# Per-result stdout line from `pixelrag visual search`: "  1. [0.4213] /abs/path.png"
_RESULT_RE = re.compile(r"^\s+\d+\. \[([0-9.]+)\] (.+)$")
DISPLAY_CAP = 300
GRID_COLS = 4
SEARCH_K = 200  # ponytail: fixed k; raise if tag-filtering a text search starves results


def get_base_dir() -> str:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", required=True)
    args, _ = parser.parse_known_args()
    return os.path.expanduser(args.base_dir)


def list_indexes(base_dir: str) -> list[str]:
    """Names of subfolders under base_dir that hold a built index."""
    base = Path(base_dir)
    if not base.is_dir():
        return []
    return sorted(p.name for p in base.iterdir() if (p / "metadata.json").exists())


@st.cache_data
def load_metadata(index_dir: str, mtime: float) -> tuple[list[str], list[str]]:
    """Read (paths, captions) from metadata.json. Cached on (dir, mtime) so it re-reads on change."""
    with open(Path(index_dir) / "metadata.json") as f:
        meta = json.load(f)
    return meta.get("paths", []), meta.get("captions", [])


@st.cache_data
def run_search(index_dir: str, query: str) -> tuple[list[tuple[str, str]], str]:
    """Run `pixelrag visual search` and parse stdout into [(path, caption)], plus an error string.

    ponytail: subprocess per search honors "just run the CLI"; swap to a warm @st.cache_resource
    load of the index if latency ever bites.
    """
    proc = subprocess.run(
        ["pixelrag", "visual", "search", "--index-dir", index_dir, "-q", query, "--k", str(SEARCH_K)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        return [], (proc.stderr.strip() or "search exited non-zero")

    lines = proc.stdout.splitlines()
    results: list[tuple[str, str]] = []
    for i, line in enumerate(lines):
        m = _RESULT_RE.match(line)
        if m:
            path = m.group(2)
            caption = lines[i + 1].strip() if i + 1 < len(lines) else ""
            results.append((path, caption))
    return results, ""


def card_label(caption: str, path: str) -> str:
    """Short label under a thumbnail: the subject tag, else the filename."""
    subject = parse_caption(caption).get("subject")
    return subject[0] if subject else Path(path).name


def run_build(base_dir: str, folder: str, name: str, recursive: bool, incremental: bool) -> None:
    folder = os.path.expanduser(folder.strip())
    name = name.strip()
    if not name:
        st.error("Index name is required.")
        return
    if not os.path.isdir(folder):
        st.error(f"Folder not found: {folder}")
        return

    output_dir = str(Path(base_dir) / name)
    cmd = ["pixelrag", "visual", "build", "--input-dir", folder, "--output-dir", output_dir]
    if recursive:
        cmd.append("--recursive")
    if incremental:
        cmd.append("--incremental")

    # ponytail: synchronous build, single local user — no job queue.
    with st.status(f"Building '{name}'…", expanded=True) as status:
        st.write(f"`$ {' '.join(cmd)}`")
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1
        )
        for line in proc.stdout:
            line = line.rstrip()
            if line:
                st.write(line)
        proc.wait()
        if proc.returncode == 0:
            status.update(label=f"Built '{name}'", state="complete")
            load_metadata.clear()
            st.rerun()
        else:
            status.update(label=f"Build failed (exit {proc.returncode})", state="error")


def render_sidebar(base_dir: str, indexes: list[str]):
    st.sidebar.title("🧠 Visual")
    index = st.sidebar.selectbox("Index", indexes) if indexes else None
    query = st.sidebar.text_input("Search", placeholder="what do you remember?")

    captions: list[str] = []
    index_dir = None
    if index:
        index_dir = str(Path(base_dir) / index)
        mtime = (Path(index_dir) / "metadata.json").stat().st_mtime
        _, captions = load_metadata(index_dir, mtime)

    tag_options = [t for t, _ in build_tag_counts(captions).most_common(40)]
    selected = st.sidebar.multiselect("Tags", tag_options)

    with st.sidebar.expander("➕ Add images / New index"):
        folder = st.text_input("Image folder", placeholder="/path/to/images")
        name = st.text_input("Index name", value=index or "")
        recursive = st.checkbox("Recurse into subfolders", value=True)
        incremental = st.checkbox("Add to existing index", value=bool(index))
        if st.button("Build", type="primary", width="stretch"):
            run_build(base_dir, folder, name, recursive, incremental)

    return index, index_dir, query.strip(), selected


def main():
    st.set_page_config(page_title="PixelRAG Visual", layout="wide")
    base_dir = get_base_dir()
    indexes = list_indexes(base_dir)

    index, index_dir, query, selected = render_sidebar(base_dir, indexes)

    if not index:
        st.info(
            f"No index selected. Point `--base-dir` at a folder of indexes, or create one with "
            f"**➕ Add images / New index** in the sidebar.\n\nBase dir: `{base_dir}`"
        )
        return

    if query:
        results, error = run_search(index_dir, query)
        if error:
            st.error(f"Search failed — is LM Studio reachable?\n\n```\n{error}\n```")
            return
        items = results
    else:
        paths, captions = load_metadata(index_dir, (Path(index_dir) / "metadata.json").stat().st_mtime)
        items = list(zip(paths, captions))

    items = [(p, c) for p, c in items if matches_tags(c, selected)]
    st.caption(f"{len(items)} image{'s' if len(items) != 1 else ''}")

    shown = items[:DISPLAY_CAP]
    cols = st.columns(GRID_COLS)
    for i, (path, caption) in enumerate(shown):
        col = cols[i % GRID_COLS]
        if os.path.exists(path):
            col.image(path, caption=card_label(caption, path), width="stretch")
        else:
            col.warning(f"missing: {Path(path).name}")

    if len(items) > DISPLAY_CAP:
        st.caption(f"Showing first {DISPLAY_CAP} of {len(items)}.")


main()
