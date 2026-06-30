"""Streamlit UI for the visual image index — pick an index, search, browse results.

Building/captioning/indexing is done from the terminal (`pixelrag visual build ...`).
This UI is search + browse only: it shells out to `pixelrag visual search` and reads the
index's metadata.json directly. Launched by `pixelrag visual web --base-dir <dir>`.
"""

import argparse
import json
import os
import re
import subprocess
from pathlib import Path

import streamlit as st

# Per-result stdout line from `pixelrag visual search`: "  1. [0.4213] /abs/path.png"
_RESULT_RE = re.compile(r"^\s+\d+\. \[([0-9.]+)\] (.+)$")
DISPLAY_CAP = 60   # images shown when the search box is empty
SEARCH_K = 50      # top-K returned for a text query
GRID_COLS = 4

SEARCH_CSS = """
<style>
/* big, google-like search box */
div[data-testid="stTextInput"] input {
    font-size: 1.4rem;
    padding: 0.85rem 1.25rem;
    border-radius: 2rem;
}
</style>
"""


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
    """Read (paths, captions) from metadata.json. Cached on (dir, mtime)."""
    with open(Path(index_dir) / "metadata.json") as f:
        meta = json.load(f)
    return meta.get("paths", []), meta.get("captions", [])


@st.cache_data
def run_search(index_dir: str, query: str) -> tuple[list[tuple[str, str]], str]:
    """Run `pixelrag visual search` and parse stdout into [(path, caption)], plus an error string."""
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


def main():
    st.set_page_config(page_title="PixelRAG Visual", layout="wide")
    base_dir = get_base_dir()
    indexes = list_indexes(base_dir)

    if not indexes:
        st.info(
            f"No indexes found in `{base_dir}`.\n\nBuild one from the terminal:\n\n"
            f"```\npixelrag visual build --input-dir <photos> "
            f"--output-dir {base_dir}/<name> --recursive\n```"
        )
        return

    st.markdown(SEARCH_CSS, unsafe_allow_html=True)

    # Top bar: choose index (left) + big search box (right).
    left, right = st.columns([1, 4])
    index = left.selectbox("Index", indexes, label_visibility="collapsed")
    query = right.text_input(
        "Search", placeholder="🔍  Search your photos…", label_visibility="collapsed"
    ).strip()

    index_dir = str(Path(base_dir) / index)
    meta_path = Path(index_dir) / "metadata.json"

    if query:
        items, error = run_search(index_dir, query)
        if error:
            st.error(f"Search failed — is LM Studio reachable?\n\n```\n{error}\n```")
            return
    else:
        paths, captions = load_metadata(index_dir, meta_path.stat().st_mtime)
        items = list(zip(paths, captions))[:DISPLAY_CAP]

    st.caption(f"{len(items)} image{'s' if len(items) != 1 else ''}")

    cols = st.columns(GRID_COLS)
    for i, (path, caption) in enumerate(items):
        with cols[i % GRID_COLS]:
            if os.path.exists(path):
                st.image(path, width="stretch")
            else:
                st.warning(f"missing: {Path(path).name}")
            # st.code gives a built-in copy button (top-right on hover).
            st.code(caption or "(no caption)", language=None, wrap_lines=True)


main()
