"""CLI for visual image search: build and query a FAISS index."""

import argparse
import json
import os
from pathlib import Path

from openai import OpenAI

from pixelrag_visual.scan import scan_images
from pixelrag_visual.caption import generate_captions
from pixelrag_visual.embed_client import get_lm_client, embed_batch
from pixelrag_visual.index_builder import build_index, load_index, merge_index, search_index


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


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
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
    p_build.add_argument(
        "--incremental", action="store_true", help="Merge new images into existing index instead of replacing"
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

    return parser.parse_args(argv)


def main():
    args = parse_args()
    if args.command == "build":
        cmd_build(args)
    elif args.command == "search":
        cmd_search(args)


if __name__ == "__main__":
    main()
