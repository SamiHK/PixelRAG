"""The `pixelrag` umbrella CLI.

`pixelrag <stage> [args...]` dispatches to a pipeline stage's own CLI. Stages live in
independently installable packages; this umbrella lazily imports the one you ask for and
prints a clear install hint if it is missing.

Stage 0 (capture a page → screenshot tiles) is the standalone `pixelshot` command, not a
subcommand here — it is the one primitive you run by hand, and it stays light (no torch).
"""

import importlib
import sys

# stage -> (module, function, workspace package, pip extra)
STAGES = {
    "chunk": ("pixelrag_embed.chunk", "main", "pixelrag-embed", "embed"),
    "embed": ("pixelrag_embed.embed", "main", "pixelrag-embed", "embed"),
    "build-index": ("pixelrag_embed.index", "main", "pixelrag-embed", "embed"),
    "index": ("pixelrag_index.pipelines", "main", "pixelrag-index", "index"),
    "monitor": ("pixelrag_index.monitor", "main", "pixelrag-index", "index"),
    "serve": ("pixelrag_serve.api", "main", "pixelrag-serve", "serve"),
    "visual": ("pixelrag_visual.cli", "main", "pixelrag-visual", "visual"),
}


def _usage() -> str:
    rows = "\n".join(f"  {name:<13} {mod[0]}" for name, mod in STAGES.items())
    return (
        "usage: pixelrag <stage> [args...]\n\n"
        "Pipeline stages:\n"
        f"{rows}\n\n"
        "Capture a page to screenshot tiles with the standalone `pixelshot` command.\n"
        "Run `pixelrag <stage> --help` for a stage's own options."
    )


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print(_usage())
        sys.exit(0 if len(sys.argv) >= 2 else 2)

    stage = sys.argv[1]
    if stage not in STAGES:
        print(f"pixelrag: unknown stage '{stage}'\n\n{_usage()}", file=sys.stderr)
        sys.exit(2)

    module, func, package, extra = STAGES[stage]
    try:
        mod = importlib.import_module(module)
    except ModuleNotFoundError:
        print(
            f"pixelrag: stage '{stage}' is not installed.\n"
            f"  → uv sync --package {package}   (dev)\n"
            f"  → pip install 'pixelrag[{extra}]'   (published)",
            file=sys.stderr,
        )
        sys.exit(1)

    # Hand argv to the stage's own argparse; prog reads as `pixelrag <stage>`.
    sys.argv = [f"pixelrag {stage}", *sys.argv[2:]]
    getattr(mod, func)()


if __name__ == "__main__":
    main()
