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
