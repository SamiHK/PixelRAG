import os
import tempfile
from pathlib import Path

from pixelrag_visual.scan import scan_images, ImageInfo


def test_scan_images_finds_png_and_jpg():
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "photo1.png").write_bytes(b"fake")
        Path(tmpdir, "photo2.jpg").write_bytes(b"fake")
        Path(tmpdir, "readme.txt").write_bytes(b"ignore me")

        result = scan_images(tmpdir)
    assert len(result) == 2
    paths = {r.path for r in result}
    assert any("photo1.png" in p for p in paths)
    assert any("photo2.jpg" in p for p in paths)


def test_scan_images_ignores_non_image_extensions():
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "a.png").write_bytes(b"")
        Path(tmpdir, "b.jpg").write_bytes(b"")
        Path(tmpdir, "c.jpeg").write_bytes(b"")
        Path(tmpdir, "d.txt").write_bytes(b"")
        Path(tmpdir, "e.pdf").write_bytes(b"")

        result = scan_images(tmpdir)
    assert len(result) == 3


def test_scan_images_recursive():
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "top.png").write_bytes(b"")
        subdir = Path(tmpdir, "sub")
        subdir.mkdir()
        (subdir / "nested.png").write_bytes(b"")

        result = scan_images(tmpdir, recursive=True)
    assert len(result) == 2


def test_scan_images_non_recursive():
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "top.png").write_bytes(b"")
        subdir = Path(tmpdir, "sub")
        subdir.mkdir()
        (subdir / "nested.png").write_bytes(b"")

        result = scan_images(tmpdir, recursive=False)
    assert len(result) == 1


def test_scan_images_deterministic_order():
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "z.png").write_bytes(b"")
        Path(tmpdir, "a.png").write_bytes(b"")
        Path(tmpdir, "m.png").write_bytes(b"")

        result = scan_images(tmpdir)
    assert [r.filename for r in result] == ["a.png", "m.png", "z.png"]


def test_image_info_is_frozen():
    info = ImageInfo(path="/tmp/x.png", filename="x.png", size_bytes=100)
    try:
        info.path = "/tmp/y.png"
        assert False, "should have raised"
    except AttributeError:
        pass  # expected on frozen dataclass


def test_scan_images_returns_absolute_paths():
    with tempfile.TemporaryDirectory() as tmpdir:
        Path(tmpdir, "x.png").write_bytes(b"")

        result = scan_images(tmpdir)
    assert os.path.isabs(result[0].path)
