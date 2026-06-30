import json
from pathlib import Path

from pixelrag_visual.app import list_indexes


def _write_index(d: Path):
    d.mkdir(parents=True, exist_ok=True)
    (d / "metadata.json").write_text(json.dumps({"paths": [], "captions": [], "dimension": 1}))


def test_base_dir_is_itself_an_index(tmp_path):
    _write_index(tmp_path)
    found = list_indexes(str(tmp_path))
    assert found == {tmp_path.name: str(tmp_path)}


def test_base_dir_parent_of_indexes(tmp_path):
    _write_index(tmp_path / "photos")
    _write_index(tmp_path / "screenshots")
    found = list_indexes(str(tmp_path))
    assert set(found) == {"photos", "screenshots"}
    assert found["photos"] == str(tmp_path / "photos")


def test_mixed_layout(tmp_path):
    _write_index(tmp_path)              # base is an index
    _write_index(tmp_path / "extra")    # and also holds one
    found = list_indexes(str(tmp_path))
    assert tmp_path.name in found
    assert "extra" in found


def test_empty_and_missing(tmp_path):
    assert list_indexes(str(tmp_path)) == {}            # dir exists, no metadata
    assert list_indexes(str(tmp_path / "nope")) == {}   # dir doesn't exist
