import os
import sys

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from m3c2.visualization.comparison_loader import (
    _load_and_mask,
    _resolve,
)


def test_resolve_prefers_local(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fid = "A"
    filename = "file.txt"
    local = tmp_path / fid
    local.mkdir()
    (local / filename).write_text("0")
    data_dir = tmp_path / "data" / fid
    data_dir.mkdir(parents=True)
    (data_dir / filename).write_text("1")

    assert _resolve(fid, filename) == os.path.join(fid, filename)


def test_load_and_mask(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    fid = "B"
    folder = tmp_path / fid
    folder.mkdir()

    (folder / "python_ref_m3c2_distances.txt").write_text("1\n2\nnan\n")
    (folder / "python_ref_ai_m3c2_distances.txt").write_text("1\nnan\n3\n")

    result = _load_and_mask(fid, ["ref", "ref_ai"])
    assert result is not None
    a, b = result
    assert np.allclose(a, np.array([1.0]))
    assert np.allclose(b, np.array([1.0]))

