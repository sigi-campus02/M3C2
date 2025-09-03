"""Tests for converting text distance files to PLY format.

These tests exercise the :mod:`m3c2.visualization.visualization_service` module
by verifying behavior of :func:`VisualizationService.txt_to_ply_with_distance_color`.
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import matplotlib

matplotlib.use("Agg")

import m3c2.visualization.services.visualization_service as vs
from m3c2.visualization.services.visualization_service import VisualizationService
import pytest


def test_txt_to_ply_calls_writer(tmp_path, monkeypatch):
    """Verify that a PLY writer is invoked during conversion.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest for file operations.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace functions and objects during the test.
    """

    txt = tmp_path / "dist.txt"
    txt.write_text("x y z distance\n0 0 0 1.0\n1 1 1 2.0\n")
    outply = tmp_path / "out.ply"

    # ensure dependency check passes
    monkeypatch.setattr(vs, "PlyData", object())
    monkeypatch.setattr(vs, "PlyElement", object())

    called = {}

    def fake_writer(*, points, colors, outply, scalar=None, scalar_name="distance", binary=True):
        called["args"] = (points, colors, outply, scalar, scalar_name, binary)

    monkeypatch.setattr(vs, "_write_ply_xyzrgb", fake_writer)

    VisualizationService.txt_to_ply_with_distance_color(str(txt), str(outply))

    assert "args" in called
    pts, cols, outarg, scalar, name, binary_flag = called["args"]
    assert outarg == str(outply)
    assert pts.shape == (2, 3)
    assert cols.shape == (2, 3)
    assert scalar.shape == (2,)
    assert name == "distance"
    assert binary_flag is True


def test_txt_to_ply_missing_dependency(tmp_path, monkeypatch):
    """Raise an error when optional dependencies are absent.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory used for generating input and output files.
    monkeypatch : pytest.MonkeyPatch
        Fixture to patch missing dependencies in the visualization service.
    """

    txt = tmp_path / "dist.txt"
    txt.write_text("x y z distance\n0 0 0 1.0\n")
    outply = tmp_path / "out.ply"

    monkeypatch.setattr(vs, "PlyData", None)
    monkeypatch.setattr(vs, "PlyElement", None)

    with pytest.raises(RuntimeError):
        VisualizationService.txt_to_ply_with_distance_color(str(txt), str(outply))
