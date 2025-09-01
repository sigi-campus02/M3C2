import os
import sys
import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import m3c2.visualization.bland_altman_plotter as bland_altman_plotter
import m3c2.visualization.passing_bablok_plotter as passing_bablok_plotter
import m3c2.visualization.linear_regression_plotter as linear_regression_plotter


def _dummy_loader(fid, ref):
    return (
        np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        np.array([1.1, 2.1, 3.1, 4.1, 5.1]),
    )


def test_bland_altman_plotter_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr(bland_altman_plotter, "_load_and_mask", _dummy_loader)
    bland_altman_plotter.plot(["F1"], ["ref", "ref_ai"], outdir=str(tmp_path))
    assert (tmp_path / "bland_altman_F1_ref_vs_ref_ai.png").exists()


def test_passing_bablok_plotter_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr(passing_bablok_plotter, "_load_and_mask", _dummy_loader)
    passing_bablok_plotter.plot(["F1"], ["ref", "ref_ai"], outdir=str(tmp_path))
    assert (tmp_path / "passing_bablok_F1_ref_vs_ref_ai.png").exists()


def test_linear_regression_plotter_creates_file(tmp_path, monkeypatch):
    monkeypatch.setattr(linear_regression_plotter, "_load_and_mask", _dummy_loader)
    linear_regression_plotter.plot(["F1"], ["ref", "ref_ai"], outdir=str(tmp_path))
    assert (tmp_path / "linear_regression_F1_ref_vs_ref_ai.png").exists()

