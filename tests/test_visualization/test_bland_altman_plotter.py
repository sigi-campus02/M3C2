"""Tests for the Bland–Altman plotter.

These tests verify that the plotter creates output images when data is
available and that it does not produce files for empty or missing inputs.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")

from m3c2.visualization import bland_altman_plotter
from m3c2.visualization.bland_altman_plotter import bland_altman_plot


def test_bland_altman_plot_creates_file(tmp_path, monkeypatch):
    """Ensure a Bland–Altman plot is written when data exists.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory for the output file.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the data loader.
    """

    def fake_load(fid, refs):
        return np.array([1.0, 2.0, 3.0]), np.array([1.1, 2.1, 3.1])

    monkeypatch.setattr(bland_altman_plotter, "_load_and_mask", fake_load)
    bland_altman_plot(["fid"], ["r1", "r2"], outdir=str(tmp_path))
    expected = tmp_path / "bland_altman_fid_r1_vs_r2.png"
    assert expected.is_file()


def test_bland_altman_plot_handles_none(tmp_path, monkeypatch):
    """Skip plotting when the loader yields ``None``.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory for the output file.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the data loader.
    """

    def fake_load(fid, refs):
        return None

    monkeypatch.setattr(bland_altman_plotter, "_load_and_mask", fake_load)
    bland_altman_plot(["fid"], ["r1", "r2"], outdir=str(tmp_path))
    assert not (tmp_path / "bland_altman_fid_r1_vs_r2.png").exists()


def test_bland_altman_plot_handles_empty_arrays(tmp_path, monkeypatch):
    """Skip plotting when the loader returns empty arrays.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory for the output file.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the data loader.
    """

    def fake_load(fid, refs):
        return np.array([]), np.array([])

    monkeypatch.setattr(bland_altman_plotter, "_load_and_mask", fake_load)
    bland_altman_plot(["fid"], ["r1", "r2"], outdir=str(tmp_path))
    assert not (tmp_path / "bland_altman_fid_r1_vs_r2.png").exists()
