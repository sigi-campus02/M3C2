"""Tests for the M3C2 executor.

These tests cover the :class:`~m3c2.pipeline.m3c2_executor.M3C2Executor`,
ensuring that it properly runs the M3C2 algorithm and writes the expected
output files to disk.
"""

from __future__ import annotations

import logging
from types import SimpleNamespace

import numpy as np

from m3c2.m3c2_core.m3c2_executor import M3C2Executor


def test_run_m3c2_writes_outputs(tmp_path, monkeypatch, caplog):
    """Run the executor and check that distance and uncertainty files exist.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory used for storing output files.
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the real runner with a dummy implementation.
    caplog : pytest.LogCaptureFixture
        Captures log records emitted during execution.

    Returns
    -------
    None
    """

    distances = np.array([1.0, np.nan])
    uncertainties = np.array([0.1, 0.2])

    class DummyRunner:
        def run(self, comparison, reference, corepoints, normal, projection):
            return distances, uncertainties

    monkeypatch.setattr("m3c2.m3c2_core.m3c2_executor.M3C2Runner", DummyRunner)

    cfg = SimpleNamespace(process_python_CC="cfg")
    comparison = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    reference = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    corepoints = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    caplog.set_level(logging.INFO)

    executor = M3C2Executor()
    d, u, _ = executor.run_m3c2(
        cfg,
        comparison,
        reference,
        corepoints,
        normal=0.5,
        projection=0.5,
        output_dir=str(tmp_path),
        run_tag="run",
    )

    assert np.allclose(d[0], 1.0) and np.isnan(d[1])
    assert np.allclose(u, uncertainties)

    dist_file = tmp_path / "cfg_run_m3c2_distances.txt"
    coords_file = tmp_path / "cfg_run_m3c2_distances_coordinates.txt"
    uncert_file = tmp_path / "cfg_run_m3c2_uncertainties.txt"
    assert dist_file.is_file()
    assert coords_file.is_file()
    assert uncert_file.is_file()

    loaded_d = np.loadtxt(dist_file)
    assert loaded_d.shape == (2,)
    assert np.allclose(loaded_d[0], 1.0) and np.isnan(loaded_d[1])

    loaded_coords = np.loadtxt(coords_file, skiprows=1)
    assert loaded_coords.shape == (2, 4)
    assert np.allclose(loaded_coords[:, 3], distances, equal_nan=True)

    loaded_u = np.loadtxt(uncert_file)
    assert np.allclose(loaded_u, uncertainties)

    assert any("Distanzen gespeichert" in rec.message for rec in caplog.records)
    assert any("Unsicherheiten gespeichert" in rec.message for rec in caplog.records)


def test_run_m3c2_skips_coordinates_on_mismatch(tmp_path, monkeypatch, caplog):
    """Warn and skip coordinate file when distances length differs."""

    distances = np.array([1.0])
    uncertainties = np.array([0.1])

    class DummyRunner:
        def run(self, comparison, reference, corepoints, normal, projection):
            return distances, uncertainties

    monkeypatch.setattr("m3c2.m3c2_core.m3c2_executor.M3C2Runner", DummyRunner)

    cfg = SimpleNamespace(process_python_CC="cfg")
    comparison = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    reference = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
    corepoints = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])

    caplog.set_level(logging.WARNING)

    executor = M3C2Executor()
    executor.run_m3c2(
        cfg,
        comparison,
        reference,
        corepoints,
        normal=0.5,
        projection=0.5,
        output_dir=str(tmp_path),
        run_tag="run",
    )

    coords_file = tmp_path / "cfg_run_m3c2_distances_coordinates.txt"
    assert not coords_file.exists()
    assert any(
        rec.levelno == logging.WARNING
        and "Anzahl Koordinaten stimmt nicht mit Distanzen überein" in rec.message
        for rec in caplog.records
    )
