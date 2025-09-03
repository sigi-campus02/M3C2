"""Tests for the ``comparedist_plots`` command-line interface."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import m3c2.cli.comparedist_plots as comparedist_plots

def test_main_creates_output(tmp_path, monkeypatch):
    """Verify that invoking the CLI generates the expected output files.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest for file output.
    monkeypatch : _pytest.monkeypatch.MonkeyPatch
        Fixture used to replace the plotting service with a fake implementation.
    """

    def fake_overlay(cls, cfg, opts):
        Path(cfg.path).mkdir(parents=True, exist_ok=True)
        Path(cfg.path, "dummy.txt").write_text("ok")

    monkeypatch.setattr(
        comparedist_plots.PlotServiceCompareDistances,
        "overlay_plots",
        classmethod(fake_overlay),
    )

    comparedist_plots.main(
        folder_ids=["id"], ref_variants=["ref"], outdir=str(tmp_path)
    )

    expected = tmp_path / "MARS_output" / "MARS_plots" / "dummy.txt"
    assert expected.exists()

