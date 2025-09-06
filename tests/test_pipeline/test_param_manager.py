"""Tests for the M3C2 parameter manager."""

from __future__ import annotations

import numpy as np
import pytest

from m3c2.config.pipeline_config import PipelineConfig
from m3c2.pipeline.param_manager import ParamManager


def _minimal_cfg(**overrides) -> PipelineConfig:
    """Create a minimal :class:`PipelineConfig` for tests."""
    defaults = dict(
        data_dir="",
        folder_id="",
        filename_comparison="",
        filename_reference="",
        filename_singlecloud="",
        project="proj",
        use_subsampled_corepoints=0,
        only_stats=False,
        stats_singleordistance="single",
        sample_size=0,
    )
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def test_save_params_creates_file(tmp_path):
    """Saving parameters should create the parameter file."""
    cfg = _minimal_cfg()
    manager = ParamManager()

    manager.save_params(cfg, 1.0, 2.0, str(tmp_path), "tag")

    params_file = tmp_path / "python_tag_m3c2_params.txt"
    assert params_file.exists()
    assert params_file.read_text() == "NormalScale=1.0\nSearchScale=2.0\n"


def test_save_params_fails_on_unwritable_dir(tmp_path, monkeypatch):
    """Attempting to save into an unwritable directory should raise."""
    cfg = _minimal_cfg()
    manager = ParamManager()

    def raise_oserror(*args, **kwargs):  # pragma: no cover - simple helper
        raise OSError("cannot write")

    monkeypatch.setattr("builtins.open", raise_oserror)

    with pytest.raises(OSError):
        manager.save_params(cfg, 1.0, 2.0, str(tmp_path), "tag")


def test_handle_existing_params_with_overrides(tmp_path):
    """Overrides take precedence over existing parameter files."""
    cfg = _minimal_cfg(normal_override=3.0, proj_override=4.0)
    manager = ParamManager()

    # Even if a params file exists, overrides should be returned
    manager.save_params(_minimal_cfg(), 1.0, 2.0, str(tmp_path), "tag")
    normal, proj = manager.handle_existing_params(cfg, str(tmp_path), "tag")
    assert normal == 3.0
    assert proj == 4.0


def test_handle_existing_params_loads_file(tmp_path):
    """Existing parameter files should be loaded when no overrides are set."""
    cfg = _minimal_cfg()
    manager = ParamManager()

    manager.save_params(cfg, 1.5, 2.5, str(tmp_path), "tag")
    normal, proj = manager.handle_existing_params(cfg, str(tmp_path), "tag")

    assert normal == 1.5
    assert proj == 2.5


def test_handle_existing_params_no_file(tmp_path):
    """Missing parameter files should yield NaN values."""
    cfg = _minimal_cfg()
    manager = ParamManager()

    normal, proj = manager.handle_existing_params(cfg, str(tmp_path), "tag")
    assert np.isnan(normal)
    assert np.isnan(proj)


def test_handle_override_params_returns_overrides():
    """handle_override_params returns override values when both are set."""
    cfg = _minimal_cfg(normal_override=1.0, proj_override=2.0)
    manager = ParamManager()

    normal, proj = manager.handle_override_params(cfg)
    assert normal == 1.0
    assert proj == 2.0


@pytest.mark.parametrize("normal,proj", [(1.0, None), (None, 2.0), (None, None)])
def test_handle_override_params_missing_values(normal, proj):
    """Missing override values should yield NaNs for both scales."""
    cfg = _minimal_cfg(normal_override=normal, proj_override=proj)
    manager = ParamManager()

    n, p = manager.handle_override_params(cfg)
    assert np.isnan(n)
    assert np.isnan(p)
