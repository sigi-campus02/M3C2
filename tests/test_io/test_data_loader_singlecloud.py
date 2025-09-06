"""Tests for :mod:`m3c2.importer.data_loader`."""

from __future__ import annotations

import os

import numpy as np
import pytest

from m3c2.config.pipeline_config import PipelineConfig
from m3c2.importer.data_loader import DataLoader


def _cfg(tmp_path):
    return PipelineConfig(
        data_dir=str(tmp_path),
        folder_id="sub",
        filename_comparison="comparison.xyz",
        filename_reference="reference.xyz",
        filename_singlecloud="single.xyz",
        use_subsampled_corepoints=1,
        only_stats=False,
        stats_singleordistance="single",
        sample_size=1,
        project="proj",
    )


def test_load_data_singlecloud(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    arr = np.zeros((1, 3))

    class DummyDS:
        def __init__(self, config):
            self.config = config
            DummyDS.instance = self

        def load_points_singlecloud(self):
            return arr

    monkeypatch.setattr("m3c2.importer.data_loader.DataSource", DummyDS)

    loader = DataLoader()
    result = loader.load_data(cfg, mode="singlecloud")

    assert result is arr
    assert DummyDS.instance.config.folder == os.path.join(cfg.data_dir, cfg.folder_id)
    assert DummyDS.instance.config.filename_singlecloud == cfg.filename_singlecloud


def test_load_data_unknown_mode(tmp_path):
    cfg = _cfg(tmp_path)
    loader = DataLoader()
    with pytest.raises(ValueError):
        loader.load_data(cfg, mode="unknown")