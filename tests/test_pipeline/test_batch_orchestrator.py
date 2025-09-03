"""Tests for the batch orchestrator's data loading mechanisms."""

from __future__ import annotations

import os
import numpy as np
from m3c2.config.pipeline_config import PipelineConfig
from m3c2.importer.data_loader import DataLoader


class DummyDS:
    """Minimal data source used to validate configuration paths in tests."""

    def __init__(self, config):
        self.config = config

    def load_points(self):
        arr = np.zeros((1, 3))
        return arr, arr, arr


def test_load_data_uses_data_dir(tmp_path, monkeypatch):
    """Verify that the loader honors the configured data directory.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory representing the pipeline's data directory.
    monkeypatch : pytest.MonkeyPatch
        Fixture for replacing the ``DataSource`` with ``DummyDS``.
    """
    cfg = PipelineConfig(
        data_dir=str(tmp_path),
        folder_id="sub",
        filename_mov="mov.xyz",
        filename_ref="ref.xyz",
        filename_singlecloud="single.xyz",
        mov_as_corepoints=True,
        use_subsampled_corepoints=1,
        only_stats=False,
        stats_singleordistance="single",
        sample_size=1,
        project="proj",
    )

    monkeypatch.setattr(
        "m3c2.pipeline.data_loader.DataSource", DummyDS
    )
    loader = DataLoader()
    ds, mov, ref, corepoints = loader.load_data(cfg, mode="multicloud")

    assert ds.config.folder == os.path.join(cfg.data_dir, cfg.folder_id)
    assert mov.shape == (1, 3)
    assert ref.shape == (1, 3)
    assert corepoints.shape == (1, 3)
