from __future__ import annotations

import os
import numpy as np
from m3c2.config.pipeline_config import PipelineConfig
from m3c2.pipeline.data_loader import DataLoader


class DummyDS:
    def __init__(self, config):
        self.config = config

    def load_points(self):
        arr = np.zeros((1, 3))
        return arr, arr, arr


def test_load_data_uses_data_dir(tmp_path, monkeypatch):
    cfg = PipelineConfig(
        data_dir=str(tmp_path),
        folder_id="sub",
        filename_mov="mov.xyz",
        filename_ref="ref.xyz",
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
    ds, mov, ref, corepoints = loader._load_data(cfg)

    assert ds.config.folder == os.path.join(cfg.data_dir, cfg.folder_id)
    assert mov.shape == (1, 3)
    assert ref.shape == (1, 3)
    assert corepoints.shape == (1, 3)
