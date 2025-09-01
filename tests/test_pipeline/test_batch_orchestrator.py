from __future__ import annotations

import os
import numpy as np
import pytest
from m3c2.config.pipeline_config import PipelineConfig
from m3c2.pipeline.batch_orchestrator import BatchOrchestrator
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


def _cfg(tmp_path, folder_id="sub"):
    return PipelineConfig(
        data_dir=str(tmp_path),
        folder_id=folder_id,
        filename_mov="mov.xyz",
        filename_ref="ref.xyz",
        mov_as_corepoints=True,
        use_subsampled_corepoints=1,
        only_stats=True,
        stats_singleordistance="single",
        sample_size=1,
        project="proj",
    )


def test_run_all_reraises_unexpected_error(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    orch = BatchOrchestrator([cfg])

    def boom(_: PipelineConfig) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(orch, "_run_single", boom)
    with pytest.raises(RuntimeError):
        orch.run_all()


def test_run_all_continues_on_known_error(tmp_path, monkeypatch):
    cfg1 = _cfg(tmp_path, "a")
    cfg2 = _cfg(tmp_path, "b")
    orch = BatchOrchestrator([cfg1, cfg2])
    call_order: list[str] = []

    def maybe_error(cfg: PipelineConfig) -> None:
        call_order.append(cfg.folder_id)
        if cfg.folder_id == "a":
            raise ValueError("expected")

    monkeypatch.setattr(orch, "_run_single", maybe_error)
    orch.run_all()
    assert call_order == ["a", "b"]


def test_run_single_reraises_unexpected_outlier_error(tmp_path, monkeypatch):
    cfg = _cfg(tmp_path)
    orch = BatchOrchestrator([cfg])

    class DS:
        def __init__(self) -> None:
            self.config = type("C", (), {"folder": "out"})

    monkeypatch.setattr(orch.data_loader, "_load_data", lambda cfg: (DS(), None, None, None))
    monkeypatch.setattr(orch.visualization_runner, "_generate_clouds_outliers", lambda *a, **k: None)
    monkeypatch.setattr(orch.statistics_runner, "_compute_statistics", lambda *a, **k: None)

    def boom(*_args, **_kwargs):
        raise RuntimeError("outliers")

    monkeypatch.setattr(orch.outlier_handler, "_exclude_outliers", boom)

    with pytest.raises(RuntimeError):
        orch._run_single(cfg)
