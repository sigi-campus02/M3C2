"""Error handling tests for :mod:`m3c2.pipeline.batch_orchestrator`.

This module exercises the :class:`~m3c2.pipeline.batch_orchestrator.BatchOrchestrator`
in a variety of failure scenarios to ensure robust error propagation:

* ``run_all`` continues processing when a configuration raises
  ``ValueError``.
* ``run_all`` re-raises unexpected exceptions.
* ``_run_single`` surfaces unexpected runtime errors from downstream
  components.
"""

from __future__ import annotations

import numpy as np
import pytest

from types import SimpleNamespace

from m3c2.config.pipeline_config import PipelineConfig
from m3c2.pipeline.batch_orchestrator import BatchOrchestrator


def _cfg(tmp_path, folder_id="run"):
    return PipelineConfig(
        data_dir=str(tmp_path),
        folder_id=folder_id,
        filename_mov="mov.xyz",
        filename_ref="ref.xyz",
        filename_singlecloud="sc.xyz",
        mov_as_corepoints=True,
        use_subsampled_corepoints=1,
        only_stats=True,
        stats_singleordistance="single",
        sample_size=1,
        project="proj",
    )


def test_run_all_continues_on_value_error(monkeypatch, tmp_path):
    cfg1 = _cfg(tmp_path, "a")
    cfg2 = _cfg(tmp_path, "b")
    orchestrator = BatchOrchestrator([cfg1, cfg2])

    calls = []

    def side_effect(cfg):
        calls.append(cfg.folder_id)
        if cfg is cfg1:
            raise ValueError("boom")

    monkeypatch.setattr(orchestrator, "_run_single", side_effect)

    orchestrator.run_all()

    assert calls == ["a", "b"]


def test_run_all_logs_unexpected(monkeypatch, tmp_path, caplog):
    cfg = _cfg(tmp_path, "a")
    orchestrator = BatchOrchestrator([cfg])

    def side_effect(cfg):
        raise RuntimeError("unexpected")

    monkeypatch.setattr(orchestrator, "_run_single", side_effect)

    with caplog.at_level("ERROR"):
        orchestrator.run_all()

    assert any("Unerwarteter Fehler" in r.message for r in caplog.records)


def test_run_single_propagates_unexpected(monkeypatch, tmp_path):
    cfg = _cfg(tmp_path, "a")
    orchestrator = BatchOrchestrator([cfg])

    dummy_ds = SimpleNamespace(config=SimpleNamespace(folder=str(tmp_path)))
    arr = np.zeros((1, 3))

    def fake_load_data(cfg, mode="multicloud"):
        if mode == "singlecloud":
            return arr
        return (dummy_ds, arr, arr, arr)

    monkeypatch.setattr(orchestrator.data_loader, "load_data", fake_load_data)
    monkeypatch.setattr(
        orchestrator.scale_estimator,
        "determine_scales",
        lambda *a, **k: (0.0, 0.0),
    )
    monkeypatch.setattr(
        orchestrator.statistics_runner,
        "single_cloud_statistics_handler",
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("bad")),
    )

    with pytest.raises(RuntimeError):
        orchestrator._run_single(cfg)

