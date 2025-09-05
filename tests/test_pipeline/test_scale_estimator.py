"""Tests for the scale estimation helpers and strategies.

This module contains small dummy implementations of strategies and estimators
to validate the behaviour of :class:`~m3c2.pipeline.scale_estimator.ScaleEstimator`
and concrete strategy classes.  The tests focus on error handling and the
integration of the scale determination pipeline.
"""

from __future__ import annotations

import numpy as np
import pytest

from m3c2.config.pipeline_config import PipelineConfig
from m3c2.m3c2_core.param_handler.scale_estimator import ScaleEstimator
from m3c2.m3c2_core.param_handler import scale_estimator as se_module
from m3c2.m3c2_core.param_handler.strategies import RadiusScanStrategy
from m3c2.m3c2_core.param_handler.strategies import ScaleScan


class DummyStrategy:
    """Minimal scan strategy used for testing.

    Parameters
    ----------
    sample_size : int, optional
        Number of points sampled from the input data.
    """

    def __init__(self, sample_size=None):
        self.sample_size = sample_size


class DummyEstimator:
    """Estimator using ``DummyStrategy`` for the tests.

    Parameters
    ----------
    strategy : DummyStrategy
        The strategy instance used to compute scale statistics.
    """

    def __init__(self, strategy):
        self.strategy = strategy

    def estimate_min_spacing(self, corepoints):
        # verify that strategy was instantiated with the configured sample size
        assert self.strategy.sample_size == 1
        return 0.5

    def scan_scales(self, corepoints, avg_spacing):
        assert avg_spacing == 0.5
        return [
            ScaleScan(
                scale=1.0,
                valid_normals=1,
                mean_population=0.0,
                roughness=0.1,
                coverage=1.0,
                mean_lambda3=0.0,
            )
        ]

    @staticmethod
    def select_scales(scans):
        return 1.0, 2.0


class DummyEstimatorNoScans(DummyEstimator):
    """Estimator returning no scan results to trigger error paths."""

    def scan_scales(self, corepoints, avg_spacing):
        return []

    @staticmethod
    def select_scales(scans):
        raise AssertionError("select_scales should not be called")


def _minimal_cfg(**kwargs) -> PipelineConfig:
    """Create a minimal :class:`PipelineConfig` for the tests.

    Parameters
    ----------
    **kwargs
        Additional configuration overrides.

    Returns
    -------
    PipelineConfig
        Configuration object with defaults updated by ``kwargs``.
    """

    defaults = dict(
        data_dir="",
        folder_id="",
        filename_comparison="",
        filename_reference="",
        filename_singlecloud="",
        use_subsampled_corepoints=0,
        only_stats=False,
        stats_singleordistance="single",
        sample_size=1,
        project="proj",
    )
    defaults.update(kwargs)
    return PipelineConfig(**defaults)


def test_determine_scales_with_mock_strategy(monkeypatch):
    """Test scale determination with mocked strategy and estimator.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture for patching strategy and estimator classes.

    Returns
    -------
    None
    """

    monkeypatch.setitem(se_module.STRATEGIES, "dummy", DummyStrategy)
    monkeypatch.setattr(se_module, "ParamEstimator", DummyEstimator)

    cfg = _minimal_cfg()
    estimator = ScaleEstimator(strategy_name="dummy")

    normal, projection = estimator.determine_scales(cfg, np.zeros((1, 3)))

    assert normal == 1.0
    assert projection == 2.0


def test_determine_scales_unknown_strategy(monkeypatch):
    """Test error handling for unknown strategy names.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest fixture for recomparison available strategies.

    Returns
    -------
    None
    """

    monkeypatch.setattr(se_module, "STRATEGIES", {})
    cfg = _minimal_cfg()
    estimator = ScaleEstimator(strategy_name="radius")

    with pytest.raises(ValueError):
        estimator.determine_scales(cfg, np.zeros((1, 3)))


def test_determine_scales_empty_scan_results(monkeypatch):
    """Test that empty scan results raise ``ValueError``.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture for patching strategy and estimator implementations.

    Returns
    -------
    None
    """

    monkeypatch.setitem(se_module.STRATEGIES, "dummy", DummyStrategy)
    monkeypatch.setattr(se_module, "ParamEstimator", DummyEstimatorNoScans)

    cfg = _minimal_cfg()
    estimator = ScaleEstimator(strategy_name="dummy")

    with pytest.raises(ValueError, match="keine Skalen"):
        estimator.determine_scales(cfg, np.zeros((1, 3)))


def test_radius_scan_strategy_evaluate_radius_scale_plane():
    """Validate ``RadiusScanStrategy.evaluate_radius_scale`` on a plane.

    Returns
    -------
    None
    """

    xs, ys = np.meshgrid(range(3), range(3))
    pts = np.column_stack((xs.ravel(), ys.ravel(), np.zeros(xs.size)))

    strategy = RadiusScanStrategy(min_points=3, log_each_scale=False)
    res = strategy.evaluate_radius_scale(pts, neighborhood_radius=1.5)

    assert res["valid_normals"] == 9
    assert res["coverage"] == 1.0
    assert res["roughness"] == pytest.approx(0.0)
    assert res["mean_lambda3"] == pytest.approx(0.0)
    assert res["mean_population"] == pytest.approx(5.444444444444445)
