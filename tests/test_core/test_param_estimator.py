"""Tests for the :mod:`m3c2.core.param_estimator` module."""

import numpy as np
import pytest

from m3c2.pipeline.strategies import ScaleScan
from m3c2.core.param_estimator import ParamEstimator


class DummyStrategy:
    """Strategy stub returning predefined scans for testing."""

    def __init__(self, scans):
        self.scans = scans
        self.called_with = None

    def scan(self, points, avg_spacing):
        self.called_with = (points, avg_spacing)
        return self.scans

def test_estimate_min_spacing_and_scan(tmp_path):
    """Test estimation of minimum spacing and scanning scales.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.

    Returns
    -------
    None
    """

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
        ]
    )

    scans = [
        ScaleScan(scale=1.0, valid_normals=10, mean_population=0, roughness=0.1, coverage=1.0, mean_lambda3=0.01),
    ]
    strategy = DummyStrategy(scans)
    estimator = ParamEstimator(strategy=strategy, k_neighbors=2)

    spacing = estimator.estimate_min_spacing(points)
    assert spacing == pytest.approx(1.0, abs=0.1)

    result_scans = estimator.scan_scales(points, spacing)
    assert result_scans is scans
    assert strategy.called_with == (points, spacing)

def test_select_scales():
    """Test selecting scales for normal and projection estimation.

    Returns
    -------
    None
    """

    scans = [
        ScaleScan(scale=1.0, valid_normals=10, mean_population=0, roughness=0.1, coverage=1.0, mean_lambda3=0.01),
        ScaleScan(scale=2.0, valid_normals=8, mean_population=0, roughness=0.05, coverage=1.0, mean_lambda3=0.02),
    ]
    normal, projection = ParamEstimator.select_scales(scans)
    assert normal == pytest.approx(2.0)
    assert projection == pytest.approx(2.0)

