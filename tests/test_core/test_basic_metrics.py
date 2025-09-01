import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from m3c2.core.statistics.basic_metrics import basic_stats, fit_distributions


def test_basic_stats_empty_array():
    res = basic_stats(np.array([]), tolerance=0.1)
    assert res["Valid Count"] == 0
    assert np.isnan(res["Mean"])
    assert np.isnan(res["Jaccard Index"])


def test_basic_stats_computation():
    arr = np.array([1.0, 2.0, 3.0])
    res = basic_stats(arr, tolerance=2.0)
    assert res["Valid Count"] == 3
    assert res["Min"] == 1.0
    assert res["Max"] == 3.0
    assert np.isclose(res["Mean"], 2.0)
    assert np.isclose(res["Median"], 2.0)
    assert np.isclose(res["Std Empirical"], np.std(arr))
    assert np.isclose(res["Within-Tolerance"], 2.0 / 3.0)
    assert np.isclose(res["Jaccard Index"], 1.0 / 3.0)


def test_fit_distributions_basic():
    rng = np.random.default_rng(0)
    data = rng.normal(loc=1.0, scale=2.0, size=500)
    hist, bin_edges = np.histogram(data, bins=20)
    res = fit_distributions(data, hist, bin_edges, None)
    assert np.isclose(res["mu"], np.mean(data), atol=0.1)
    assert np.isclose(res["std"], np.std(data), atol=0.1)
    assert res["pearson_gauss"] >= 0
    assert res["pearson_weib"] >= 0


def test_fit_distributions_length_mismatch():
    with pytest.raises(AssertionError):
        fit_distributions(np.arange(10), np.ones(5), np.arange(6), None)
