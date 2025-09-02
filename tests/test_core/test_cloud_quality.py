"""Tests for cloud quality statistics calculations."""

import sys
from pathlib import Path

import numpy as np
import pytest
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from m3c2.core.statistics.cloud_quality import _calc_single_cloud_stats

def test_calc_single_cloud_stats_empty():
    """Ensure empty arrays raise ``ValueError``.

    Parameters
    ----------
    None

    Raises
    ------
    ValueError
        If the input point array is empty.
    """

    with pytest.raises(ValueError):
        _calc_single_cloud_stats(np.array([]))


def test_calc_single_cloud_stats_wrong_shape():
    """Ensure arrays with invalid shape raise ``ValueError``.

    Parameters
    ----------
    None

    Raises
    ------
    ValueError
        If the input array does not have shape ``(n, 3)``.
    """

    with pytest.raises(ValueError):
        _calc_single_cloud_stats(np.zeros((3, 2)))


def test_calc_single_cloud_stats_basic():
    """Compute statistics for a small point cloud.

    Parameters
    ----------
    None

    Returns
    -------
    None
        Asserts that computed statistics match expected values.
    """

    points = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 2.0],
            [1.0, 1.0, 3.0],
        ]
    )
    # compute expected NN distances for comparison
    k = 2
    nn = NearestNeighbors(n_neighbors=min(k + 1, len(points))).fit(points)
    dists, _ = nn.kneighbors(points)
    expected_all = float(np.mean(dists[:, 1:]))
    expected_kth = float(np.mean(dists[:, min(k, dists.shape[1] - 1)]))

    stats = _calc_single_cloud_stats(
        points,
        area_m2=1.0,
        radius=2.0,
        k=k,
        sample_size=None,
        use_convex_hull=False,
    )

    assert stats["Num Points"] == 4
    assert stats["Area Source"] == "given"
    assert stats["Density Global [pt/m^2]"] == pytest.approx(4.0)
    assert stats["Z Min"] == 0.0
    assert stats["Z Max"] == 3.0
    assert stats["Z Mean"] == pytest.approx(1.5)
    assert stats["Sampled Points"] == 4
    assert stats["Mean NN Dist All"] == pytest.approx(expected_all)
    assert stats["Mean NN Dist k-th"] == pytest.approx(expected_kth)
