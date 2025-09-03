"""Tests for the outlier exclusion functionality.

This module verifies that synthetic distance data is correctly split into
inliers and outliers across various statistical methods and through the
``OutlierDetector`` class interface.
"""

import numpy as np

from m3c2.archive_moduls.exclude_outliers import (
    OutlierConfig,
    OutlierDetector,
    OutlierResult,
    exclude_outliers,
)


def _write_distances(path):
    """Create a synthetic distance file.

    Parameters
    ----------
    path : str or path-like
        Destination for the generated distance data.
    """

    data = np.array(
        [
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.1],
            [0.0, 0.0, 0.0, 0.2],
            [0.0, 0.0, 0.0, 0.3],
            [0.0, 0.0, 0.0, -0.1],
            [0.0, 0.0, 0.0, 0.15],
            [0.0, 0.0, 0.0, 5.0],
            [0.0, 0.0, 0.0, -5.0],
            [0.0, 0.0, 0.0, np.nan],
        ]
    )
    np.savetxt(path, data, header="x y z distance")


def _assert_result(res: OutlierResult) -> None:
    """Check that the outlier detection result matches expectations.

    Parameters
    ----------
    res : OutlierResult
        Output from the outlier detection routine.

    Raises
    ------
    AssertionError
        If the result does not contain the expected number of inliers and
        outliers.
    """

    assert res.inliers.shape[0] == 6
    assert res.outliers.shape[0] == 2
    assert set(np.round(res.outliers[:, 3])) == {5.0, -5.0}


def test_exclude_outliers_methods(tmp_path):
    """Validate exclusion across multiple statistical methods.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.
    """

    file_path = tmp_path / "distances.txt"
    _write_distances(file_path)

    for method, factor in [("rmse", 1.0), ("iqr", 3.0), ("std", 1.0), ("nmad", 3.0)]:
        res = exclude_outliers(str(file_path), method, factor)
        _assert_result(res)

        inlier_file = tmp_path / f"distances_inlier_{method}.txt"
        outlier_file = tmp_path / f"distances_outlier_{method}.txt"
        assert inlier_file.exists()
        assert outlier_file.exists()


def test_outlier_detector_class(tmp_path):
    """Ensure the ``OutlierDetector`` class identifies outliers correctly.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.
    """

    file_path = tmp_path / "distances.txt"
    _write_distances(file_path)

    config = OutlierConfig(
        dists_path=str(file_path), method="rmse", outlier_multiplicator=1.0
    )
    detector = OutlierDetector(config)
    res = detector.run()
    _assert_result(res)

