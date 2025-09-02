import numpy as np

from m3c2.archive_moduls.exclude_outliers import (
    OutlierConfig,
    OutlierDetector,
    OutlierResult,
    exclude_outliers,
)


def _write_distances(path):
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
    assert res.inliers.shape[0] == 6
    assert res.outliers.shape[0] == 2
    assert set(np.round(res.outliers[:, 3])) == {5.0, -5.0}


def test_exclude_outliers_methods(tmp_path):
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
    file_path = tmp_path / "distances.txt"
    _write_distances(file_path)

    config = OutlierConfig(
        file_path=str(file_path), method="rmse", outlier_multiplicator=1.0
    )
    detector = OutlierDetector(config)
    res = detector.run()
    _assert_result(res)

