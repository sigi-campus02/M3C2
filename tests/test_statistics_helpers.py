import numpy as np

from src.statistics_service import StatisticsService


def test_load_params(tmp_path):
    params = tmp_path / "params.txt"
    params.write_text("NormalScale=1.5\nSearchScale=2.5\n")

    normal, search = StatisticsService._load_params(str(params))

    assert normal == 1.5
    assert search == 2.5


def test_compute_outliers():
    data = np.array([0.0, 1.0, 2.0, 100.0])
    rms = 1.0  # deliberately small to classify 100 as outlier
    median = float(np.median(data))

    result = StatisticsService._compute_outliers(data, rms, median)

    assert result["outlier_count"] == 1
    assert result["inlier_count"] == 3
    assert result["mean_out"] == 100.0


def test_fit_distributions():
    rng = np.random.default_rng(42)
    data = rng.normal(0, 1, size=1000)
    clipped = data[(data >= -3) & (data <= 3)]
    hist, bin_edges = np.histogram(clipped, bins=10, range=(-3, 3))

    result = StatisticsService._fit_distributions(
        clipped, hist.astype(float), bin_edges, min_expected=None
    )

    assert abs(result["mu"]) < 0.1
    assert abs(result["std"] - 1) < 0.1

