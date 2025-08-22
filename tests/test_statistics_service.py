import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from statistics_service import StatisticsService


def test_calc_stats_inlier_outlier_split():
    distances = np.concatenate([np.zeros(100), np.array([100.0])])
    stats = StatisticsService.calc_stats(distances)

    assert stats["Outlier Count"] == 1
    assert stats["Inlier Count"] == 100
    assert stats["Mean Inlier"] == 0.0
    assert stats["Max Inlier"] == 0.0
    assert stats["Mean Outlier"] == 100.0
    assert stats["Valid Count Inlier"] == 100
