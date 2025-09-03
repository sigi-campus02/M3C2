"""Utilities for analysing M3C2 distance arrays.

This module provides helpers that previously lived on
``StatisticsService``.  The functions are now available at module level
to keep responsibilities focused and make reuse easier.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import logging
import os

import numpy as np

from .distance_basic_metrics import basic_stats, fit_distributions
from .distance_outlier_metrics import compute_outliers, get_outlier_mask

logger = logging.getLogger(__name__)


def _load_params(params_path: Optional[str]) -> Tuple[float, float]:
    """Load M3C2 configuration values from a parameter file.

    The M3C2 computation writes a small text file with the parameters used
    during the run.  This helper reads the file and extracts the normal and
    search scales that are stored in lines beginning with ``"NormalScale="``
    and ``"SearchScale="``.  When no file is supplied or a value is
    missing the respective scale defaults to ``numpy.nan``.

    Parameters
    ----------
    params_path:
        Path to the parameter file.  If ``None`` or the file does not
        exist, ``numpy.nan`` is returned for both values.

    Returns
    -------
    Tuple[float, float]
        The ``(normal_scale, search_scale)`` read from the file.
    """

    normal_scale = np.nan
    search_scale = np.nan
    if params_path and os.path.exists(params_path):
        try:
            with open(params_path, "r") as f:
                for line in f:
                    if line.startswith("NormalScale="):
                        normal_scale = float(line.strip().split("=")[1])
                    elif line.startswith("SearchScale="):
                        search_scale = float(line.strip().split("=")[1])
        except (OSError, ValueError) as exc:
            logger.warning("Failed to load params from %s: %s", params_path, exc)
            return np.nan, np.nan
    return normal_scale, search_scale


def calc_stats(
    distances: np.ndarray,
    params_path: Optional[str] = None,
    bins: int = 256,
    range_override: Optional[Tuple[float, float]] = None,
    min_expected: Optional[float] = None,
    tolerance: float = 0.01,
    outlier_multiplicator: float = 3.0,
    outlier_method: str = "rmse",
) -> Dict:
    """Compute descriptive statistics for a set of M3C2 distances.

    Parameters
    ----------
    distances : np.ndarray
        Array of distance values. ``NaN`` entries are ignored.
    params_path : Optional[str], optional
        Path to a ``*_m3c2_params.txt`` file from which the normal and
        search scale are loaded.
    bins : int, default 256
        Number of histogram bins used for distribution fitting.
    range_override : Optional[Tuple[float, float]], optional
        Explicit ``(min, max)`` limits used instead of the data range.
    min_expected : Optional[float], optional
        Minimum expected count per histogram bin. If provided it is passed
        to the distribution fitting routine.
    tolerance : float, default 0.01
        Absolute distance threshold used for several ratio metrics.
    outlier_multiplicator : float, default 3.0
        Multiplicative factor applied to the chosen outlier metric to
        derive an inlier threshold.
    outlier_method : str, default "rmse"
        Name of the method used to compute the outlier threshold.

    Returns
    -------
    Dict
        A mapping with numerous aggregated statistics. The dictionary
        contains overall counts and sums, summary metrics (mean, median,
        RMS, standard deviation, MAE, NMAD), inlier/outlier information,
        quantiles and interquartile ranges for all data and inliers only,
        goodness-of-fit parameters for Gaussian and Weibull distributions
        as well as higher order moments such as skewness and kurtosis.
    """

    total_count = len(distances)
    nan_count = int(np.isnan(distances).sum())
    valid = distances[~np.isnan(distances)]
    if valid.size == 0:
        raise ValueError("No valid distances")

    if range_override is None:
        data_min, data_max = float(np.min(valid)), float(np.max(valid))
    else:
        data_min, data_max = map(float, range_override)

    clipped = valid[(valid >= data_min) & (valid <= data_max)]
    if clipped.size == 0:
        raise ValueError("All values fall outside the selected range")

    stats_all = basic_stats(clipped, tolerance)
    valid_sum = stats_all["Valid Sum"]
    valid_squared_sum = stats_all["Valid Squared Sum"]
    avg = stats_all["Mean"]
    med = stats_all["Median"]
    rms = stats_all["RMS"]
    std_empirical = stats_all["Std Empirical"]
    mae = stats_all["MAE"]
    nmad = stats_all["NMAD"]

    hist, bin_edges = np.histogram(clipped, bins=bins, range=(data_min, data_max))
    hist = hist.astype(float)

    fit_results = fit_distributions(clipped, hist, bin_edges, min_expected)
    mu = fit_results["mu"]
    std = fit_results["std"]
    pearson_gauss = fit_results["pearson_gauss"]
    a = fit_results["a"]
    b = fit_results["b"]
    loc = fit_results["loc"]
    pearson_weib = fit_results["pearson_weib"]
    skew_weibull = fit_results["skew_weibull"]
    mode_weibull = fit_results["mode_weibull"]

    normal_scale, search_scale = _load_params(params_path)

    outlier_mask, outlier_threshold = get_outlier_mask(
        clipped, outlier_method, outlier_multiplicator
    )
    inliers = clipped[~outlier_mask]
    outliers = clipped[outlier_mask]
    outlier_info = compute_outliers(inliers, outliers)
    outlier_count = outlier_info["outlier_count"]
    inlier_count = outlier_info["inlier_count"]
    mean_out = outlier_info["mean_out"]
    std_out = outlier_info["std_out"]
    pos_out = outlier_info["pos_out"]
    neg_out = outlier_info["neg_out"]
    pos_in = outlier_info["pos_in"]
    neg_in = outlier_info["neg_in"]

    stats_in = basic_stats(inliers, tolerance)
    mean_in = stats_in["Mean"]
    std_in = stats_in["Std Empirical"]
    mae_in = stats_in["MAE"]
    nmad_in = stats_in["NMAD"]
    min_in = stats_in["Min"]
    max_in = stats_in["Max"]
    median_in = stats_in["Median"]
    rms_in = stats_in["RMS"]
    q05_in = stats_in["Q05"]
    q25_in = stats_in["Q25"]
    q75_in = stats_in["Q75"]
    q95_in = stats_in["Q95"]
    iqr_in = stats_in["IQR"]
    skew_in = stats_in["Skewness"]
    kurt_in = stats_in["Kurtosis"]
    share_abs_gt_in = stats_in["Anteil |Distanz| > 0.01"]
    share_2std_in = stats_in["Anteil [-2Std,2Std]"]
    max_abs_in = stats_in["Max |Distanz|"]
    bias_in = stats_in["Bias"]
    within_tolerance_in = stats_in["Within-Tolerance"]
    jaccard_in = stats_in["Jaccard Index"]
    dice_in = stats_in["Dice Coefficient"]
    valid_count_in = stats_in["Valid Count"]
    valid_sum_in = stats_in["Valid Sum"]
    valid_squared_sum_in = stats_in["Valid Squared Sum"]

    bias = stats_all["Bias"]
    within_tolerance = stats_all["Within-Tolerance"]

    icc = np.nan
    mean_dist = float(np.mean(clipped))
    std_dist = float(np.std(clipped))
    ccc = (
        (2 * mean_dist * std_dist) / (mean_dist**2 + std_dist**2)
        if mean_dist != 0
        else np.nan
    )

    bland_altman_lower = bias - 1.96 * std_dist
    bland_altman_upper = bias + 1.96 * std_dist

    jaccard_index = stats_all["Jaccard Index"]
    dice_coefficient = stats_all["Dice Coefficient"]

    return {
        "Total Points": total_count,
        "NaN": nan_count,
        "% NaN": (nan_count / total_count) if total_count > 0 else np.nan,
        "% Valid": (1 - nan_count / total_count) if total_count > 0 else np.nan,
        "Valid Count": int(clipped.size),
        "Valid Sum": valid_sum,
        "Valid Squared Sum": valid_squared_sum,
        "Valid Count Inlier": int(valid_count_in),
        "Valid Sum Inlier": valid_sum_in,
        "Valid Squared Sum Inlier": valid_squared_sum_in,
        "Normal Scale": normal_scale,
        "Search Scale": search_scale,
        "Min": float(np.nanmin(distances)),
        "Max": float(np.nanmax(distances)),
        "Mean": avg,
        "Median": med,
        "RMS": rms,
        "Std Empirical": std_empirical,
        "MAE": mae,
        "NMAD": nmad,
        "Min Inlier": min_in,
        "Max Inlier": max_in,
        "Mean Inlier": mean_in,
        "Median Inlier": median_in,
        "RMS Inlier": rms_in,
        "Std Inlier": std_in,
        "MAE Inlier": mae_in,
        "NMAD Inlier": nmad_in,
        "Outlier Count": outlier_count,
        "Inlier Count": inlier_count,
        "Mean Outlier": mean_out,
        "Std Outlier": std_out,
        "Pos Outlier": pos_out,
        "Neg Outlier": neg_out,
        "Pos Inlier": pos_in,
        "Neg Inlier": neg_in,
        "Outlier Multiplicator": outlier_multiplicator,
        "Outlier Threshold": outlier_threshold,
        "Outlier Method": outlier_method,
        "Q05": stats_all["Q05"],
        "Q25": stats_all["Q25"],
        "Q75": stats_all["Q75"],
        "Q95": stats_all["Q95"],
        "IQR": stats_all["IQR"],
        "Q05 Inlier": q05_in,
        "Q25 Inlier": q25_in,
        "Q75 Inlier": q75_in,
        "Q95 Inlier": q95_in,
        "IQR Inlier": iqr_in,
        "Gauss Mean": float(mu),
        "Gauss Std": float(std),
        "Gauss Chi2": float(pearson_gauss),
        "Weibull a": float(a),
        "Weibull b": float(b),
        "Weibull shift": float(loc),
        "Weibull mode": mode_weibull,
        "Weibull skewness": skew_weibull,
        "Weibull Chi2": float(pearson_weib),
        "Skewness": stats_all["Skewness"],
        "Kurtosis": stats_all["Kurtosis"],
    }

