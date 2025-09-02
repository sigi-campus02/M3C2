from __future__ import annotations

from typing import Dict, Optional

import logging
import numpy as np
import pandas as pd
from scipy.stats import norm, weibull_min


logger = logging.getLogger(__name__)


def basic_stats(values: np.ndarray, tolerance: float) -> Dict[str, float]:
    """Berechne Grundkennzahlen für ein Werte-Array."""
    logger.info("basic_stats received %d values", values.size)
    if values.size == 0:
        result = {
            "Valid Count": 0,
            "Valid Sum": 0.0,
            "Valid Squared Sum": 0.0,
            "Min": np.nan,
            "Max": np.nan,
            "Mean": np.nan,
            "Median": np.nan,
            "RMS": np.nan,
            "Std Empirical": np.nan,
            "MAE": np.nan,
            "NMAD": np.nan,
            "Q05": np.nan,
            "Q25": np.nan,
            "Q75": np.nan,
            "Q95": np.nan,
            "IQR": np.nan,
            "Skewness": np.nan,
            "Kurtosis": np.nan,
            "Anteil |Distanz| > 0.01": np.nan,
            "Anteil [-2Std,2Std]": np.nan,
            "Max |Distanz|": np.nan,
            "Bias": np.nan,
            "Within-Tolerance": np.nan,
            "Jaccard Index": np.nan,
            "Dice Coefficient": np.nan,
        }
        logger.info(
            "basic_stats summary: min=%s, max=%s, mean=%s",
            result["Min"],
            result["Max"],
            result["Mean"],
        )
        return result

    valid_count = int(values.size)
    valid_sum = float(np.sum(values))
    valid_squared_sum = float(np.sum(values ** 2))
    min_val = float(np.min(values))
    max_val = float(np.max(values))
    mean_val = float(np.mean(values))
    median_val = float(np.median(values))
    rms_val = float(np.sqrt(np.mean(values ** 2)))
    std_emp = float(np.std(values))
    mae = float(np.mean(np.abs(values)))
    mad = float(np.median(np.abs(values - median_val)))
    nmad = float(1.4826 * mad)
    q05 = float(np.percentile(values, 5))
    q25 = float(np.percentile(values, 25))
    q75 = float(np.percentile(values, 75))
    q95 = float(np.percentile(values, 95))
    iqr = float(q75 - q25)
    skew = float(pd.Series(values).skew())
    kurt = float(pd.Series(values).kurt())
    share_abs_gt = float(np.mean(np.abs(values) > 0.01))
    share_2std = float(np.mean((values > -2 * std_emp) & (values < 2 * std_emp)))
    max_abs = float(np.max(np.abs(values)))
    bias = mean_val
    within_tolerance = float(np.mean(np.abs(values) <= tolerance))
    intersection = np.sum((values > -tolerance) & (values < tolerance))
    union = len(values)
    jaccard_index = intersection / union if union > 0 else np.nan
    dice_coefficient = (2 * intersection) / (2 * union) if union > 0 else np.nan

    result = {
        "Valid Count": valid_count,
        "Valid Sum": valid_sum,
        "Valid Squared Sum": valid_squared_sum,
        "Min": min_val,
        "Max": max_val,
        "Mean": mean_val,
        "Median": median_val,
        "RMS": rms_val,
        "Std Empirical": std_emp,
        "MAE": mae,
        "NMAD": nmad,
        "Q05": q05,
        "Q25": q25,
        "Q75": q75,
        "Q95": q95,
        "IQR": iqr,
        "Skewness": skew,
        "Kurtosis": kurt,
        "Anteil |Distanz| > 0.01": share_abs_gt,
        "Anteil [-2Std,2Std]": share_2std,
        "Max |Distanz|": max_abs,
        "Bias": bias,
        "Within-Tolerance": within_tolerance,
        "Jaccard Index": jaccard_index,
        "Dice Coefficient": dice_coefficient,
    }
    logger.info(
        "basic_stats summary: min=%f, max=%f, mean=%f",
        min_val,
        max_val,
        mean_val,
    )
    return result


def fit_distributions(
    clipped: np.ndarray,
    hist: np.ndarray,
    bin_edges: np.ndarray,
    min_expected: Optional[float],
) -> Dict[str, float]:
    """Fit Gaussian and Weibull distributions and compute Chi² metrics."""
    N = int(hist.sum())
    assert N == len(clipped), f"Histogram N ({N}) != len(clipped) ({len(clipped)})"

    mu, std = norm.fit(clipped)

    cdfL = norm.cdf(bin_edges[:-1], mu, std)
    cdfR = norm.cdf(bin_edges[1:], mu, std)
    expected_gauss = N * (cdfR - cdfL)

    eps = 1e-12
    thr = min_expected if min_expected is not None else eps
    maskG = expected_gauss > thr
    pearson_gauss = float(
        np.sum((hist[maskG] - expected_gauss[maskG]) ** 2 / expected_gauss[maskG])
    )

    a, loc, b = weibull_min.fit(clipped)
    cdfL = weibull_min.cdf(bin_edges[:-1], a, loc=loc, scale=b)
    cdfR = weibull_min.cdf(bin_edges[1:], a, loc=loc, scale=b)
    expected_weib = N * (cdfR - cdfL)
    maskW = expected_weib > thr
    pearson_weib = float(
        np.sum((hist[maskW] - expected_weib[maskW]) ** 2 / expected_weib[maskW])
    )

    skew_weibull = float(weibull_min(a, loc=loc, scale=b).stats(moments="s"))
    mode_weibull = float(loc + b * ((a - 1) / a) ** (1 / a)) if a > 1 else float(loc)

    logger.info(
        "fit_distributions Gaussian: mu=%f, std=%f, chi2=%f",
        mu,
        std,
        pearson_gauss,
    )
    logger.info(
        "fit_distributions Weibull: a=%f, b=%f, loc=%f, chi2=%f",
        a,
        b,
        loc,
        pearson_weib,
    )

    return {
        "mu": float(mu),
        "std": float(std),
        "pearson_gauss": pearson_gauss,
        "a": float(a),
        "loc": float(loc),
        "b": float(b),
        "pearson_weib": pearson_weib,
        "skew_weibull": skew_weibull,
        "mode_weibull": mode_weibull,
    }
