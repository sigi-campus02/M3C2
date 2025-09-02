"""High-level service functions for M3C2 statistics.

The module exposes the :class:`StatisticsService` facade which bundles
operations for analysing M3C2 distance arrays and point clouds.  Key
helpers include:

* :func:`StatisticsService.calc_stats` – derive descriptive metrics and
  distribution fits for a set of distances.
* :func:`StatisticsService.compute_m3c2_statistics` – aggregate statistics
  for multiple folders and export the combined results.
* :func:`StatisticsService.calc_single_cloud_stats` – evaluate quality
  metrics for individual point clouds.

These high-level functions form the public API used by the CLI and other
modules when working with statistics.
"""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import logging
import numpy as np
import pandas as pd

from m3c2.io.datasource import DataSource

from .basic_metrics import basic_stats, fit_distributions
from .outliers import compute_outliers, get_outlier_mask
from .cloud_quality import _calc_single_cloud_stats
from .exporters import (
    _append_df_to_excel,
    _append_df_to_json,
    write_cloud_stats,
)


logger = logging.getLogger(__name__)


class StatisticsService:
    """Facade for computing and exporting statistics.

    The service wraps low-level metric calculations and exposes
    three main entry points:

    * :meth:`calc_stats` – compute statistics for a distance array.
    * :meth:`compute_m3c2_statistics` – aggregate results across
      multiple folders and optionally persist them.
    * :meth:`calc_single_cloud_stats` – derive metrics for individual
      point clouds.
    """

    @staticmethod
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

        normal_scale, search_scale = StatisticsService._load_params(params_path)

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

    @classmethod
    def compute_m3c2_statistics(
        cls,
        folder_ids: List[str],
        filename_ref: str = "",
        process_python_CC: str = "python",
        bins: int = 256,
        range_override: Optional[Tuple[float, float]] = None,
        min_expected: Optional[float] = None,
        out_path: str = "m3c2_stats_all.xlsx",
        sheet_name: str = "Results",
        output_format: str = "excel",
        outlier_multiplicator: float = 3.0,
        outlier_method: str = "rmse",
    ) -> pd.DataFrame:
        logger.info("Starting compute_m3c2_statistics for %d folders", len(folder_ids))
        rows: List[Dict] = []

        for fid in folder_ids:
            if process_python_CC == "python":
                py_dist_path = cls._resolve(
                    fid, f"python_{filename_ref}_m3c2_distances.txt"
                )
                py_params_path = cls._resolve(
                    fid, f"python_{filename_ref}_m3c2_params.txt"
                )
                if os.path.exists(py_dist_path):
                    values = np.loadtxt(py_dist_path)
                    stats = cls.calc_stats(
                        values,
                        params_path=py_params_path if os.path.exists(py_params_path) else None,
                        bins=bins,
                        range_override=range_override,
                        min_expected=min_expected,
                        outlier_multiplicator=outlier_multiplicator,
                        outlier_method=outlier_method,
                    )
                    rows.append(
                        {
                            "Folder": fid,
                            "Version": filename_ref or "",
                            "Distances Path": py_dist_path,
                            "Params Path": py_params_path if os.path.exists(py_params_path) else "",
                            **stats,
                        }
                    )

            if process_python_CC == "CC":
                cc_path = cls._resolve(fid, f"CC_{filename_ref}_m3c2_distances.txt")
                cc_params_path = cls._resolve(fid, f"CC_{filename_ref}_m3c2_params.txt")
                if os.path.exists(cc_path):
                    try:
                        df = pd.read_csv(cc_path, sep=";")
                        col = "M3C2 distance"
                        if col in df.columns:
                            values = df[col].astype(float).values
                            stats = cls.calc_stats(
                                values,
                                params_path=cc_params_path if os.path.exists(cc_params_path) else None,
                                bins=bins,
                                range_override=range_override,
                                min_expected=min_expected,
                                outlier_multiplicator=outlier_multiplicator,
                                outlier_method=outlier_method,
                            )
                            rows.append(
                                {
                                    "Folder": fid,
                                    "Version": filename_ref or "",
                                    "Distances Path": cc_path,
                                    "Params Path": cc_params_path if os.path.exists(cc_params_path) else "",
                                    **stats,
                                }
                            )
                        else:
                            logger.warning("[Stats] Spalte '%s' fehlt in: %s", col, cc_path)
                    except Exception as e:
                        logger.error("[Stats] Konnte CC-Datei nicht lesen für %s: %s", fid, e)

        df_result = pd.DataFrame(rows)

        if out_path and not df_result.empty:
            if output_format.lower() == "json":
                _append_df_to_json(df_result, out_path)
            else:
                _append_df_to_excel(df_result, out_path, sheet_name=sheet_name)

        logger.info("Finished compute_m3c2_statistics for %d folders", len(folder_ids))
        return df_result

    @classmethod
    def calc_single_cloud_stats(
        cls,
        folder_ids: List[str],
        filename_mov: str = "mov",
        filename_ref: str = "ref",
        area_m2: Optional[float] = None,
        radius: float = 1.0,
        k: int = 6,
        sample_size: Optional[int] = 100_000,
        use_convex_hull: bool = True,
        out_path: str = "m3c2_stats_clouds.xlsx",
        sheet_name: str = "CloudStats",
        output_format: str = "excel",
    ) -> pd.DataFrame:
        rows: List[Dict] = []

        for fid in folder_ids:
            ds = DataSource(fid, filename_mov, filename_ref)
            mov, ref, _ = ds.load_points()
            for fname, epoch in ((filename_mov, mov), (filename_ref, ref)):
                pts = epoch.cloud if hasattr(epoch, "cloud") else epoch
                stats = _calc_single_cloud_stats(
                    pts,
                    area_m2=area_m2,
                    radius=radius,
                    k=k,
                    sample_size=sample_size,
                    use_convex_hull=use_convex_hull,
                )
                stats.update({"File": fname, "Folder": fid})
                rows.append(stats)

        df_result = pd.DataFrame(rows)
        if out_path and rows:
            write_cloud_stats(
                rows,
                out_path=out_path,
                sheet_name=sheet_name,
                output_format=output_format,
            )
        return df_result

    @staticmethod
    def _load_params(params_path: Optional[str]) -> Tuple[float, float]:
        """Load M3C2 configuration values from a parameter file.

        The M3C2 computation writes a small text file with the parameters
        used during the run.  This helper reads the file and extracts the
        normal and search scales that are stored in lines beginning with
        ``"NormalScale="`` and ``"SearchScale="``.  When no file is
        supplied or a value is missing the respective scale defaults to
        ``numpy.nan``.

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
            with open(params_path, "r") as f:
                for line in f:
                    if line.startswith("NormalScale="):
                        normal_scale = float(line.strip().split("=")[1])
                    elif line.startswith("SearchScale="):
                        search_scale = float(line.strip().split("=")[1])
        return normal_scale, search_scale

    @staticmethod
    def _resolve(fid: str, filename: str) -> str:
        p1 = os.path.join(fid, filename)
        if os.path.exists(p1):
            return p1
        return os.path.join("data", fid, filename)
