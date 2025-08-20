# statistics_service.py
from __future__ import annotations

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import norm, weibull_min
from datetime import datetime

from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors


class StatisticsService:
    # =============================
    # Public API
    # =============================

    @staticmethod
    def calc_stats(
        distances: np.ndarray,
        params_path: Optional[str] = None,
        bins: int = 256,
        range_override: Optional[Tuple[float, float]] = None,
        min_expected: Optional[float] = None,
        tolerance: float = 0.01,
    ) -> Dict:
        """Berechne diverse Metriken aus den gegebenen Distanzwerten.

        Parameters
        ----------
        distances:
            Array mit Distanzwerten, das NaN enthalten kann.
        params_path:
            Optionaler Pfad zu einer Datei mit ``NormalScale`` und ``SearchScale`` Parametern.
        bins:
            Anzahl der Bins für das Histogramm.
        range_override:
            Optionales Tupel zur expliziten Festlegung des Wertebereichs.
        min_expected:
            Mindestanzahl erwarteter Werte pro Bin für die Chi²-Berechnung.
        tolerance:
            Toleranzschwelle zur Bewertung der Distanzwerte.
        """

        total_count = len(distances)
        nan_count = int(np.isnan(distances).sum())
        valid = distances[~np.isnan(distances)]
        if valid.size == 0:
            raise ValueError("No valid distances")

        # Range wie in CC
        if range_override is None:
            data_min, data_max = float(np.min(valid)), float(np.max(valid))
        else:
            data_min, data_max = map(float, range_override)

        # Clip (wie CC) für Histogramm + Fits
        clipped = valid[(valid >= data_min) & (valid <= data_max)]
        if clipped.size == 0:
            raise ValueError("All values fall outside the selected range")

        # Summen/Momente (CLIPPED)
        valid_sum = float(np.sum(clipped))
        valid_squared_sum = float(np.sum(clipped ** 2))
        avg = float(np.mean(clipped))
        rms = float(np.sqrt(np.mean(clipped ** 2)))

        # NEW: MAE (gegen 0) & NMAD (robuste Sigma)
        mae = float(np.mean(np.abs(clipped)))  # Mean Absolute Error

        med = float(np.median(clipped))
        mad = float(np.median(np.abs(clipped - med)))
        nmad = float(1.4826 * mad)  # Normalized MAD

        # Histogramm
        hist, bin_edges = np.histogram(clipped, bins=bins, range=(data_min, data_max))
        hist = hist.astype(float)

        # Fits (Gauss & Weibull)
        fit_results = StatisticsService._fit_distributions(
            clipped, hist, bin_edges, min_expected
        )
        mu = fit_results["mu"]
        std = fit_results["std"]
        pearson_gauss = fit_results["pearson_gauss"]
        a = fit_results["a"]
        b = fit_results["b"]
        loc = fit_results["loc"]
        pearson_weib = fit_results["pearson_weib"]
        skew_weibull = fit_results["skew_weibull"]
        mode_weibull = fit_results["mode_weibull"]

        # Optional: CC-/Params-Datei
        normal_scale, search_scale = StatisticsService._load_params(params_path)

        # Outliers
        outlier_info = StatisticsService._compute_outliers(clipped, rms, med)
        outlier_count = outlier_info["outlier_count"]
        inlier_count = outlier_info["inlier_count"]
        mean_in = outlier_info["mean_in"]
        mean_out = outlier_info["mean_out"]
        std_in = outlier_info["std_in"]
        std_out = outlier_info["std_out"]
        pos_out = outlier_info["pos_out"]
        neg_out = outlier_info["neg_out"]
        pos_in = outlier_info["pos_in"]
        neg_in = outlier_info["neg_in"]
        mae_in = outlier_info["mae_in"]
        nmad_in = outlier_info["nmad_in"]

        # Bias & Toleranz
        bias = float(np.mean(clipped))
        within_tolerance = float(np.mean(np.abs(clipped) <= tolerance))

        # ICC/CCC/Bland-Altman
        icc = np.nan  # Placeholder
        mean_dist = float(np.mean(clipped))
        std_dist = float(np.std(clipped))
        ccc = (2 * mean_dist * std_dist) / (mean_dist**2 + std_dist**2) if mean_dist != 0 else np.nan

        bland_altman_lower = bias - 1.96 * std_dist
        bland_altman_upper = bias + 1.96 * std_dist

        # Overlap (Jaccard/Dice)
        intersection = np.sum((clipped > -tolerance) & (clipped < tolerance))
        union = len(clipped)
        jaccard_index = intersection / union if union > 0 else np.nan
        dice_coefficient = (2 * intersection) / (2 * union) if union > 0 else np.nan

        return {
            # 1) Counts & Scales
            "Gesamt": total_count,
            "NaN": nan_count,
            "% NaN": (nan_count / total_count) if total_count > 0 else np.nan,
            "% Valid": (1 - nan_count / total_count) if total_count > 0 else np.nan,
            "Valid Count": int(clipped.size),
            "Valid Sum": valid_sum,
            "Valid Squared Sum": valid_squared_sum,
            "Normal Scale": normal_scale,
            "Search Scale": search_scale,

            # 2) Lage & Streuung
            "Min": float(np.nanmin(distances)),
            "Max": float(np.nanmax(distances)),
            "Mean": avg,
            "Median": float(np.median(clipped)),
            "RMS": rms,
            "Std": float(np.std(clipped)),
            "MAE": mae,
            "MAE Inlier": mae_in,
            "NMAD": nmad,
            "NMAD Inlier": nmad_in,

            # 3) Outlier / Inlier
            "Outlier Count": outlier_count,
            "Inlier Count": inlier_count,
            "Mean Inlier": mean_in,
            "Std Inlier": std_in,
            "Mean Outlier": mean_out,
            "Std Outlier": std_out,
            "Pos Outlier": pos_out,
            "Neg Outlier": neg_out,
            "Pos Inlier": pos_in,
            "Neg Inlier": neg_in,

            # 4) Quantile
            "Q05": float(np.percentile(clipped, 5)),
            "Q25": float(np.percentile(clipped, 25)),
            "Q75": float(np.percentile(clipped, 75)),
            "Q95": float(np.percentile(clipped, 95)),

            # 5) Fit-Metriken
            "Gauss Mean": float(mu),
            "Gauss Std": float(std),
            "Gauss Chi2": float(pearson_gauss),
            "Weibull a": float(a),
            "Weibull b": float(b),
            "Weibull shift": float(loc),
            "Weibull mode": mode_weibull,
            "Weibull skewness": skew_weibull,
            "Weibull Chi2": float(pearson_weib),

            # 6) Weitere Kennzahlen
            "Skewness": float(pd.Series(clipped).skew()),
            "Kurtosis": float(pd.Series(clipped).kurt()),
            "Anteil |Distanz| > 0.01": float(np.mean(np.abs(clipped) > 0.01)),
            "Anteil [-2Std,2Std]": float(np.mean((clipped > -2*std) & (clipped < 2*std))),
            "Max |Distanz|": float(np.max(np.abs(clipped))),
            "Bias": bias,
            "Within-Tolerance": within_tolerance,
            "ICC": icc,
            "CCC": ccc,
            "Bland-Altman Lower": bland_altman_lower,
            "Bland-Altman Upper": bland_altman_upper,
            "Jaccard Index": jaccard_index,
            "Dice Coefficient": dice_coefficient
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
        out_xlsx: str = "m3c2_stats_all.xlsx",
        sheet_name: str = "Results",
    ) -> pd.DataFrame:
        """
        Liest je Folder {version}_m3c2_distances.txt (Python) und optional CloudCompare
        und hängt die Ergebnisse an eine Excel-Datei an (mit Timestamp).
        Spalten: Folder | Version | Typ | ... (Metriken)
        """
        rows: List[Dict] = []

        for fid in folder_ids:
            # ----- Python
            if process_python_CC == "python":
                py_dist_path   = cls._resolve(fid, f"python_{filename_ref}_m3c2_distances.txt")
                py_params_path = cls._resolve(fid, f"python_{filename_ref}_m3c2_params.txt")
                if os.path.exists(py_dist_path):
                    values = np.loadtxt(py_dist_path)
                    stats = cls.calc_stats(
                        values,
                        params_path=py_params_path if os.path.exists(py_params_path) else None,
                        bins=bins,
                        range_override=range_override,
                        min_expected=min_expected,
                    )
                    rows.append({
                        "Folder": fid,
                        "Version": filename_ref or "",
                        "Typ": process_python_CC,
                        "Distances Path": py_dist_path,
                        "Params Path": py_params_path if os.path.exists(py_params_path) else "",
                        **stats
                    })

            # ----- CloudCompare
            if process_python_CC == "CC":
                cc_path        = cls._resolve(fid, f"CC_{filename_ref}_m3c2_distances.txt")
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
                            )
                            rows.append({
                                "Folder": fid,
                                "Version": filename_ref or "",
                                "Typ": process_python_CC,
                                "Distances Path": cc_path,
                                "Params Path": cc_params_path if os.path.exists(cc_params_path) else "",
                                **stats
                            })
                        else:
                            print(f"[Stats] Spalte '{col}' fehlt in: {cc_path}")
                    except Exception as e:
                        print(f"[Stats] Konnte CC-Datei nicht lesen für {fid}: {e}")

        df_result = pd.DataFrame(rows)

        # Excel-Append (mit Timestamp)
        if out_xlsx and not df_result.empty:
            cls._append_df_to_excel(df_result, out_xlsx, sheet_name=sheet_name)

        return df_result


    @staticmethod
    def write_table(rows: List[Dict], out_path: str = "m3c2_stats_all.xlsx", sheet_name: str = "Results") -> None:
        df = pd.DataFrame(rows)
        if df.empty:
            return
        StatisticsService._append_df_to_excel(df, out_path, sheet_name=sheet_name)



    # =============================
    # Helpers
    # =============================

    @staticmethod
    def _fit_distributions(
        clipped: np.ndarray,
        hist: np.ndarray,
        bin_edges: np.ndarray,
        min_expected: Optional[float],
    ) -> Dict[str, float]:
        """Fit Gaussian and Weibull distributions and compute Chi² metrics.

        Parameters
        ----------
        clipped:
            Werte, die innerhalb des gewählten Bereichs liegen.
        hist, bin_edges:
            Histogramm der ``clipped`` Werte und die zugehörigen Grenzen.
        min_expected:
            Mindestanzahl erwarteter Werte pro Bin für die Chi²-Berechnung.

        Returns
        -------
        dict
            Kennzahlen der Fits, inklusive Parameter und Chi²-Werten.
        """

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

    @staticmethod
    def _compute_outliers(
        clipped: np.ndarray, rms: float, median: float
    ) -> Dict[str, float]:
        """Bestimme Outlier- und Inlier-Kennzahlen.

        Die Abgrenzung erfolgt über ``3 * rms``. Zusätzlich werden Kennwerte
        für Inlier und Outlier berechnet.
        """

        outlier_mask = np.abs(clipped) > (3 * rms)
        inliers = clipped[~outlier_mask]
        outliers = clipped[outlier_mask]

        mean_out = float(np.mean(outliers)) if outliers.size else np.nan
        mean_in = float(np.mean(inliers)) if inliers.size else np.nan

        std_in = float(np.std(inliers)) if inliers.size > 0 else np.nan
        std_out = float(np.std(outliers)) if outliers.size > 0 else np.nan

        pos_out = int(np.sum(outliers > 0))
        neg_out = int(np.sum(outliers < 0))
        pos_in = int(np.sum(inliers > 0))
        neg_in = int(np.sum(inliers < 0))

        mae_in = float(np.mean(np.abs(inliers))) if inliers.size > 0 else np.nan
        nmad_in = (
            float(1.4826 * np.median(np.abs(inliers - median)))
            if inliers.size > 0
            else np.nan
        )

        return {
            "outlier_count": int(outlier_mask.sum()),
            "inlier_count": int((~outlier_mask).sum()),
            "mean_out": mean_out,
            "mean_in": mean_in,
            "std_in": std_in,
            "std_out": std_out,
            "pos_out": pos_out,
            "neg_out": neg_out,
            "pos_in": pos_in,
            "neg_in": neg_in,
            "mae_in": mae_in,
            "nmad_in": nmad_in,
        }

    @staticmethod
    def _load_params(params_path: Optional[str]) -> Tuple[float, float]:
        """Lese ``NormalScale`` und ``SearchScale`` aus einer Parameterdatei.

        Returns ``(np.nan, np.nan)`` falls die Datei nicht existiert oder die
        Werte fehlen.
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

    @staticmethod
    def _now_timestamp() -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


    @staticmethod
    def _append_df_to_excel(df_new: pd.DataFrame, out_xlsx: str, sheet_name: str = "Results") -> None:
        """
        Hängt df_new an eine bestehende Excel-Datei an (oder erzeugt sie),
        sorgt für 'Timestamp' als erste Spalte und harmonisiert Spalten.
        """

            # Reihenfolge
        CANONICAL_COLUMNS = [
            "Timestamp", "Folder", "Version", "Typ",
            "Gesamt", "NaN", "% NaN", "% Valid",
            "Valid Count", "Valid Sum", "Valid Squared Sum",
            "Normal Scale", "Search Scale",
            "Min", "Max", "Mean", "Median", "RMS", "Std", "MAE", "MAE Inlier", "NMAD", "NMAD Inlier",
            "Outlier Count", "Inlier Count",
            "Mean Inlier", "Std Inlier", "Mean Outlier", "Std Outlier",
            "Pos Outlier", "Neg Outlier", "Pos Inlier", "Neg Inlier",
            "Q05", "Q25", "Q75", "Q95",
            "Gauss Mean", "Gauss Std", "Gauss Chi2",
            "Weibull a", "Weibull b", "Weibull shift", "Weibull mode",
            "Weibull skewness", "Weibull Chi2",
            "Skewness", "Kurtosis",
            "Anteil |Distanz| > 0.01", "Anteil [-2Std,2Std]",
            "Max |Distanz|", "Bias", "Within-Tolerance",
            "ICC", "CCC", "Bland-Altman Lower", "Bland-Altman Upper", "Jaccard Index",
            "Dice Coefficient", "Distances Path", "Params Path",
        ]

        if df_new is None or df_new.empty:
            return

        # Timestamp-Spalte einfügen (erste Spalte)
        ts = StatisticsService._now_timestamp()
        df_new = df_new.copy()
        df_new.insert(0, "Timestamp", ts)

        # Falls Datei existiert: einlesen, Spaltenunion bilden, zusammenführen
        if os.path.exists(out_xlsx):
            try:
                df_old = pd.read_excel(out_xlsx, sheet_name=sheet_name)
            except Exception:
                # Falls Sheet fehlt oder Datei leer/korrupt ist: so behandeln, als gäbe es nichts
                df_old = pd.DataFrame(columns=["Timestamp"])

            # Spaltenreihenfolge beibehalten: Timestamp + bestehende + neue (ohne Duplikate)
            cols = list(df_old.columns) if not df_old.empty else ["Timestamp"]
            if "Timestamp" not in cols:
                cols.insert(0, "Timestamp")

            for c in df_new.columns:
                if c not in cols:
                    cols.append(c)

            df_old = df_old.reindex(columns=cols)
            df_new = df_new.reindex(columns=cols)

            df_all = pd.concat([df_old, df_new], ignore_index=True)
        else:
            df_all = df_new

        # Passt Reihenfolge in Excel an
        for c in CANONICAL_COLUMNS:
            if c not in df_all.columns:
                df_all[c] = np.nan
        df_all = df_all.reindex(columns=CANONICAL_COLUMNS)

        # Schreiben (überschreibt Datei; Inhalt ist df_all)
        try:
            with pd.ExcelWriter(out_xlsx, engine="openpyxl", mode="w") as writer:
                df_all.to_excel(writer, index=False, sheet_name=sheet_name)
        except ModuleNotFoundError as e:
            raise ModuleNotFoundError(
                "Zum Schreiben nach Excel wird 'openpyxl' benötigt. Bitte installieren: pip install openpyxl"
            ) from e


# --------------------------------------------- #
# SINGLE-CLOUD STATISTIKEN (ohne Distanzwerte)
# --------------------------------------------- #

    @staticmethod
    def calc_single_cloud_stats(
        points: np.ndarray,
        area_m2: Optional[float] = None,
        radius: float = 1.0,
        k: int = 6,
        sample_size: Optional[int] = 100_000,
        use_convex_hull: bool = True,
    ) -> Dict:
        """
        Basis-Qualitätsmetriken für EINE Punktwolke (XYZ in Metern).
        """
        if points is None or len(points) == 0:
            raise ValueError("Points array is empty")
        P = np.asarray(points, dtype=float)
        if P.ndim != 2 or P.shape[1] != 3:
            raise ValueError("points must be of shape (N, 3)")

        # Globale Z-Statistik
        z = P[:, 2]
        num = len(P)
        z_min, z_max = float(np.min(z)), float(np.max(z))
        z_mean = float(np.mean(z))
        z_median = float(np.median(z))
        z_std = float(np.std(z))
        z_q05, z_q25, z_q75, z_q95 = map(float, np.percentile(z, [5, 25, 75, 95]))

        # XY-Fläche
        xy = P[:, :2]
        if area_m2 is None:
            area_bbox = StatisticsService._bbox_area_xy(xy)
            area_hull = StatisticsService._convex_hull_area_xy(xy) if use_convex_hull else np.nan
            area_m2_est = area_hull if use_convex_hull and not np.isnan(area_hull) else area_bbox
            area_used = float(area_m2_est)
            area_src = "convex_hull" if use_convex_hull and not np.isnan(area_hull) else "bbox"
        else:
            area_used = float(area_m2)
            area_src = "given"

        # Globale Dichte
        density_global = float(num / area_used) if area_used > 0 else np.nan

        # Subsample für lokale Metriken
        idx = np.arange(num)
        if sample_size and num > sample_size:
            idx = np.random.choice(num, size=sample_size, replace=False)
        S = P[idx]

        # kNN-Abstände
        nn = NearestNeighbors(n_neighbors=min(k + 1, len(S))).fit(S)
        dists_knn, _ = nn.kneighbors(S)
        mean_nn_all = float(np.mean(dists_knn[:, 1:]))
        mean_nn_kth = float(np.mean(dists_knn[:, min(k, dists_knn.shape[1]-1)]))

        # Radius-Nachbarschaften
        nbrs = NearestNeighbors(radius=radius).fit(S)
        ind_list = nbrs.radius_neighbors(S, return_distance=False)
        vol = 4.0 / 3.0 * np.pi * (radius ** 3)
        local_dens: List[float] = []
        rough: List[float] = []
        lin_list: List[float] = []
        pla_list: List[float] = []
        sph_list: List[float] = []
        anis_list: List[float] = []
        omni_list: List[float] = []
        eigent_list: List[float] = []
        curv_list: List[float] = []
        vert_list: List[float] = []
        normals: List[np.ndarray] = []

        for ind in ind_list:
            if ind.size < 3:
                continue
            neigh = S[ind]
            local_dens.append(ind.size / vol)
            c = np.mean(neigh, axis=0)
            U = neigh - c
            pca = PCA(n_components=3).fit(U)
            n = pca.components_[-1]
            d = np.abs(U @ n)
            rough.append(float(np.std(d)))
            evals = np.sort(pca.explained_variance_)[::-1]
            if evals[0] <= 0:
                continue
            linearity = (evals[0] - evals[1]) / evals[0]
            planarity = (evals[1] - evals[2]) / evals[0]
            sphericity = evals[2] / evals[0]
            anisotropy = (evals[0] - evals[2]) / evals[0]
            omnivariance = float(np.cbrt(np.prod(evals)))
            sum_eval = float(np.sum(evals))
            if sum_eval > 0:
                ratios = evals / sum_eval
                eigenentropy = float(-np.sum(ratios * np.log(ratios + 1e-15)))
                curvature = float(evals[2] / sum_eval)
            else:
                eigenentropy = np.nan
                curvature = np.nan
            verticality = float(
                np.degrees(np.arccos(np.clip(np.abs(n[2]), -1.0, 1.0)))
            )
            lin_list.append(float(linearity))
            pla_list.append(float(planarity))
            sph_list.append(float(sphericity))
            anis_list.append(float(anisotropy))
            omni_list.append(omnivariance)
            eigent_list.append(eigenentropy)
            curv_list.append(curvature)
            vert_list.append(verticality)
            normals.append(n)

        def _agg(arr):
            if len(arr) == 0:
                return (np.nan, np.nan, np.nan, np.nan)
            a = np.asarray(arr, dtype=float)
            return (
                float(np.mean(a)),
                float(np.median(a)),
                float(np.percentile(a, 5)),
                float(np.percentile(a, 95)),
            )

        dens_mean, dens_med, dens_q05, dens_q95 = _agg(local_dens)
        rough_mean, rough_med, rough_q05, rough_q95 = _agg(rough)
        lin_mean, lin_med, _, _ = _agg(lin_list)
        pla_mean, pla_med, _, _ = _agg(pla_list)
        sph_mean, sph_med, _, _ = _agg(sph_list)
        anis_mean, anis_med, _, _ = _agg(anis_list)
        omni_mean, omni_med, _, _ = _agg(omni_list)
        eig_mean, eig_med, _, _ = _agg(eigent_list)
        curv_mean, curv_med, _, _ = _agg(curv_list)
        vert_mean, vert_med, vert_q05, vert_q95 = _agg(vert_list)

        # Normalenkonsistenz
        normal_std_deg = np.nan
        if len(normals) > 3:
            N = np.asarray(normals)
            mean_n = np.mean(N, axis=0)
            if np.linalg.norm(mean_n) > 0:
                mean_n = mean_n / np.linalg.norm(mean_n)
            for i in range(N.shape[0]):
                if np.dot(N[i], mean_n) < 0:
                    N[i] = -N[i]
            cosang = np.clip(N @ mean_n, -1.0, 1.0)
            ang = np.degrees(np.arccos(cosang))
            normal_std_deg = float(np.std(ang))

        return {
            "Num Points": num,
            "Area Source": area_src,
            "Area XY [m^2]": area_used,
            "Density Global [pt/m^2]": density_global,
            "Z Min": z_min,
            "Z Max": z_max,
            "Z Mean": z_mean,
            "Z Median": z_median,
            "Z Std": z_std,
            "Z Q05": z_q05,
            "Z Q25": z_q25,
            "Z Q75": z_q75,
            "Z Q95": z_q95,
            f"Mean NN Dist (1..{k})": mean_nn_all,
            f"Mean Dist to {k}-NN": mean_nn_kth,
            "Local Density Mean [pt/m^3]": dens_mean,
            "Local Density Median [pt/m^3]": dens_med,
            "Local Density Q05 [pt/m^3]": dens_q05,
            "Local Density Q95 [pt/m^3]": dens_q95,
            "Roughness Mean [m]": rough_mean,
            "Roughness Median [m]": rough_med,
            "Roughness Q05 [m]": rough_q05,
            "Roughness Q95 [m]": rough_q95,
            "Linearity Mean": lin_mean,
            "Linearity Median": lin_med,
            "Planarity Mean": pla_mean,
            "Planarity Median": pla_med,
            "Sphericity Mean": sph_mean,
            "Sphericity Median": sph_med,
            "Anisotropy Mean": anis_mean,
            "Anisotropy Median": anis_med,
            "Omnivariance Mean": omni_mean,
            "Omnivariance Median": omni_med,
            "Eigenentropy Mean": eig_mean,
            "Eigenentropy Median": eig_med,
            "Curvature Mean": curv_mean,
            "Curvature Median": curv_med,
            "Verticality Mean [deg]": vert_mean,
            "Verticality Median [deg]": vert_med,
            "Verticality Q05 [deg]": vert_q05,
            "Verticality Q95 [deg]": vert_q95,
            "Normal Std Angle [deg]": normal_std_deg,
            "Radius [m]": float(radius),
            "k-NN": int(k),
            "Sampled Points": int(len(S)),
        }

    @staticmethod
    def _bbox_area_xy(xy: np.ndarray) -> float:
        x_min, y_min = np.min(xy[:, 0]), np.min(xy[:, 1])
        x_max, y_max = np.max(xy[:, 0]), np.max(xy[:, 1])
        return float((x_max - x_min) * (y_max - y_min))

    @staticmethod
    def _convex_hull_area_xy(xy: np.ndarray) -> float:
        try:
            from scipy.spatial import ConvexHull
        except Exception:
            return np.nan
        hull = ConvexHull(xy)
        return float(hull.volume)

    @staticmethod
    def write_cloud_stats(rows: List[Dict], out_xlsx: str = "m3c2_stats_clouds.xlsx", sheet_name: str = "CloudStats") -> None:
        df = pd.DataFrame(rows)
        if df.empty:
            return
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df.insert(0, "Timestamp", ts)
        if os.path.exists(out_xlsx):
            try:
                old = pd.read_excel(out_xlsx, sheet_name=sheet_name)
            except Exception:
                old = pd.DataFrame(columns=["Timestamp"])
            cols = list(old.columns) if not old.empty else ["Timestamp"]
            for c in df.columns:
                if c not in cols:
                    cols.append(c)
            old = old.reindex(columns=cols)
            df = df.reindex(columns=cols)
            all_df = pd.concat([old, df], ignore_index=True)
        else:
            all_df = df
        with pd.ExcelWriter(out_xlsx, engine="openpyxl", mode="w") as w:
            all_df.to_excel(w, index=False, sheet_name=sheet_name)
