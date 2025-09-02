"""Compute descriptive quality metrics for point clouds.

This module provides utilities to summarize characteristics of 3D point
clouds, including global density, height distributions, nearest neighbour
statistics, local geometric descriptors such as linearity and planarity, and
orientation measures like verticality and normal variation.  The metrics can
be used to assess the quality and structure of point cloud data.
"""

from __future__ import annotations

from typing import Dict, List, Optional

import logging

import numpy as np
from joblib import Parallel, delayed
from sklearn.neighbors import NearestNeighbors


logger = logging.getLogger(__name__)


def _bbox_area_xy(xy: np.ndarray) -> float:
    """Return area of axis-aligned bounding box for 2D points.

    Args:
        xy: Array of ``(x, y)`` coordinates with shape ``(N, 2)``.

    Returns:
        The area spanned by the minimal axis-aligned bounding box
        covering the points, in square units.
    """
    x_min, y_min = np.min(xy[:, 0]), np.min(xy[:, 1])
    x_max, y_max = np.max(xy[:, 0]), np.max(xy[:, 1])
    return float((x_max - x_min) * (y_max - y_min))


def _convex_hull_area_xy(xy: np.ndarray) -> float:
    """Return the area of the convex hull defined by XY coordinates.

    Parameters
    ----------
    xy : np.ndarray
        Array of shape ``(n_points, 2)`` with the x and y coordinates of the
        point cloud.

    Returns
    -------
    float
        Area of the convex hull in the XY plane. If ``scipy`` is not available,
        ``NaN`` is returned.
    """

    try:
        from scipy.spatial import ConvexHull
    except ImportError:
        logger.warning("scipy is missing. Returning NaN for convex hull area.")
        return np.nan
    hull = ConvexHull(xy)
    return float(hull.volume)


def _eigenfeatures(neigh: np.ndarray) -> Dict[str, float]:
    """Compute eigen-based local geometric features for a neighbourhood.

    Parameters
    ----------
    neigh : np.ndarray
        Array of shape ``(n_points, 3)`` containing the coordinates of the
        neighbouring points.

    Returns
    -------
    Dict[str, float]
        Dictionary with keys ``roughness``, ``linearity``, ``planarity``,
        ``sphericity``, ``anisotropy``, ``omnivariance``, ``eigenentropy``,
        ``curvature``, ``verticality`` and ``normal``.  Values may be ``NaN``
        when the eigen decomposition is degenerate.
    """

    c = np.mean(neigh, axis=0)
    U = neigh - c
    cov = (U.T @ U) / U.shape[0]
    evals, evecs = np.linalg.eigh(cov)
    order = np.argsort(evals)[::-1]
    evals = evals[order]
    evecs = evecs[:, order]
    n = evecs[:, -1]
    d = np.abs(U @ n)
    roughness = float(np.std(d))

    if evals[0] <= 0:
        return {
            "roughness": roughness,
            "linearity": np.nan,
            "planarity": np.nan,
            "sphericity": np.nan,
            "anisotropy": np.nan,
            "omnivariance": np.nan,
            "eigenentropy": np.nan,
            "curvature": np.nan,
            "verticality": np.nan,
            "normal": n,
        }

    linearity = (evals[0] - evals[1]) / evals[0]
    planarity = (evals[1] - evals[2]) / evals[0]
    sphericity = evals[2] / evals[0]
    anisotropy = (evals[0] - evals[2]) / evals[0]
    omnivariance = float(np.cbrt(np.prod(evals)))
    sum_eval = float(np.sum(evals))
    ratios = evals / sum_eval
    eigenentropy = float(-np.sum(ratios * np.log(ratios + 1e-15)))
    curvature = float(evals[2] / sum_eval)
    verticality = float(
        np.degrees(np.arccos(np.clip(np.abs(n[2]), -1.0, 1.0)))
    )

    return {
        "roughness": roughness,
        "linearity": float(linearity),
        "planarity": float(planarity),
        "sphericity": float(sphericity),
        "anisotropy": float(anisotropy),
        "omnivariance": omnivariance,
        "eigenentropy": eigenentropy,
        "curvature": curvature,
        "verticality": verticality,
        "normal": n,
    }


def calc_single_cloud_stats(
    points: np.ndarray,
    area_m2: Optional[float] = None,
    radius: float = None,
    k: int = 24,
    sample_size: Optional[int] = 100_000,
    use_convex_hull: bool = True,
    min_pts: int = 10,
    max_eval: Optional[int] = 50_000,
) -> Dict:
    """Berechne Qualitätsmetriken für eine Punktwolke."""

    points = np.asarray(points)

    logger.debug(f"points type: {type(points)}, shape: {getattr(points, 'shape', None)}")
    logger.debug(f"points: {points}")
    logger.debug(f"points shape: {points.shape}")
    logger.info(f"Parameters used: area_m2={area_m2}, radius={radius}, k={k}, sample_size={sample_size}, use_convex_hull={use_convex_hull}")

    if points is None or len(points) == 0:
        raise ValueError("Points array is empty")
    P = np.asarray(points, dtype=float)
    if P.ndim != 2 or P.shape[1] != 3:
        raise ValueError("points must be of shape (N, 3)")
    num = len(P)
    logger.info(
        "Calculating cloud stats for %d points (radius=%s, k=%s, sample_size=%s)",
        num,
        radius,
        k,
        sample_size,
    )
    z = P[:, 2]
    z_min, z_max = float(np.min(z)), float(np.max(z))
    z_mean = float(np.mean(z))
    z_median = float(np.median(z))
    z_std = float(np.std(z))
    z_q05, z_q25, z_q75, z_q95 = map(float, np.percentile(z, [5, 25, 75, 95]))

    xy = P[:, :2]
    if area_m2 is None:
        area_bbox = _bbox_area_xy(xy)
        area_hull = _convex_hull_area_xy(xy) if use_convex_hull else np.nan
        area_m2_est = area_hull if use_convex_hull and not np.isnan(area_hull) else area_bbox
        area_used = float(area_m2_est)
        area_src = "convex_hull" if use_convex_hull and not np.isnan(area_hull) else "bbox"
    else:
        area_used = float(area_m2)
        area_src = "given"

    density_global = float(num / area_used) if area_used > 0 else np.nan

    idx = np.arange(num)
    if sample_size and num > sample_size:
        idx = np.random.choice(num, size=sample_size, replace=False)
    S = P[idx]

    nn = NearestNeighbors(n_neighbors=min(k + 1, len(S))).fit(S)
    dists_knn, _ = nn.kneighbors(S)
    mean_nn_all = float(np.mean(dists_knn[:, 1:]))
    mean_nn_kth = float(np.mean(dists_knn[:, min(k, dists_knn.shape[1]-1)]))

    nbrs = NearestNeighbors(radius=radius).fit(S)
    ind_list = nbrs.radius_neighbors(S, return_distance=False)
    vol = 4.0 / 3.0 * np.pi * (radius ** 3)

    counts = np.array([ind.size for ind in ind_list], dtype=int)
    valid_mask = counts >= min_pts
    valid_idx = np.nonzero(valid_mask)[0]

    if max_eval and len(valid_idx) > max_eval:
        valid_idx = np.random.choice(valid_idx, size=max_eval, replace=False)

    logger.info(
        "Evaluating %d neighborhoods (min_pts=%s, max_eval=%s)",
        len(valid_idx),
        min_pts,
        max_eval,
    )

    if len(valid_idx) == 0:
        logger.warning("No valid neighborhoods remain after filtering (min_pts=%s)", min_pts)

    local_dens = (counts[valid_idx] / vol).tolist()
    neighs = [S[ind_list[i]] for i in valid_idx]

    if len(neighs) > 0:
        feats = Parallel(n_jobs=-1)(delayed(_eigenfeatures)(n) for n in neighs)
    else:
        feats = []

    rough = [f["roughness"] for f in feats]
    lin_list = [f["linearity"] for f in feats if not np.isnan(f["linearity"])]
    pla_list = [f["planarity"] for f in feats if not np.isnan(f["planarity"])]
    sph_list = [f["sphericity"] for f in feats if not np.isnan(f["sphericity"])]
    anis_list = [f["anisotropy"] for f in feats if not np.isnan(f["anisotropy"])]
    omni_list = [f["omnivariance"] for f in feats if not np.isnan(f["omnivariance"])]
    eigent_list = [f["eigenentropy"] for f in feats if not np.isnan(f["eigenentropy"])]
    curv_list = [f["curvature"] for f in feats if not np.isnan(f["curvature"])]
    vert_list = [f["verticality"] for f in feats if not np.isnan(f["verticality"])]
    normals = [f["normal"] for f in feats if not np.isnan(f["linearity"])]

    def _agg(arr):
        """Aggregate statistics for a sequence.

        Computes the mean, median, and the 5th and 95th percentiles of
        the given array-like input. When the provided sequence is empty,
        a tuple of NaN values is returned instead.

        Parameters
        ----------
        arr : Sequence[float]
            Values to aggregate.

        Returns
        -------
        Tuple[float, float, float, float]
            Mean, median, 5th percentile, and 95th percentile.
        """
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

    if len(local_dens) == 0 or np.isnan(np.asarray(local_dens, dtype=float)).any():
        logger.warning("Local density array is empty or contains NaN")
    if len(rough) == 0 or np.isnan(np.asarray(rough, dtype=float)).any():
        logger.warning("Roughness array is empty or contains NaN")
    logger.info(
        "Computed metrics: global density=%.3f, roughness_mean=%.3f",
        density_global,
        rough_mean,
    )

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
        "Mean NN Dist All": mean_nn_all,
        "Mean NN Dist k-th": mean_nn_kth,
        "Local Density Mean": dens_mean,
        "Local Density Median": dens_med,
        "Local Density Q05": dens_q05,
        "Local Density Q95": dens_q95,
        "Roughness Mean": rough_mean,
        "Roughness Median": rough_med,
        "Roughness Q05": rough_q05,
        "Roughness Q95": rough_q95,
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


# Backwards compatible alias
_calc_single_cloud_stats = calc_single_cloud_stats
