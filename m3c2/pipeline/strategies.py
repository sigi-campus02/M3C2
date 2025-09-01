"""Strategies for scanning M3C2 normal and projection scales."""

from __future__ import annotations
import logging
from dataclasses import dataclass
from typing import List, Optional, Protocol, Dict
import numpy as np
from sklearn.neighbors import NearestNeighbors


@dataclass
class ScaleScan:
    """Container for statistics gathered during a scale scan.

    Parameters
    ----------
    scale : float
        The tested normal scale ``D``.
    valid_normals : int
        Number of neighbourhoods with sufficient points to compute a normal.
    mean_population : float
        Average number of neighbours per core point.
    roughness : float
        Mean roughness :math:`\sigma(D)` of the neighbourhoods.
    coverage : float
        Fraction of core points that produced a valid normal.
    mean_lambda3 : float
        Average smallest eigenvalue, used as a planarity measure.
    total_points, std_population, perc97_population, relative_roughness,
    total_voxels : optional
        Additional diagnostic metrics recorded during the scan.
    """

    # 'scale' is D (normal scale, not the neighbourhood radius)
    scale: float
    valid_normals: int
    mean_population: float
    roughness: float              # mean σ(D): standard deviation of orthogonal residuals
    coverage: float
    mean_lambda3: float           # mean λ_min (planarity measure)
    # optional additional metrics
    total_points: Optional[int] = None
    std_population: Optional[float] = None
    perc97_population: Optional[int] = None
    relative_roughness: Optional[float] = None
    total_voxels: Optional[int] = None

# ============================================================
# Radius-based strategy
# ============================================================

def _fit_plane_pca(points: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Derive a best-fit plane for a set of 3‑D points using PCA.

    Parameters
    ----------
    points : (N, 3) ndarray of float
        Neighbourhood points (e.g., a sphere around the core point). ``N`` must
        be at least three.

    Returns
    -------
    centroid : (3,) ndarray
        Centroid of the neighbourhood.
    normal : (3,) ndarray
        Unit normal vector of the fitted plane.
    eigenvalues : (3,) ndarray
        Ascending eigenvalues of the covariance matrix.
    sigma : float
        Orthogonal roughness :math:`\sigma(D)` based on point-to-plane
        distances.
    """
    centroid = points.mean(axis=0)
    centered_points = points - centroid
    # Determine principal directions via covariance eigen-decomposition
    covariance = (centered_points.T @ centered_points) / max(len(points) - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)

    # The smallest eigenvalue corresponds to the normal direction
    normal_vector = eigenvectors[:, 0]
    norm = np.linalg.norm(normal_vector)
    if norm > 0:
        normal_vector = normal_vector / norm

    # Compute distances of points to the plane for the roughness metric
    ortho = centered_points @ normal_vector
    sigma = float(np.std(ortho, ddof=1))
    return centroid, normal_vector, eigenvalues, sigma


class RadiusScanStrategy:
    """Scan a range of scales using radius-based neighbourhoods.

    The strategy evaluates multiple normal scales by constructing radius
    neighbourhoods and fitting planes via PCA.  Collected metrics help
    determine optimal parameters for the M3C2 algorithm.
    """

    def __init__(
        self,
        i_min: int = -3,           # e.g. 2^-3 = 1/8
        i_max: int = 8,            # e.g. 2^8  = 256
        sample_size: Optional[int] = None,
        min_points: int = 10,
        log_each_scale: bool = True,
        signed: bool = False,
        up_dir: np.ndarray | None = None,
    ) -> None:
        """Initialize the scanning strategy.

        Parameters
        ----------
        i_min, i_max : int
            Power range used to derive scanned scales
            ``D = min_spacing * 2**i``.
        sample_size : int, optional
            Randomly subsample the input cloud to this size if provided.
        min_points : int
            Minimum number of neighbours required for a valid normal.
        log_each_scale : bool
            Whether to log statistics for every evaluated scale.
        signed : bool
            If ``True``, orient normals relative to ``up_dir``.
        up_dir : ndarray, optional
            Reference direction used for normal orientation.
        """
        self.i_min = i_min
        self.i_max = i_max
        self.sample_size = sample_size
        self.min_points = min_points
        self.log_each_scale = log_each_scale
        self.signed = signed
        self.up_dir = up_dir

    # 1) Evaluate a single scale (D via neighborhood_radius = D/2)
    def evaluate_radius_scale(self, point_cloud: np.ndarray, neighborhood_radius: float) -> Dict:
        """Evaluate one normal scale using a given neighbourhood radius.

        Parameters
        ----------
        point_cloud : ndarray
            Input point cloud where normals are evaluated.
        neighborhood_radius : float
            Radius used for neighbour searches corresponding to ``D/2``.

        Returns
        -------
        dict
            Collected statistics such as roughness and coverage for the scale.
        """
        if point_cloud.dtype != np.float64:
            point_cloud = point_cloud.astype(np.float64, copy=False)

        # Build a KD-tree for radius-based neighbour searches
        neighbor_search = NearestNeighbors(radius=neighborhood_radius, algorithm="kd_tree")
        neighbor_search.fit(point_cloud)
        neighbor_indices_list = neighbor_search.radius_neighbors(point_cloud, return_distance=False)

        valid_neighbors_count = 0
        sigma_values: list[float] = []
        lambda_min_values: list[float] = []
        population_sizes: list[int] = []

        for neighbor_indices in neighbor_indices_list:
            if neighbor_indices.size < self.min_points:
                continue

            neighbor_points = point_cloud[neighbor_indices]
            _, normal_vec, eigenvalues, sigma = _fit_plane_pca(neighbor_points)

            lambda_min = float(eigenvalues[0])  # smallest eigenvalue (planarity)

            sigma_values.append(float(sigma))
            lambda_min_values.append(lambda_min)
            population_sizes.append(int(neighbor_indices.size))
            valid_neighbors_count += 1

        total_points = int(len(point_cloud))
        mean_population = float(np.mean(population_sizes)) if population_sizes else 0.0
        std_population = float(np.std(population_sizes)) if population_sizes else 0.0
        perc97_population = int(np.percentile(population_sizes, 97)) if population_sizes else 0

        mean_sigma = float(np.mean(sigma_values)) if sigma_values else np.nan
        mean_lambda3 = float(np.mean(lambda_min_values)) if lambda_min_values else np.nan

        scale_D = 2.0 * neighborhood_radius  # D = 2 * (D/2)
        relative_roughness = (mean_sigma / scale_D) if (scale_D > 0 and not np.isnan(mean_sigma)) else np.nan
        coverage = (valid_neighbors_count / total_points) if total_points > 0 else 0.0

        return {
            "scale":               float(scale_D),
            "valid_normals":       int(valid_neighbors_count),
            "total_points":        total_points,
            "mean_population":     mean_population,
            "std_population":      std_population,
            "perc97_population":   perc97_population,
            "roughness":           mean_sigma,          # mean σ(D)
            "mean_lambda3":        mean_lambda3,        # planarity measure
            "relative_roughness":  relative_roughness,  # σ(D)/D
            "coverage":            coverage,
        }

    # 2) Scan multiple scales (D_i = min_spacing * 2^i)
    def scan(self, points: np.ndarray, min_spacing: float) -> List[ScaleScan]:
        """Scan a sequence of scales and collect statistics for each.

        Parameters
        ----------
        points : ndarray
            Point cloud to analyse.
        min_spacing : float
            Minimal point spacing used as the base scale.

        Returns
        -------
        list of :class:`ScaleScan`
            Statistics for each evaluated scale.
        """
        pts = points
        if self.sample_size and len(pts) > self.sample_size:
            # Randomly subsample the cloud to speed up evaluation
            idx = np.random.choice(len(pts), size=self.sample_size, replace=False)
            pts = pts[idx]
            logging.info(f"[RadiusScan] Subsample: {self.sample_size}/{len(points)}")

        scans: List[ScaleScan] = []
        for level in range(self.i_min, self.i_max + 1):
            D = float(min_spacing) * (2.0 ** float(level))
            radius = D / 2.0
            # Evaluate quality metrics for this scale
            res = self.evaluate_radius_scale(pts, radius)

            scans.append(
                ScaleScan(
                    scale=res["scale"],                   # == D
                    valid_normals=res["valid_normals"],
                    mean_population=res["mean_population"],
                    roughness=res["roughness"],           # mean σ(D)
                    coverage=res["coverage"],
                    mean_lambda3=res["mean_lambda3"],     # planarity measure
                    total_points=res["total_points"],
                    std_population=res["std_population"],
                    perc97_population=res["perc97_population"],
                    relative_roughness=res["relative_roughness"],  # σ(D)/D
                )
            )

            if self.log_each_scale:
                logging.info(
                    "[RadiusScan] D=%g | pop=%4.1f±%3.1f | 97%%>%d | valid=%d/%d (%s) | sigma=%g | lambda_min=%g | Sigma/D=%s",
                    D,
                    res["mean_population"],
                    res["std_population"],
                    res["perc97_population"],
                    res["valid_normals"],
                    res["total_points"],
                    f"{res['coverage']:.0%}",
                    res["roughness"],
                    res["mean_lambda3"],
                    "nan" if np.isnan(res["relative_roughness"]) else f"{res['relative_roughness']:.4f}",
                )
        return scans
