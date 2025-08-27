# strategies.py
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import List, Optional, Protocol

import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import NearestNeighbors


# ============================================================
# Gemeinsame Dataklasse für Scan-Ergebnisse
# ============================================================

@dataclass
class ScaleScan:
    scale: float
    valid_normals: int
    mean_population: float
    roughness: float
    coverage: float
    # optionale Zusatzmetriken (falls vorhanden)
    total_points: Optional[int] = None
    std_population: Optional[float] = None
    perc97_population: Optional[int] = None
    relative_roughness: Optional[float] = None
    total_voxels: Optional[int] = None


class ScaleStrategy(Protocol):
    def scan(self, points: np.ndarray, avg_spacing: float) -> List[ScaleScan]: ...
    # nur Signatur – keine Implementierung!


# ============================================================
# Shared Helpers
# ============================================================

def _fit_plane_pca(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    PCA-Ebene: centroid & Normale (kleinster Eigenvektor der Kovarianz).
    """
    c = pts.mean(axis=0)
    Q = pts - c
    cov = (Q.T @ Q) / max(len(pts) - 1, 1)
    # eigenvalues aufsteigend, v[:, 0] = Richtung kleinster Varianz
    _, v = np.linalg.eigh(cov)
    n = v[:, 0]
    n = n / np.linalg.norm(n)
    return c, n


def _point_plane_distance(
        p: np.ndarray,
        centroid: np.ndarray,
        normal: np.ndarray,
        signed: bool = False,
        up_dir: np.ndarray | None = None,
) -> float:
    """
    Punkt-zu-Ebene-Abstand; optional vorzeichenstabilisiert bzgl. up_dir.
    """
    n = normal
    if up_dir is not None and np.dot(n, up_dir) < 0:
        n = -n
    d = float(np.dot(p - centroid, n))
    return d if signed else abs(d)


def _voxel_grid_partition(points: np.ndarray, voxel_size: float) -> list[np.ndarray]:
    """
    Partitioniere Punkte in Voxels der Größe voxel_size.
    """
    idx = np.floor(points / voxel_size).astype(int)
    voxels: dict[tuple[int, int, int], list[np.ndarray]] = {}
    for k, p in zip(map(tuple, idx), points):
        voxels.setdefault(k, []).append(p)
    return [np.asarray(v) for v in voxels.values()]


def _plane_roughness(points: np.ndarray) -> float | None:
    """
    RMS der Planar-Residuen (z ≈ f(x,y)) als Roughness.
    """
    if len(points) < 3:
        return None
    reg = LinearRegression()
    reg.fit(points[:, :2], points[:, 2])
    pred = reg.predict(points[:, :2])
    d = points[:, 2] - pred
    return float(np.sqrt(np.mean(d ** 2)))


# ============================================================
# Radius-basierte Strategie
# ============================================================

class RadiusScanStrategy(ScaleStrategy):
    """
    CloudCompare-ähnlicher Radius-Scan:
    - Neighborhood via KD-Tree (radius)
    - Ebene per PCA, Punkt-zu-Ebene-Abstand als Roughness
    - Aggregation über alle Punkte
    """

    def __init__(
            self,
            multipliers: Optional[List[int]] = None,
            sample_size: Optional[int] = None,
            min_points: int = 3,
            signed: bool = False,
            up_dir: np.ndarray | None = None,
    ) -> None:
        # CC-nahe Default-Multiplikatoren (8..64 in 2er-Schritten)
        self.multipliers = multipliers or list(range(8, 65, 2))
        self.sample_size = sample_size
        self.min_points = min_points
        self.signed = signed
        self.up_dir = up_dir

    # 1) Einzel-Skala evaluieren
    def evaluate_radius_scale(self, points: np.ndarray, radius: float) -> dict:
        if points.dtype != np.float64:
            points = points.astype(np.float64, copy=False)

        nn = NearestNeighbors(radius=radius, algorithm="kd_tree")
        nn.fit(points)
        indices_list = nn.radius_neighbors(points, return_distance=False)

        valid_count = 0
        roughness_vals: list[float] = []
        populations: list[int] = []

        for i, idx in enumerate(indices_list):
            if idx.size < self.min_points:
                continue
            neigh = points[idx]
            c, n = _fit_plane_pca(neigh)
            r = _point_plane_distance(points[i], c, n, signed=self.signed, up_dir=self.up_dir)
            roughness_vals.append(r)
            populations.append(idx.size)
            valid_count += 1

        total = len(points)
        mean_pop = float(np.mean(populations)) if populations else 0.0
        std_pop = float(np.std(populations)) if populations else 0.0
        mean_rough = float(np.mean(roughness_vals)) if roughness_vals else np.nan
        rel_rough = mean_rough / radius if radius > 0 else np.nan
        coverage = valid_count / total if total > 0 else 0.0
        perc97 = int(np.percentile(populations, 97)) if populations else 0

        return {
            "scale": radius,
            "valid_normals": valid_count,
            "total_points": total,
            "mean_population": mean_pop,
            "std_population": std_pop,
            "perc97_population": perc97,
            "roughness": mean_rough,
            "relative_roughness": rel_rough,
            "coverage": coverage,
        }

    # 2) Mehrere Skalen scannen (ParamEstimator ruft diese Methode)
    def scan(self, points: np.ndarray, avg_spacing: float) -> List[ScaleScan]:
        pts = points
        if self.sample_size and len(pts) > self.sample_size:
            idx = np.random.choice(len(pts), size=self.sample_size, replace=False)
            pts = pts[idx]
            logging.info(f"[RadiusScan] Subsample: {self.sample_size}/{len(points)}")

        scans: List[ScaleScan] = []
        for m in self.multipliers:
            r = float(m) * float(avg_spacing)
            res = self.evaluate_radius_scale(pts, r)
            scans.append(
                ScaleScan(
                    scale=res["scale"],
                    valid_normals=res["valid_normals"],
                    mean_population=res["mean_population"],
                    roughness=res["roughness"],
                    coverage=res["coverage"],
                    total_points=res["total_points"],
                    std_population=res["std_population"],
                    perc97_population=res["perc97_population"],
                    relative_roughness=res["relative_roughness"],
                )
            )
            logging.info(
                f"[RadiusScan] scale={r:.6g} | pop={res['mean_population']:.1f}±{res['std_population']:.1f} | "
                f"97%>{res['perc97_population']} | valid={res['valid_normals']}/{res['total_points']} "
                f"({res['coverage']:.0%}) | rel_rough={res['relative_roughness']:.6f}"
            )
        return scans

    # 3) Optional: „Kompatibilitäts“-Alias zu deiner alten Signatur
    def octree_guess_params(
            self,
            points: np.ndarray,
            avg_spacing: float,
            multipliers: Optional[List[int]] = None,
            sample_size: Optional[int] = None,
    ) -> List[dict]:
        if multipliers is not None:
            self.multipliers = multipliers
        if sample_size is not None:
            self.sample_size = sample_size
        scans = self.scan(points, avg_spacing)
        # für Rückwärtskompatibilität als dict-Liste
        return [s.__dict__ for s in scans]


# ============================================================
# Voxel-basierte Strategie
# ============================================================

class VoxelScanStrategy(ScaleStrategy):
    """
    Voxel-Scan:
    - Partition in Voxelzellen (Grid)
    - Pro Voxel: Roughness via Planarfit (RMS), Normale via PCA (Validitätskriterium)
    - Aggregation über Zellen
    """

    def __init__(
            self,
            steps: int = 10,
            start_pow: int = 5,
            sample_size: Optional[int] = None,
            min_points: int = 6,
    ) -> None:
        self.steps = steps
        self.start_pow = start_pow
        self.sample_size = sample_size
        self.min_points = min_points

    # 1) Einzel-Skala evaluieren
    def evaluate_voxel_scale(self, points: np.ndarray, voxel_size: float) -> dict:
        voxels = _voxel_grid_partition(points, voxel_size)

        valid_normals = 0
        roughness_list: list[float] = []
        populations: list[int] = []

        for cell in voxels:
            if len(cell) < self.min_points:
                continue
            populations.append(len(cell))

            # PCA nur um „Normale könnte geschätzt werden“ anzudeuten
            _ = PCA(n_components=3).fit(cell)

            r = _plane_roughness(cell)
            if r is not None:
                roughness_list.append(r)
            valid_normals += 1

        return {
            "scale": voxel_size,
            "total_voxels": len(voxels),
            "valid_normals": valid_normals,
            "mean_population": float(np.mean(populations)) if populations else 0.0,
            "roughness": float(np.mean(roughness_list)) if roughness_list else 0.0,
            "coverage": valid_normals / len(voxels) if voxels else 0.0,
        }

    # 2) Mehrere Skalen scannen (ParamEstimator ruft diese Methode)
    def scan(self, points: np.ndarray, avg_spacing: float) -> List[ScaleScan]:
        pts = points
        if self.sample_size and len(pts) > self.sample_size:
            idx = np.random.choice(len(pts), size=self.sample_size, replace=False)
            pts = pts[idx]
            logging.info(f"[VoxelScan] Subsample: {self.sample_size}/{len(points)}")

        scales = [2.0 ** i * avg_spacing for i in range(self.start_pow, self.start_pow + self.steps)]
        scans: List[ScaleScan] = []
        for s in scales:
            res = self.evaluate_voxel_scale(pts, s)
            scans.append(
                ScaleScan(
                    scale=res["scale"],
                    valid_normals=res["valid_normals"],
                    mean_population=res["mean_population"],
                    roughness=res["roughness"],
                    coverage=res["coverage"],
                    total_voxels=res["total_voxels"],
                )
            )
            logging.info(
                f"[VoxelScan] scale={s:.6g} | pop={res['mean_population']:.1f} | "
                f"valid_normals={res['valid_normals']} | total_voxels={res['total_voxels']} | "
                f"coverage={res['coverage']:.0%} | roughness={res['roughness']:.6f}"
            )
        return scans

    # 3) Optional: „Kompatibilitäts“-Alias zu deiner alten Signatur
    def octree_guess_params(self, points: np.ndarray, avg_spacing: float, steps: int = 10) -> List[dict]:
        self.steps = steps
        scans = self.scan(points, avg_spacing)
        return [s.__dict__ for s in scans]


