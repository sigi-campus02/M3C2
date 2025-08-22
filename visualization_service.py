# visualization_service.py
from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

# seaborn optional
try:
    import seaborn as sns  # type: ignore
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False

# plyfile optional
try:
    from plyfile import PlyData, PlyElement  # type: ignore
except Exception:
    PlyData = None
    PlyElement = None


class VisualizationService:
    """
    Visualisierungs-Utilities:
      - histogram(distances, path, bins)
      - colorize(points, distances, outply) -> colors (Nx3, uint8)
      - export_valid(points, colors, distances, outply)
    """

    # ---------- Diagramme ----------

    @staticmethod
    def histogram(
        distances: np.ndarray,
        path: Optional[str] = None,
        bins: int = 256,
        title: str = "Verteilung der M3C2-Distanzen",
    ) -> None:
        """Speichert (oder zeigt) ein Histogramm der gültigen Distanzen."""
        vals = distances[~np.isnan(distances)]
        plt.figure(figsize=(10, 6))
        if _HAS_SNS:
            sns.histplot(vals, bins=bins, kde=False)
        else:
            plt.hist(vals, bins=bins)
        plt.title(title)
        plt.xlabel("M3C2-Distanz")
        plt.ylabel("Anzahl Punkte")
        plt.grid(True)
        plt.tight_layout()
        if path:
            d = os.path.dirname(path)
            if d:
                os.makedirs(d, exist_ok=True)
            plt.savefig(path)
        plt.close()

    # ---------- PLY-Exports ----------

    @staticmethod
    def colorize(
        points: np.ndarray,
        distances: np.ndarray,
        outply: str,
        nan_color: Tuple[int, int, int] = (255, 255, 255),
        percentile_range: Tuple[float, float] = (0.0, 100.0),
    ) -> np.ndarray:
        """
        Punktwolke anhand Distanz einfärben und als PLY speichern.
        - NaN-Distanzen: nan_color
        - percentile_range: z.B. (1, 99) für robustes Clipping
        Rückgabe: colors (uint8, Nx3)
        """
        if PlyData is None or PlyElement is None:
            raise RuntimeError("PLY-Export nicht verfügbar (pip install plyfile).")

        n = len(distances)
        colors = np.zeros((n, 3), dtype=np.uint8)

        valid_mask = ~np.isnan(distances)
        if valid_mask.any():
            v = distances[valid_mask]
            p_lo, p_hi = percentile_range
            vmin = float(np.percentile(v, p_lo))
            vmax = float(np.percentile(v, p_hi))
            if vmax <= vmin:
                vmax = vmin + 1e-12
            normed = (np.clip(v, vmin, vmax) - vmin) / (vmax - vmin)

            # CC-ähnliche Farbskala: blau → grün → gelb → rot
            cc_colors = [(0.0, "blue"), (0.33, "green"), (0.66, "yellow"), (1.0, "red")]
            cc_cmap = LinearSegmentedColormap.from_list("CC_Colormap", cc_colors)

            colored_valid = (cc_cmap(normed)[:, :3] * 255).astype(np.uint8)
            colors[valid_mask] = colored_valid

        # NaNs als weiß (oder nan_color)
        colors[~valid_mask] = np.array(nan_color, dtype=np.uint8)

        # -> PLY schreiben
        vertex = np.array(
            [(x, y, z, r, g, b) for (x, y, z), (r, g, b) in zip(points, colors)],
            dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                   ("red", "u1"), ("green", "u1"), ("blue", "u1")],
        )
        el = PlyElement.describe(vertex, "vertex")

        d = os.path.dirname(outply)
        if d:
            os.makedirs(d, exist_ok=True)   
        PlyData([el], text=False).write(outply)
        return colors

    @staticmethod
    def export_valid(
        points: np.ndarray,
        colors: np.ndarray,
        distances: np.ndarray,
        outply: str,
    ) -> None:
        """Nur gültige Punkte (non-NaN) mit Farben als PLY exportieren."""
        if PlyData is None or PlyElement is None:
            raise RuntimeError("PLY-Export nicht verfügbar (pip install plyfile).")

        mask = ~np.isnan(distances)
        pts = points[mask]
        cols = colors[mask]

        vertex = np.array(
            [(x, y, z, r, g, b) for (x, y, z), (r, g, b) in zip(pts, cols)],
            dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                   ("red", "u1"), ("green", "u1"), ("blue", "u1")],
        )
        el = PlyElement.describe(vertex, "vertex")
        d = os.path.dirname(outply)
        if d:
            os.makedirs(d, exist_ok=True)   
        PlyData([el], text=False).write(outply)
