# visualization_service.py
"""Utility helpers for visualising point cloud distance data.

This module offers convenience functions for turning distance results into
plots or coloured PLY point clouds.  Optional dependencies such as
``seaborn`` or ``plyfile`` are imported lazily so that the module can be
imported in environments where they are not installed.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os
import logging

logger = logging.getLogger(__name__)

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
    """High level routines for plotting and exporting distance results."""

    @staticmethod
    def txt_to_ply_with_distance_color(
        txt_path: str,
        outply: str,
        nan_color: Tuple[int, int, int] = (255, 255, 255),
        percentile_range: Tuple[float, float] = (0.0, 100.0),
        scalar_name: str = "distance",
        write_binary: bool = True,
    ) -> None:
        """Convert a distance text file to a colourised PLY file.

        Parameters
        ----------
        txt_path:
            Path to the text file containing four columns ``x``, ``y``, ``z``
            and ``distance``.
        outply:
            Destination path of the generated PLY file.
        nan_color:
            RGB colour used for points where the distance value is ``NaN``.
        percentile_range:
            Percentile range used to clip distance values before colour
            mapping. This helps to reduce the effect of outliers.
        scalar_name:
            Name of the scalar field written to the PLY file.
        write_binary:
            If ``True`` the PLY file is written in binary format; otherwise
            ASCII.

        Raises
        ------
        RuntimeError
            If the :mod:`plyfile` dependency is not installed.
        ValueError
            If the text file has an unexpected number of columns or contains
            no data.
        """
        if PlyData is None or PlyElement is None:
            raise RuntimeError("PLY-Export nicht verfügbar (pip install plyfile).")

        arr = np.loadtxt(txt_path, skiprows=1)
        if arr.size == 0:
            logger.warning("TXT-Datei enthält keine Werte: %s", txt_path)
            raise ValueError(f"TXT-Datei enthält keine Werte: {txt_path}")
        if arr.ndim == 1:
            logger.warning("TXT-Datei hat nur eine Zeile: %s", txt_path)
            arr = arr.reshape(1, -1)
        if arr.shape[1] != 4:
            logger.warning(
                "TXT-Datei muss 4 Spalten haben (%s hat %d)", txt_path, arr.shape[1]
            )
            raise ValueError(f"TXT-Datei muss 4 Spalten haben: {txt_path}")

        points = arr[:, :3]
        distances = arr[:, 3]
        n = len(distances)

        # Compute per-point colours based on distance percentiles
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

            # Map the normalised distances to a CloudCompare-like colour scale
            cc_colors = [(0.0, "blue"), (0.33, "green"), (0.66, "yellow"), (1.0, "red")]
            cc_cmap = LinearSegmentedColormap.from_list("CC_Colormap", cc_colors)
            colored_valid = (cc_cmap(normed)[:, :3] * 255).astype(np.uint8)
            colors[valid_mask] = colored_valid

        # Assign a uniform colour to invalid entries (NaNs)
        colors[~valid_mask] = np.array(nan_color, dtype=np.uint8)

        # Write the coloured cloud including the distance as scalar field
        _write_ply_xyzrgb(
            points=points,
            colors=colors,
            outply=outply,
            scalar=distances.astype(np.float32),
            scalar_name=scalar_name,
            binary=write_binary,
        )

        logger.info(
            "[TXT->PLY] %s -> %s (%d Punkte, SF='%s')",
            txt_path,
            outply,
            n,
            scalar_name,
        )



    # ---------- Diagramme ----------

    @staticmethod
    def histogram(
        distances: np.ndarray,
        path: Optional[str] = None,
        bins: int = 256,
        title: str = "Verteilung der M3C2-Distanzen",
    ) -> None:
        """Save or show a histogram of the valid distances.

        Parameters
        ----------
        distances:
            Array of distance values; ``NaN`` entries are ignored.
        path:
            If provided, the plot is written to this path instead of being
            displayed interactively.
        bins:
            Number of histogram bins.
        title:
            Title shown above the plot.
        """
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
            logger.info("Histogram saved to %s", path)
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
        """Colour a point cloud based on distance values and export it.

        Parameters
        ----------
        points:
            Point coordinates of shape ``(N, 3)``.
        distances:
            Distance values corresponding to ``points``.
        outply:
            Destination path of the coloured PLY file.
        nan_color:
            RGB colour used for entries where ``distances`` is ``NaN``.
        percentile_range:
            Percentile range used to clip distances before colour mapping.

        Returns
        -------
        numpy.ndarray
            The RGB colour array with shape ``(N, 3)``.

        Raises
        ------
        RuntimeError
            If the :mod:`plyfile` dependency is not installed.
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

            # Apply a CloudCompare-like colour map
            cc_colors = [(0.0, "blue"), (0.33, "green"), (0.66, "yellow"), (1.0, "red")]
            cc_cmap = LinearSegmentedColormap.from_list("CC_Colormap", cc_colors)

            colored_valid = (cc_cmap(normed)[:, :3] * 255).astype(np.uint8)
            colors[valid_mask] = colored_valid

        # Assign default colour to NaN distances
        colors[~valid_mask] = np.array(nan_color, dtype=np.uint8)

        # Serialise the coloured points to a PLY file
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
        logger.info("Colorized %d points -> %s", n, outply)
        return colors

    # --- OPTIONAL: export_valid(...) gleich mit Scalar schreiben (nur wenn du willst) ---
    @staticmethod
    def export_valid(
        points: np.ndarray,
        colors: np.ndarray,
        distances: np.ndarray,
        outply: str,
        scalar_name: str = "distance",
        write_binary: bool = True,
    ) -> None:
        """Export only valid points with colours and a scalar field.

        Parameters
        ----------
        points:
            Array of point coordinates.
        colors:
            RGB colour array associated with ``points``.
        distances:
            Distance values used to filter out invalid points.
        outply:
            Path of the resulting PLY file.
        scalar_name:
            Name of the scalar field written for ``distances``.
        write_binary:
            If ``True`` the PLY file is written in binary format.

        Raises
        ------
        RuntimeError
            If the :mod:`plyfile` dependency is not installed.
        """
        if PlyData is None or PlyElement is None:
            raise RuntimeError("PLY-Export nicht verfügbar (pip install plyfile).")

        if (
            points.size == 0
            or colors.size == 0
            or distances.size == 0
            or points.ndim != 2
            or colors.ndim != 2
            or points.shape[1] != 3
            or colors.shape[1] != 3
            or points.shape[0] != colors.shape[0]
            or points.shape[0] != distances.shape[0]
        ):
            logger.warning(
                "Ungültige Eingabearrays: points%s colors%s distances%s",
                points.shape,
                colors.shape,
                distances.shape,
            )
            return

        # Keep only rows with finite distance values
        mask = ~np.isnan(distances)
        if not mask.any():
            logger.warning("Keine gültigen Punkte zum Exportieren vorhanden")
            return
        pts = points[mask]
        cols = colors[mask]
        dists = distances[mask].astype(np.float32)

        # Delegate writing to the common helper which also stores the scalar
        _write_ply_xyzrgb(
            points=pts,
            colors=cols,
            outply=outply,
            scalar=dists,
            scalar_name=scalar_name,
            binary=write_binary,
        )

        logger.info(
            "[export_valid] %d Punkte exportiert -> %s (SF='%s')",
            pts.shape[0],
            outply,
            scalar_name,
        )



def _write_ply_xyzrgb(
    points: np.ndarray,
    colors: np.ndarray,
    outply: str,
    scalar: Optional[np.ndarray] = None,
    scalar_name: str = "distance",
    binary: bool = True,
) -> None:
    """Write an XYZRGB PLY file with an optional scalar field.

    Parameters
    ----------
    points:
        Array of point coordinates.
    colors:
        RGB colour array aligned with ``points``.
    outply:
        Output path of the generated PLY file.
    scalar:
        Optional scalar values written alongside each vertex.
    scalar_name:
        Name of the scalar property in the PLY file.
    binary:
        If ``True`` the file is written in binary format; otherwise ASCII.

    Raises
    ------
    RuntimeError
        If the :mod:`plyfile` dependency is not installed.
    ValueError
        If the provided arrays have mismatching lengths.
    """
    if PlyData is None or PlyElement is None:
        raise RuntimeError("PLY-Export nicht verfügbar (pip install plyfile).")

    n = points.shape[0]
    if colors.shape[0] != n:
        raise ValueError("Anzahl colors != Anzahl Punkte")

    base_dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    if scalar is not None:
        if scalar.shape[0] != n:
            raise ValueError("Anzahl scalar != Anzahl Punkte")
        base_dtype.append((scalar_name, "f4"))

    if scalar is None:
        vertex = np.array(
            [(x, y, z, r, g, b) for (x, y, z), (r, g, b) in zip(points, colors)],
            dtype=base_dtype,
        )
    else:
        vertex = np.array(
            [(x, y, z, r, g, b, s) for (x, y, z), (r, g, b), s in zip(points, colors, scalar)],
            dtype=base_dtype,
        )

    el = PlyElement.describe(vertex, "vertex")
    d = os.path.dirname(outply)
    if d:
        os.makedirs(d, exist_ok=True)
    # ``binary=True`` yields smaller files; CloudCompare understands both
    PlyData([el], text=not binary).write(outply)

