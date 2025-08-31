"""Visualisation utilities for point cloud distance data.

This module contains helpers to colour point clouds based on distance values,
export them as PLY files and create basic plots for exploratory analysis.
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import os

# seaborn is optional; fall back to matplotlib if unavailable
try:
    import seaborn as sns  # type: ignore
    _HAS_SNS = True
except Exception:
    _HAS_SNS = False

# plyfile is optional; only required for exporting PLY files
try:
    from plyfile import PlyData, PlyElement  # type: ignore
except Exception:
    PlyData = None
    PlyElement = None


class VisualizationService:
    """Functions for colouring point clouds and creating simple plots."""

    # Export helpers -----------------------------------------------------
    @staticmethod
    def txt_to_ply_with_distance_color(
        txt_path: str,
        outply: str,
        nan_color: Tuple[int, int, int] = (255, 255, 255),
        percentile_range: Tuple[float, float] = (0.0, 100.0),
        scalar_name: str = "distance",              # <— NEU: frei benennbar
        write_binary: bool = True,                  # <— optional
    ) -> None:
        """Convert a TXT point cloud to a coloured PLY file.

        Parameters
        ----------
        txt_path:
            Path to a text file containing columns ``x y z distance``.
        outply:
            Output path for the generated PLY file.
        nan_color:
            RGB colour used for points with undefined distance values.
        percentile_range:
            Percentile range used to clip distances for colour mapping.
        scalar_name:
            Name of the scalar field to store the distance values under.
        write_binary:
            Whether to write the PLY file in binary format.

        Raises
        ------
        RuntimeError
            If the optional ``plyfile`` dependency is not installed.
        ValueError
            If the input file does not contain four columns.
        """

        if PlyData is None or PlyElement is None:
            raise RuntimeError("PLY export requires the 'plyfile' package.")

        arr = np.loadtxt(txt_path, skiprows=1)
        if arr.size == 0:
            raise ValueError(f"TXT file contains no values: {txt_path}")
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        if arr.shape[1] != 4:
            raise ValueError(f"TXT file must have 4 columns: {txt_path}")

        points = arr[:, :3]
        distances = arr[:, 3]
        n = len(distances)

        # Compute colours based on distance values
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

            # Colour ramp similar to CloudCompare: blue → green → yellow → red
            cc_colors = [(0.0, "blue"), (0.33, "green"), (0.66, "yellow"), (1.0, "red")]
            cc_cmap = LinearSegmentedColormap.from_list("CC_Colormap", cc_colors)
            colored_valid = (cc_cmap(normed)[:, :3] * 255).astype(np.uint8)
            colors[valid_mask] = colored_valid

        # Uncoloured points (NaN distances) appear as the specified nan_color
        colors[~valid_mask] = np.array(nan_color, dtype=np.uint8)

        # Write PLY file including distance as an additional scalar property
        _write_ply_xyzrgb(
            points=points,
            colors=colors,
            outply=outply,
            scalar=distances.astype(np.float32),
            scalar_name=scalar_name,
            binary=write_binary,
        )
        import logging
        logger = logging.getLogger(__name__)
        logger.info(
            f"[TXT->PLY] {txt_path} -> {outply} ({n} points, scalar='{scalar_name}')"
        )



    # ---------- Diagramme ----------

    @staticmethod
    def histogram(
        distances: np.ndarray,
        path: Optional[str] = None,
        bins: int = 256,
        title: str = "Distribution of M3C2 distances",
    ) -> None:
        """Plot and optionally save a histogram of valid distance values.

        Parameters
        ----------
        distances:
            Array containing distance values with possible NaNs.
        path:
            Optional path to save the figure; if omitted the plot is only
            displayed.
        bins:
            Number of histogram bins to use.
        title:
            Title for the plot.
        """

        vals = distances[~np.isnan(distances)]
        plt.figure(figsize=(10, 6))
        if _HAS_SNS:
            sns.histplot(vals, bins=bins, kde=False)
        else:
            plt.hist(vals, bins=bins)
        plt.title(title)
        plt.xlabel("M3C2 distance")
        plt.ylabel("Number of points")
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
        """Colour a point cloud by distance and export it as a PLY file.

        Parameters
        ----------
        points:
            ``(N,3)`` array of point coordinates.
        distances:
            ``(N,)`` array of distance values used for colouring.
        outply:
            Output path for the coloured PLY file.
        nan_color:
            Colour assigned to points with NaN distances.
        percentile_range:
            Percentile range to clip distances for colour mapping.

        Returns
        -------
        numpy.ndarray
            Array of RGB colours used for each point.

        Raises
        ------
        RuntimeError
            If the optional ``plyfile`` dependency is missing.
        """

        if PlyData is None or PlyElement is None:
            raise RuntimeError("PLY export requires the 'plyfile' package.")

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

            # Colour ramp similar to CloudCompare
            cc_colors = [(0.0, "blue"), (0.33, "green"), (0.66, "yellow"), (1.0, "red")]
            cc_cmap = LinearSegmentedColormap.from_list("CC_Colormap", cc_colors)

            colored_valid = (cc_cmap(normed)[:, :3] * 255).astype(np.uint8)
            colors[valid_mask] = colored_valid

        # Assign default colour to NaN distances
        colors[~valid_mask] = np.array(nan_color, dtype=np.uint8)

        # Write the coloured points to a PLY file
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

    # Optional helper to export only valid points while writing distances as a scalar
    @staticmethod
    def export_valid(
        points: np.ndarray,
        colors: np.ndarray,
        distances: np.ndarray,
        outply: str,
        scalar_name: str = "distance",     # <— NEU
        write_binary: bool = True,         # <— NEU
    ) -> None:
        """Export only valid (non-NaN) points as a coloured PLY file.

        Parameters
        ----------
        points:
            ``(N,3)`` point coordinates.
        colors:
            ``(N,3)`` colour array matching ``points``.
        distances:
            Distance values used to filter valid points and written as a
            scalar field.
        outply:
            Output path for the PLY file.
        scalar_name:
            Name of the stored scalar field.
        write_binary:
            Write the PLY in binary format when ``True``.
        """
        if PlyData is None or PlyElement is None:
            raise RuntimeError("PLY-Export nicht verfügbar (pip install plyfile).")

        mask = ~np.isnan(distances)
        pts = points[mask]
        cols = colors[mask]
        dists = distances[mask].astype(np.float32)

        _write_ply_xyzrgb(
            points=pts,
            colors=cols,
            outply=outply,
            scalar=dists,
            scalar_name=scalar_name,
            binary=write_binary,
        )



def _write_ply_xyzrgb(
    points: np.ndarray,
    colors: np.ndarray,
    outply: str,
    scalar: Optional[np.ndarray] = None,
    scalar_name: str = "distance",
    binary: bool = True,
) -> None:
    """Write a PLY file with geometry, colour and optional scalar values.

    Parameters
    ----------
    points:
        ``(N,3)`` array of point coordinates.
    colors:
        ``(N,3)`` array of uint8 RGB colours.
    outply:
        Destination path for the PLY file.
    scalar:
        Optional ``(N,)`` array to write as an additional float property.
    scalar_name:
        Name of the scalar field stored for each vertex.
    binary:
        Write binary PLY when ``True`` for smaller files.

    Raises
    ------
    RuntimeError
        If the ``plyfile`` dependency is not available.
    ValueError
        If the array sizes do not match.
    """

    if PlyData is None or PlyElement is None:
        raise RuntimeError("PLY export requires the 'plyfile' package.")

    n = points.shape[0]
    if colors.shape[0] != n:
        raise ValueError("colors array must match points length")

    base_dtype = [
        ("x", "f4"), ("y", "f4"), ("z", "f4"),
        ("red", "u1"), ("green", "u1"), ("blue", "u1"),
    ]
    if scalar is not None:
        if scalar.shape[0] != n:
            raise ValueError("scalar array must match points length")
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
    # binary=True yields smaller files; CloudCompare understands both formats
    PlyData([el], text=not binary).write(outply)
