"""Helpers for exporting coloured point clouds to PLY files.

This module contains standalone helper functions kept free of class wrappers,
making imports lightweight and avoiding optional dependencies unless required.
"""

from __future__ import annotations

from typing import Optional, Tuple

import logging
import os

import numpy as np
from matplotlib.colors import LinearSegmentedColormap

logger = logging.getLogger(__name__)

# ``plyfile`` is an optional dependency.  Importing it lazily allows this module
# to be used in environments where the package is not installed.
try:  # pragma: no cover - exercised in tests via monkeypatch
    from plyfile import PlyData, PlyElement  # type: ignore
except ImportError:  # pragma: no cover - informative logging only
    PlyData = None
    PlyElement = None
    logger.info("Missing optional dependency 'plyfile'.")


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
        cc_colors = [
            (0.0, "blue"),
            (0.33, "green"),
            (0.66, "yellow"),
            (1.0, "red"),
        ]
        cc_cmap = LinearSegmentedColormap.from_list("CC_Colormap", cc_colors)

        colored_valid = (cc_cmap(normed)[:, :3] * 255).astype(np.uint8)
        colors[valid_mask] = colored_valid

    # Assign default colour to NaN distances
    colors[~valid_mask] = np.array(nan_color, dtype=np.uint8)

    # Serialise the coloured points to a PLY file
    vertex = np.array(
        [
            (x, y, z, r, g, b)
            for (x, y, z), (r, g, b) in zip(points, colors)
        ],
        dtype=[
            ("x", "f4"),
            ("y", "f4"),
            ("z", "f4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ],
    )
    el = PlyElement.describe(vertex, "vertex")

    d = os.path.dirname(outply)
    if d:
        os.makedirs(d, exist_ok=True)
    PlyData([el], text=False).write(outply)
    logger.info("Colorized %d points -> %s", n, outply)
    return colors


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
        ("x", "f4"),
        ("y", "f4"),
        ("z", "f4"),
        ("red", "u1"),
        ("green", "u1"),
        ("blue", "u1"),
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
            [
                (x, y, z, r, g, b, s)
                for (x, y, z), (r, g, b), s in zip(points, colors, scalar)
            ],
            dtype=base_dtype,
        )

    el = PlyElement.describe(vertex, "vertex")
    d = os.path.dirname(outply)
    if d:
        os.makedirs(d, exist_ok=True)
    # ``binary=True`` yields smaller files; CloudCompare understands both
    PlyData([el], text=not binary).write(outply)


__all__ = [
    "colorize",
    "export_valid",
]

