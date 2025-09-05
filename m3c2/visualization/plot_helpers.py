"""Plotting utilities for visualising distance distributions."""

from __future__ import annotations

from typing import Optional

import logging
import os

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

# NOTE: `_square_limits` is a small helper that ensures axis limits are
# square and padded slightly so all points are visible.  It is used by
# several plotters.
def _square_limits(x: np.ndarray, y: np.ndarray, pad: float = 0.05) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return square axis limits covering the ``x`` and ``y`` data.

    Parameters
    ----------
    x, y:
        Arrays of x and y coordinates.
    pad:
        Fractional padding applied to the half-width of the square.

    Returns
    -------
    tuple[tuple[float, float], tuple[float, float]]
        ``(x_limits, y_limits)`` where each is a ``(min, max)`` tuple.
    """

    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    v_min = min(x_min, y_min)
    v_max = max(x_max, y_max)
    cx = cy = (v_min + v_max) / 2.0
    half = max((x_max - x_min), (y_max - y_min)) / 2.0
    half = half * (1.0 + pad) if half > 0 else 1.0
    return (cx - half, cx + half), (cy - half, cy + half)

# seaborn is optional; importing it lazily keeps dependencies light
try:  # pragma: no cover - tested via monkeypatch
    import seaborn as sns  # type: ignore
    _HAS_SNS = True
except ImportError:  # pragma: no cover - informative logging
    _HAS_SNS = False
    logger.info("Missing optional dependency 'seaborn'.")


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
        If provided, the plot is written to this path instead of being displayed
        interactively.
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


__all__ = ["_square_limits", "histogram"]

