"""Plotting utilities for visualising distance distributions."""

from __future__ import annotations

from typing import Optional

import logging
import os

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)

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


__all__ = ["histogram"]

