"""OLS plotting utility for visualizing linear regression between two variants.

This module loads paired reference measurements, fits an ordinary least squares
regression with confidence intervals, and saves the resulting comparison
plots for each folder.
"""

from __future__ import annotations

import logging
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .comparison_loader import _load_and_mask

logger = logging.getLogger(__name__)


def _square_limits(x: np.ndarray, y: np.ndarray, pad: float = 0.05):
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    v_min = min(x_min, y_min)
    v_max = max(x_max, y_max)
    cx = cy = (v_min + v_max) / 2.0
    half = max((x_max - x_min), (y_max - y_min)) / 2.0
    half = half * (1.0 + pad) if half > 0 else 1.0
    return (cx - half, cx + half), (cy - half, cy + half)


def linear_regression_plot(
    folder_ids: List[str],
    ref_variants: List[str],
    outdir: str = "LinearRegression",
) -> None:
    """Create OLS linear regression plots for given folders."""

    if len(ref_variants) != 2:
        raise ValueError("ref_variants must contain exactly two entries")

    os.makedirs(outdir, exist_ok=True)

    for fid in folder_ids:
        logger.info("[OLS] Processing folder: %s", fid)
        result = _load_and_mask(fid, ref_variants)
        if result is None:
            continue
        x, y = result

        max_n = 1000
        if x.size > max_n:
            idx = np.random.choice(x.size, size=max_n, replace=False)
            x, y = x[idx], y[idx]

        if x.size < 3:
            logger.warning("[OLS] Zu wenige Punkte in %s – übersprungen", fid)
            continue

        n = x.size
        xbar = float(np.mean(x))
        ybar = float(np.mean(y))
        Sxx = float(np.sum((x - xbar) ** 2))
        if Sxx == 0.0:
            logger.warning("[OLS] Sxx=0 (keine Varianz in x) – übersprungen: %s", fid)
            continue

        Sxy = float(np.sum((x - xbar) * (y - ybar)))
        b = Sxy / Sxx
        a = ybar - b * xbar

        resid = y - (a + b * x)
        SSE = float(np.sum(resid ** 2))
        s2 = SSE / (n - 2)
        se_b = float(np.sqrt(s2 / Sxx))
        se_a = float(np.sqrt(s2 * (1.0 / n + (xbar ** 2) / Sxx)))

        from scipy.stats import t

        tcrit = float(t.ppf(0.975, df=n - 2))
        b_L, b_U = b - tcrit * se_b, b + tcrit * se_b
        a_L, a_U = a - tcrit * se_a, a + tcrit * se_a

        fig = plt.figure(figsize=(8, 6), constrained_layout=True)
        ax = fig.add_subplot(111)

        ax.scatter(x, y, alpha=0.35, label="Daten", s=12)

        (xl, xu), (yl, yu) = _square_limits(x, y, pad=0.05)
        xx = np.array([xl, xu], dtype=float)

        ax.plot(xx, xx, linestyle="--", color="grey", label="y = x")
        ax.plot(xx, a + b * xx, color="red", label=f"OLS: y = {a:.4f} + {b:.4f} x")
        ax.plot(
            xx,
            a_U + b_U * xx,
            linestyle="--",
            alpha=0.7,
            label=f"CI oben: y = {a_U:.4f} + {b_U:.4f} x",
        )
        ax.plot(
            xx,
            a_L + b_L * xx,
            linestyle="--",
            alpha=0.7,
            label=f"CI unten: y = {a_L:.4f} + {b_L:.4f} x",
        )
        ax.fill_between(xx, a_L + b_L * xx, a_U + b_U * xx, alpha=0.12)

        ax.set_xlim(xl, xu)
        ax.set_ylim(yl, yu)
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel(ref_variants[0])
        ax.set_ylabel(ref_variants[1])
        ax.set_title(f"Linear Regression {fid}: {ref_variants[0]} vs {ref_variants[1]}")
        ax.legend(frameon=False)

        outpath = os.path.join(
            outdir,
            f"linear_regression_{fid}_{ref_variants[0]}_vs_{ref_variants[1]}.png",
        )
        plt.savefig(outpath, dpi=300)
        plt.close()

        logger.info(
            f"[OLS] {fid}: b={b:.6f} [{b_L:.6f},{b_U:.6f}], "
            f"a={a:.6f} [{a_L:.6f},{a_U:.6f}] -> {outpath}"
        )
