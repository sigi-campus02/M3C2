from __future__ import annotations

import logging
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .comparison_loader import _load_and_mask

logger = logging.getLogger(__name__)


def _square_limits(x: np.ndarray, y: np.ndarray, pad: float = 0.05):
    """Return square axis limits covering all points with a small padding."""
    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    v_min = min(x_min, y_min)
    v_max = max(x_max, y_max)
    cx = cy = (v_min + v_max) / 2.0
    half = max((x_max - x_min), (y_max - y_min)) / 2.0
    half = half * (1.0 + pad) if half > 0 else 1.0
    return (cx - half, cx + half), (cy - half, cy + half)


def passing_bablok_plot(
    folder_ids: List[str],
    ref_variants: List[str],
    outdir: str = "PassingBablok",
) -> None:
    """Create Passing–Bablok regression plots."""

    if len(ref_variants) != 2:
        raise ValueError("ref_variants must contain exactly two entries")

    os.makedirs(outdir, exist_ok=True)

    for fid in folder_ids:
        logger.info("[PassingBablok] Processing folder: %s", fid)
        result = _load_and_mask(fid, ref_variants)
        if result is None:
            continue
        x, y = result

        max_n = 1000
        if x.size > max_n:
            idx = np.random.choice(x.size, size=max_n, replace=False)
            x, y = x[idx], y[idx]

        if x.size < 2:
            logger.warning("[PassingBablok] Zu wenige Punkte in %s – übersprungen", fid)
            continue

        n = int(len(x))
        S = []
        for i in range(n - 1):
            x_i, y_i = x[i], y[i]
            for j in range(i + 1, n):
                x_j, y_j = x[j], y[j]
                if (x_i == x_j) and (y_i == y_j):
                    continue
                if x_i == x_j:
                    S.append(np.inf if (y_i > y_j) else -np.inf)
                    continue
                g = (y_i - y_j) / (x_i - x_j)
                if g == -1:
                    continue
                S.append(g)

        if not S:
            logger.warning("[PassingBablok] Keine gültigen Paare in %s", fid)
            continue

        S = np.array(S, dtype=float)
        S.sort()
        N = int(len(S))
        K = int((S < -1).sum())
        if N % 2 != 0:
            idx = int((N + 1) / 2 + K) - 1
            b = float(S[idx])
        else:
            idx = int(N / 2 + K) - 1
            b = float(0.5 * (S[idx] + S[idx + 1]))
        a = float(np.median(y - b * x))

        from scipy import stats as st

        C = 0.95
        gamma = 1 - C
        q = 1 - (gamma / 2.0)
        w = float(st.norm.ppf(q))

        C_gamma = w * np.sqrt((n * (n - 1) * (2 * n + 5)) / 18.0)
        M1 = int(np.round((N - C_gamma) / 2.0))
        M2 = int(N - M1 + 1)
        b_L = float(S[M1 + K - 1])
        b_U = float(S[M2 + K - 1])
        a_L = float(np.median(y - b_U * x))
        a_U = float(np.median(y - b_L * x))

        fig = plt.figure(figsize=(8, 6), constrained_layout=True)
        ax = fig.add_subplot(111)

        ax.scatter(x, y, alpha=0.35, label="Daten", s=12)

        (xl, xu), (yl, yu) = _square_limits(x, y, pad=0.05)
        xx = np.array([xl, xu], dtype=float)

        ax.plot(xx, xx, linestyle="--", color="grey", label="y = x")
        ax.plot(xx, a + b * xx, color="red", label=f"PB: y = {a:.4f} + {b:.4f} x")
        ax.plot(xx, (a_U + b_U * xx), linestyle="--", alpha=0.7,
                label=f"CI oben: y = {a_U:.4f} + {b_U:.4f} x")
        ax.plot(xx, (a_L + b_L * xx), linestyle="--", alpha=0.7,
                label=f"CI unten: y = {a_L:.4f} + {b_L:.4f} x")
        ax.fill_between(xx, a_L + b_L * xx, a_U + b_U * xx, alpha=0.12)

        ax.set_xlim(xl, xu)
        ax.set_ylim(yl, yu)
        ax.set_aspect("equal", adjustable="box")

        ax.set_xlabel(ref_variants[0])
        ax.set_ylabel(ref_variants[1])
        ax.set_title(f"Passing–Bablok {fid}: {ref_variants[0]} vs {ref_variants[1]}")
        ax.legend(frameon=False)

        outpath = os.path.join(
            outdir,
            f"passing_bablok_{fid}_{ref_variants[0]}_vs_{ref_variants[1]}.png",
        )
        plt.savefig(outpath, dpi=300)
        plt.close()

        logger.info(
            f"[PassingBablok] {fid}: b={b:.6f} "
            f"[{b_L:.6f},{b_U:.6f}], a={a:.6f} [{a_L:.6f},{a_U:.6f}] -> {outpath}"
        )
