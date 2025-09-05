"""Bland–Altman plot creation utilities.

This module loads paired distance measurements for given folders and
reference variants and visualizes their agreement using Bland–Altman
scatter plots. It computes the mean difference and 95% limits of agreement
and saves the resulting plots for further analysis.
"""

from __future__ import annotations

import logging
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from ..loaders.comparison_loader import _load_and_mask

logger = logging.getLogger(__name__)


def bland_altman_plot(
    folder_ids: List[str],
    reference_variants: List[str],
    outdir: str = "BlandAltman",
) -> None:
    """Create Bland–Altman plots for given folders and reference variants.

    Parameters
    ----------
    folder_ids : List[str]
        Identifiers of the folders whose comparison data should be plotted.
    reference_variants : List[str]
        Names of the two reference variants to compare. Must contain exactly
        two entries.
    outdir : str, optional
        Directory where the generated plots are stored. Defaults to
        ``"BlandAltman"``.

    The function saves one PNG per folder in ``outdir`` with the filename
    pattern ``bland_altman_<folder>_<ref1>_vs_<ref2>.png``.
    """
    if len(reference_variants) != 2:
        raise ValueError("reference_variants must contain exactly two entries")

    os.makedirs(outdir, exist_ok=True)

    for fid in folder_ids:
        result = _load_and_mask(fid, reference_variants)
        if result is None:
            continue
        a, b = result

        if a.size == 0 or b.size == 0:
            logger.warning("[BlandAltman] Leere Distanzwerte in %s, übersprungen", fid)
            continue

        mean_vals = (a + b) / 2.0
        diff_vals = a - b
        mean_diff = float(np.mean(diff_vals))
        std_diff = float(np.std(diff_vals, ddof=1))
        upper = mean_diff + 1.96 * std_diff
        lower = mean_diff - 1.96 * std_diff

        logger.debug(
            "[BlandAltman] %s statistics: mean_diff=%.6f, std_diff=%.6f, upper=%.6f, lower=%.6f, n=%d",
            fid,
            mean_diff,
            std_diff,
            upper,
            lower,
            a.size,
        )

        logger.info(
            f"[BlandAltman] {fid}: mean_diff={mean_diff:.6f}, std_diff={std_diff:.6f}, "
            f"upper={upper:.6f}, lower={lower:.6f}, n={a.size} -> {outdir}"
        )

        plt.figure(figsize=(8, 6))
        plt.scatter(mean_vals, diff_vals, alpha=0.3)
        plt.axhline(mean_diff, color="red", linestyle="--", label=f"Mean diff {mean_diff:.4f}")
        plt.axhline(upper, color="green", linestyle="--", label=f"+1.96 SD {upper:.4f}")
        plt.axhline(lower, color="green", linestyle="--", label=f"-1.96 SD {lower:.4f}")
        plt.xlabel("Mean of measurements")
        plt.ylabel("Difference")
        plt.title(f"Bland-Altman {fid}: {reference_variants[0]} vs {reference_variants[1]}")
        plt.legend()
        outpath = os.path.join(
            outdir,
            f"bland_altman_{fid}_{reference_variants[0]}_vs_{reference_variants[1]}.png",
        )
        plt.tight_layout()
        plt.savefig(outpath, dpi=300)
        plt.close()
