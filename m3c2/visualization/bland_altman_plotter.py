from __future__ import annotations

import logging
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np

from .comparison_loader import _load_and_mask

logger = logging.getLogger(__name__)


def bland_altman_plot(
    folder_ids: List[str],
    ref_variants: List[str],
    outdir: str = "BlandAltman",
) -> None:
    """Create Bland–Altman plots for given folders and reference variants."""
    if len(ref_variants) != 2:
        raise ValueError("ref_variants must contain exactly two entries")

    os.makedirs(outdir, exist_ok=True)

    for fid in folder_ids:
        result = _load_and_mask(fid, ref_variants)
        if result is None:
            continue
        a, b = result

        if a.size == 0 or b.size == 0:
            print(f"[BlandAltman] Leere Distanzwerte in {fid}, übersprungen")
            continue

        mean_vals = (a + b) / 2.0
        diff_vals = a - b
        mean_diff = float(np.mean(diff_vals))
        std_diff = float(np.std(diff_vals, ddof=1))
        upper = mean_diff + 1.96 * std_diff
        lower = mean_diff - 1.96 * std_diff

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
        plt.title(f"Bland-Altman {fid}: {ref_variants[0]} vs {ref_variants[1]}")
        plt.legend()
        outpath = os.path.join(
            outdir,
            f"bland_altman_{fid}_{ref_variants[0]}_vs_{ref_variants[1]}.png",
        )
        plt.tight_layout()
        plt.savefig(outpath, dpi=300)
        plt.close()
