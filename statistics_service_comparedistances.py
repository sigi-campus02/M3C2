from __future__ import annotations
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np


class StatisticsCompareDistances:
    """Service class generating comparison plots for distance files."""

    @staticmethod
    def _resolve(fid: str, filename: str) -> str:
        """Return the path to *filename* for the given folder ID.

        The helper searches first in ``<fid>/`` and then in ``data/<fid>/``
        to mirror the behaviour of other services in this repository.
        """

        p1 = os.path.join(fid, filename)
        if os.path.exists(p1):
            return p1
        return os.path.join("data", fid, filename)

    @classmethod
    def bland_altman_plot(
        cls,
        folder_ids: List[str],
        ref_variants: List[str],
        outdir: str = "BlandAltman",
    ) -> None:
        """Create Bland–Altman plots for the given folder IDs.

        Parameters
        ----------
        folder_ids:
            List of folder identifiers or ranges (e.g. ``"0001-0003"``).
        ref_variants:
            Exactly two variants that form the file name pattern
            ``python_{variant}_m3c2_distances.txt``.
        outdir:
            Directory in which the PNG plots will be stored.
        """

        if len(ref_variants) != 2:
            raise ValueError("ref_variants must contain exactly two entries")

        os.makedirs(outdir, exist_ok=True)

        for fid in folder_ids:
            paths = []
            for variant in ref_variants:
                basename = f"python_{variant}_m3c2_distances.txt"
                path = cls._resolve(fid, basename)
                if not os.path.exists(path):
                    print(f"[BlandAltman] Datei nicht gefunden: {path}")
                    path = None
                paths.append(path)

            if None in paths:
                # At least one file is missing -> skip this folder
                continue

            data = [np.loadtxt(p) for p in paths]
            
            a_raw, b_raw = data
            mask = ~np.isnan(a_raw) & ~np.isnan(b_raw)
            a = a_raw[mask]
            b = b_raw[mask]


            if a.size == 0 or b.size == 0:
                print(f"[BlandAltman] Leere Distanzwerte in {fid}, übersprungen")
                continue

            # Bland–Altman calculations
            mean_vals = (a + b) / 2.0
            diff_vals = a - b
            mean_diff = float(np.mean(diff_vals))
            std_diff = float(np.std(diff_vals, ddof=1))
            upper = mean_diff + 1.96 * std_diff
            lower = mean_diff - 1.96 * std_diff

            # Plot
            plt.figure(figsize=(8, 6))
            plt.scatter(mean_vals, diff_vals, alpha=0.3)
            plt.axhline(mean_diff, color="red", linestyle="--",
                        label=f"Mean diff {mean_diff:.4f}")
            plt.axhline(upper, color="green", linestyle="--",
                        label=f"+1.96 SD {upper:.4f}")
            plt.axhline(lower, color="green", linestyle="--",
                        label=f"-1.96 SD {lower:.4f}")
            plt.xlabel("Mean of measurements")
            plt.ylabel("Difference")
            plt.title(
                f"Bland-Altman {fid}: {ref_variants[0]} vs {ref_variants[1]}"
            )
            plt.legend()
            outpath = os.path.join(
                outdir,
                f"bland_altman_{fid}_{ref_variants[0]}_vs_{ref_variants[1]}.png",
            )
            plt.tight_layout()
            plt.savefig(outpath, dpi=300)
            plt.close()

    @classmethod
    def passing_bablok_plot(
        cls,
        folder_ids: List[str],
        ref_variants: List[str],
        outdir: str = "PassingBablok",
    ) -> None:
        """Create Passing–Bablok regression plots for the given folder IDs.

        Parameters
        ----------
        folder_ids:
            List of folder identifiers or ranges (e.g. ``"0001-0003"``).
        ref_variants:
            Exactly two variants that form the file name pattern
            ``python_{variant}_m3c2_distances.txt``.
        outdir:
            Directory in which the PNG plots will be stored.
        """

        if len(ref_variants) != 2:
            raise ValueError("ref_variants must contain exactly two entries")

        os.makedirs(outdir, exist_ok=True)

        for fid in folder_ids:
            logger.info(f"Processing folder: {fid}")
            paths = []
            for variant in ref_variants:
                logger.info(f" Looking for variant: {variant}")
                basename = f"python_{variant}_m3c2_distances.txt"
                path = cls._resolve(fid, basename)
                if not os.path.exists(path):
                    logger.warning(f"[PassingBablok] Datei nicht gefunden: {path}")
                    path = None
                paths.append(path)

            if None in paths:
                # At least one file is missing -> skip this folder
                continue

            data = [np.loadtxt(p) for p in paths]

            x_raw, y_raw = data
            mask = ~np.isnan(x_raw) & ~np.isnan(y_raw)
            x = x_raw[mask]
            y = y_raw[mask]


            max_n = 5000
            if len(x) > max_n:
                idx = np.random.choice(len(x), size=max_n, replace=False)
                x, y = x[idx], y[idx]

            if x.size == 0 or y.size == 0:
                print(f"[PassingBablok] Leere Distanzwerte in {fid}, übersprungen")
                continue

            # Passing–Bablok regression
            slopes = []
            n = len(x)
            for i in range(n - 1):
                for j in range(i + 1, n):
                    if x[j] != x[i]:
                        slopes.append((y[j] - y[i]) / (x[j] - x[i]))

            if not slopes:
                print(f"[PassingBablok] Keine gültigen Paare in {fid}, übersprungen")
                continue

            slopes = np.array(slopes)
            slopes_sorted = np.sort(slopes)
            N = len(slopes_sorted)

            # Median slope
            slope = float(np.median(slopes_sorted))

            # CI-Berechnung nach Passing & Bablok
            z = 1.96  # für 95%
            C_alpha = z * np.sqrt(N*(N-1)*(2*N+5)/18)
            L = int(np.floor((N - C_alpha)/2))
            U = int(np.ceil( (N + C_alpha)/2.0 )) - 1

            L = max(L, 0)
            U = min(U, N-1)

            slope_low = float(slopes_sorted[L])
            slope_high = float(slopes_sorted[U])

            # Intercept + CI
            intercept = float(np.median(y - slope * x))
            intercepts = y - slope * x
            intercept_low = float(np.median(y - slope_high * x))
            intercept_high = float(np.median(y - slope_low * x))


            plt.figure(figsize=(8, 6))
            plt.scatter(x, y, alpha=0.3)

            min_val = float(min(np.min(x), np.min(y)))
            max_val = float(max(np.max(x), np.max(y)))
            line_x = np.array([min_val, max_val], dtype=float)

            # Identity
            plt.plot(line_x, line_x, color="grey", linestyle="--", label="Identity y=x")

            # PB-Linie + 95%-CI-Band
            plt.plot(line_x, intercept + slope * line_x,
                    color="red", label=f"PB: y = {slope:.4f}x + {intercept:.4f}")
            plt.plot(line_x, intercept_low  + slope_low  * line_x, color="orange", linestyle="--",
                    label=f"Slope 95% CI: [{slope_low:.4f}, {slope_high:.4f}]")
            plt.plot(line_x, intercept_high + slope_high * line_x, color="orange", linestyle="--")

            plt.xlabel(ref_variants[0]); plt.ylabel(ref_variants[1])
            plt.title(f"Passing–Bablok {fid}: {ref_variants[0]} vs {ref_variants[1]}")
            plt.legend(); plt.tight_layout()
            plt.savefig(outpath, dpi=300); plt.close()

            plt.close()


