"""Utility for comparing distance outputs using a Bland–Altman plot.

This module loads two distance result files for a set of folder IDs and
creates Bland–Altman plots showing the agreement of the two variants.

Example
-------
>>> from statistics_service_comparedistances import StatisticsCompareDistances
>>> StatisticsCompareDistances.bland_altman_plot(
...     folder_ids=["0342-0349"],
...     ref_variants=["ref", "ref_ai"],
... )

The function searches files following the pattern
``python_{variant}_m3c2_distances.txt`` in either ``<fid>/`` or
``data/<fid>/`` and writes one PNG file per folder ID to the output
directory.
"""

from __future__ import annotations

import os
from typing import Iterable, List

import matplotlib.pyplot as plt
import numpy as np


class StatisticsCompareDistances:
    """Service class generating Bland–Altman plots for distance files."""

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

    @staticmethod
    def _expand_folders(folder_ids: Iterable[str]) -> List[str]:
        """Expand ranges like ``'0001-0003'`` into a list of IDs."""

        result: List[str] = []
        for fid in folder_ids:
            if "-" in fid:
                start, end = fid.split("-", 1)
                for num in range(int(start), int(end) + 1):
                    result.append(f"{num:04d}")
            else:
                result.append(fid)
        return result

    @classmethod
    def bland_altman_plot(
        cls,
        folder_ids: Iterable[str],
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
        fids = cls._expand_folders(folder_ids)

        for fid in fids:
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
            a, b = (arr[~np.isnan(arr)] for arr in data)
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate Bland-Altman plots for distance outputs"
    )
    parser.add_argument(
        "--folder_ids",
        nargs="+",
        required=True,
        help="Folder IDs or ranges (e.g. 0001-0005)",
    )
    parser.add_argument(
        "--ref_variants",
        nargs=2,
        required=True,
        help="Two reference variants, e.g. ref ref_ai",
    )
    parser.add_argument(
        "--outdir", default="BlandAltman", help="Output directory for plots"
    )
    args = parser.parse_args()

    StatisticsCompareDistances.bland_altman_plot(
        folder_ids=args.folder_ids,
        ref_variants=list(args.ref_variants),
        outdir=args.outdir,
    )

