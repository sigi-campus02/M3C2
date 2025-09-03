"""Grouped bar chart utilities.

Functions in this module visualise mean and standard deviation for
WITH/INLIER distance data.  They were split from
:mod:`m3c2.visualization.services.report_service` to separate plotting
concerns from report orchestration.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def _folder_of(label: str) -> str:
    """Return the folder ID extracted from a label.

    Labels are typically formatted as ``<version>_<folder>``.  Only the
    substring after the first underscore represents the folder identifier.
    If no underscore is present, the label itself is treated as the folder
    name.
    """

    return label.split("_", 1)[1] if "_" in label else label


def plot_grouped_bar_means_stds_dual(
    fid: str,
    fname: str,
    data_with: Dict[str, np.ndarray],
    data_inlier: Dict[str, np.ndarray],
    colors: Dict[str, str],
    outdir: str,
) -> None:
    """Create grouped bar plots comparing WITH and INLIER data per folder."""

    folder_to_with: Dict[str, List[np.ndarray]] = {}
    folder_to_inl: Dict[str, List[np.ndarray]] = {}

    for k, arr in data_with.items():
        f = _folder_of(k)
        folder_to_with.setdefault(f, [])
        folder_to_with[f].append(arr)
    for k, arr in data_inlier.items():
        f = _folder_of(k)
        folder_to_inl.setdefault(f, [])
        folder_to_inl[f].append(arr)

    all_folders = sorted(set(folder_to_with.keys()) | set(folder_to_inl.keys()))

    means_with, means_inl, stds_with, stds_inl, xlabels, bar_colors = [], [], [], [], [], []
    for f in all_folders:
        arr_with = (
            np.concatenate(folder_to_with.get(f, [])) if f in folder_to_with else np.array([])
        )
        arr_inl = (
            np.concatenate(folder_to_inl.get(f, [])) if f in folder_to_inl else np.array([])
        )

        mean_w_signed = float(np.mean(arr_with)) if arr_with.size else np.nan
        std_w = float(np.std(arr_with)) if arr_with.size else np.nan
        mean_i_signed = float(np.mean(arr_inl)) if arr_inl.size else np.nan
        std_i = float(np.std(arr_inl)) if arr_inl.size else np.nan

        xlabels.append(f)
        mean_w = float(np.abs(mean_w_signed)) if np.isfinite(mean_w_signed) else np.nan
        mean_i = float(np.abs(mean_i_signed)) if np.isfinite(mean_i_signed) else np.nan

        means_with.append(mean_w)
        stds_with.append(std_w)
        means_inl.append(mean_i)
        stds_inl.append(std_i)

        candidate_label = next((k for k in data_with.keys() if k.endswith("_" + f)), None)
        c = colors.get(candidate_label, "#8aa2ff")
        bar_colors.append(c)

    x = np.arange(len(all_folders))
    width = 0.4

    fig, ax = plt.subplots(2, 1, figsize=(max(10, len(all_folders) * 1.8), 8), sharex=True)

    ax[0].bar(x - width / 2, means_with, width, label="mit Outlier (WITH)", color=bar_colors)
    ax[0].bar(
        x + width / 2, means_inl, width, label="ohne Outlier (INLIER)", color=bar_colors, alpha=0.55
    )
    ax[0].set_ylabel("Mittelwert (|μ|)")
    ax[0].set_title(f"Mittelwert je Folder – {fid}/{fname}")
    ax[0].set_ylim(bottom=0)
    ax[0].legend()

    ax[1].bar(x - width / 2, stds_with, width, label="mit Outlier (WITH)", color=bar_colors)
    ax[1].bar(
        x + width / 2, stds_inl, width, label="ohne Outlier (INLIER)", color=bar_colors, alpha=0.55
    )
    ax[1].set_ylabel("Standardabweichung (σ)")
    ax[1].set_title(f"Standardabweichung je Folder – {fid}/{fname}")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(xlabels, rotation=30, ha="right")
    ax[1].set_ylim(bottom=0)
    ax[1].legend()

    plt.tight_layout()
    out = os.path.join(outdir, f"{fid}_{fname}_GroupedBar_Mean_Std.png")
    plt.savefig(out)
    plt.close()
    logger.info("[Report] Plot gespeichert: %s", out)
