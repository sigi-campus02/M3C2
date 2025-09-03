"""PDF and plot generation utilities for M3C2 reports.

This module gathers helper functions that load distance measurements,
compute statistics, and create visualizations which are combined into
multi-page PDF documents.
"""

from __future__ import annotations

import logging
import os
from typing import Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from m3c2.config.plot_config import PlotConfig, PlotOptions
from ..loaders.distance_loader import load_1col_distances, load_coordinates_inlier_distances
from ..plotters.overlay_plotter import (
    get_common_range,
    plot_overlay_boxplot,
    plot_overlay_gauss,
    plot_overlay_histogram,
    plot_overlay_qq,
    plot_overlay_violin,
    plot_overlay_weibull,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _resolve(fid: str, filename: str) -> str:
    """Return the path to *filename* for the given folder ID."""
    p1 = os.path.join(fid, filename)
    if os.path.exists(p1):
        return p1
    return os.path.join("data", "Multi-illumination", "Job_0378_8400-110", "1-3_2-3", fid, filename)


def _load_data(fid: str, filenames: List[str], versions: List[str]) -> Tuple[
    Dict[str, np.ndarray], Dict[str, Tuple[float, float]]
]:
    """Load distance data for a folder and compute Gaussian parameters."""
    data_with: Dict[str, np.ndarray] = {}
    gauss_with: Dict[str, Tuple[float, float]] = {}

    for v in versions:
        base_with = f"{v}_Job_0378_8400-110-rad-{fid}_cloud_moved_m3c2_distances.txt"
        path_with = _resolve(fid, base_with)
        logger.info("[Report] Lade WITH: %s", path_with)
        if not os.path.exists(path_with):
            logger.warning("[Report] Datei fehlt (WITH): %s", path_with)
            continue
        try:
            if v.lower() == "cc":
                try:
                    arr = load_1col_distances(path_with)
                except (OSError, ValueError) as e:
                    logger.warning(
                        "[Report] Standard-Loader fehlgeschlagen (WITH: %s): %s – versuche CC-Fallback",
                        path_with,
                        e,
                    )
                    try:
                        df = pd.read_csv(path_with, sep=";")
                        num_cols = df.select_dtypes(include=[np.number]).columns
                        if len(num_cols) == 0:
                            raise ValueError("Keine numerische Spalte gefunden (CC).")
                        arr = df[num_cols[0]].astype(float).to_numpy()
                        arr = arr[np.isfinite(arr)]
                    except (OSError, ValueError) as e:
                        logger.error(
                            "[Report] CC-Fallback fehlgeschlagen (WITH: %s): %s",
                            path_with,
                            e,
                        )
                        continue
            else:
                arr = load_1col_distances(path_with)
        except (OSError, ValueError) as e:
            logger.error("[Report] Laden fehlgeschlagen (WITH: %s): %s", path_with, e)
            continue

        if arr.size:
            label = f"{v}_{fid}"
            data_with[label] = arr
            mu, std = norm.fit(arr)
            gauss_with[label] = (float(mu), float(std))

    return data_with, gauss_with


def _plot_grouped_bar_means_stds_dual(
    fid: str,
    fname: str,
    data_with: Dict[str, np.ndarray],
    data_inlier: Dict[str, np.ndarray],
    colors: Dict[str, str],
    outdir: str,
) -> None:
    """Create grouped bar plots comparing WITH and INLIER data per folder."""
    def _folder_of(label: str) -> str:
        """Return the folder ID from a combined version/folder label.

        Labels are typically formatted as ``<version>_<folder>`` (for
        example ``"cc_Part_1"``). Only the substring after the first
        underscore represents the folder identifier. If no underscore is
        present, the label itself is treated as the folder name.
        """
        return label.split("_", 1)[1] if "_" in label else label

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


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def overlay_plots(config: PlotConfig, options: PlotOptions) -> None:
    """Create combined overlay plots for a set of distance files.

    Parameters
    ----------
    config : PlotConfig
        Configuration describing the input files and output location. The
        ``folder_ids`` and ``versions`` are combined with ``filenames`` to
        locate the distance data. ``config.ensure_colors`` provides a mapping
        of plot labels to colors and ``config.path`` determines where the PNG
        images are written.
    options : PlotOptions
        Flags selecting which plot types to generate. Available plots are
        overlay histograms, Gaussian and Weibull fits, box plots, Q-Q plots,
        grouped bar charts comparing WITH/INLIER statistics and violin plots.
        Only plots with the corresponding flag set to ``True`` are created.

    Produces
    --------
    For all provided folders and versions the data is aggregated into a single
    "WITH" and an optional "INLIER" collection. For each enabled plot type a
    PNG image is saved in ``config.path`` showing the combined distributions.
    """
    colors = config.ensure_colors()
    os.makedirs(config.path, exist_ok=True)

    data_with_all: Dict[str, np.ndarray] = {}
    for fid in config.folder_ids:
        data_with, _ = _load_data(fid, config.filenames, config.versions)
        if not data_with:
            logger.warning("[Report] Keine WITH-Daten für %s gefunden.", fid)
            continue
        data_with_all.update(data_with)

    if not data_with_all:
        logger.warning("[Report] Keine Daten gefunden – keine Plots erzeugt.")
        return

    data_inlier_all: Dict[str, np.ndarray] = {}
    for fid in config.folder_ids:
        for v in config.versions:
            label = f"{v}_{fid}"
            base_inl = f"{v}_Job_0378_8400-110-rad-{fid}_cloud_moved_m3c2_distances_coordinates_inlier_std.txt"
            path_inl = _resolve(fid, base_inl)
            logger.info("[Report] Lade INLIER: %s", path_inl)
            if not os.path.exists(path_inl):
                logger.warning("[Report] Datei fehlt (INLIER): %s", path_inl)
                continue
            try:
                arr = load_coordinates_inlier_distances(path_inl)
            except (OSError, ValueError) as e:
                logger.error("[Report] Laden fehlgeschlagen (INLIER: %s): %s", path_inl, e)
                continue
            if arr.size:
                data_inlier_all[label] = arr

    data_min, data_max, x = get_common_range(data_with_all)
    fid = "ALLFOLDERS"

    fname = "ALL_WITH"
    gauss_with = {k: norm.fit(v) for k, v in data_with_all.items() if v.size}
    if options.plot_hist:
        plot_overlay_histogram(fid, fname, data_with_all, config.bins, data_min, data_max, colors, config.path)
    if options.plot_gauss:
        plot_overlay_gauss(fid, fname, data_with_all, gauss_with, x, colors, config.path)
    if options.plot_weibull:
        plot_overlay_weibull(fid, fname, data_with_all, x, colors, config.path)
    if options.plot_box:
        plot_overlay_boxplot(fid, fname, data_with_all, colors, config.path)
    if options.plot_qq:
        plot_overlay_qq(fid, fname, data_with_all, colors, config.path)
    if options.plot_grouped_bar:
        _plot_grouped_bar_means_stds_dual(fid, fname, data_with_all, data_inlier_all, colors, config.path)
    if options.plot_violin:
        plot_overlay_violin(fid, fname, data_with_all, colors, config.path)
    logger.info("[Report] PNGs für %s (WITH) erzeugt.", fid)

    fname = "ALL_INLIER"
    if data_inlier_all:
        gauss_inl = {k: norm.fit(v) for k, v in data_inlier_all.items() if v.size}
        if options.plot_hist:
            plot_overlay_histogram(fid, fname, data_inlier_all, config.bins, data_min, data_max, colors, config.path)
        if options.plot_gauss:
            plot_overlay_gauss(fid, fname, data_inlier_all, gauss_inl, x, colors, config.path)
        if options.plot_weibull:
            plot_overlay_weibull(fid, fname, data_inlier_all, x, colors, config.path)
        if options.plot_box:
            plot_overlay_boxplot(fid, fname, data_inlier_all, colors, config.path)
        if options.plot_qq:
            plot_overlay_qq(fid, fname, data_inlier_all, colors, config.path)
        if options.plot_grouped_bar:
            _plot_grouped_bar_means_stds_dual(fid, fname, data_with_all, data_inlier_all, colors, config.path)
        if options.plot_violin:
            plot_overlay_violin(fid, fname, data_inlier_all, colors, config.path)
        logger.info("[Report] PNGs für %s (INLIER) erzeugt.", fid)
    else:
        logger.warning("[Report] Keine INLIER-Daten gefunden – zweite Seite bleibt leer.")


