"""PDF and plot generation utilities for M3C2 reports.

This module gathers helper functions that load distance measurements,
compute statistics, and create visualizations which are combined into
multi-page PDF documents.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, List, Tuple

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm

from m3c2.config.plot_config import PlotConfig, PlotOptions
from .distance_loader import load_1col_distances, load_coordinates_inlier_distances
from .overlay_plotter import (
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
                except Exception:
                    df = pd.read_csv(path_with, sep=";")
                    num_cols = df.select_dtypes(include=[np.number]).columns
                    if len(num_cols) == 0:
                        raise ValueError("Keine numerische Spalte gefunden (CC).")
                    arr = df[num_cols[0]].astype(float).to_numpy()
                    arr = arr[np.isfinite(arr)]
            else:
                arr = load_1col_distances(path_with)
        except Exception as e:
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
            except Exception as e:
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


def summary_pdf(config: PlotConfig) -> None:
    plot_types = [
        ("OverlayHistogramm", "Histogramm", (0, 0)),
        ("Boxplot", "Boxplot", (0, 1)),
        ("OverlayGaussFits", "Gauss-Fit", (0, 2)),
        ("OverlayWeibullFits", "Weibull-Fit", (1, 0)),
        ("QQPlot", "Q-Q-Plot", (1, 1)),
        ("GroupedBar_Mean_Std", "Mittelwert & Std Dev", (1, 2)),
    ]

    fid = "ALLFOLDERS"
    outfile = os.path.join(config.path, f"{fid}_comparison_report.pdf")
    pdf = PdfPages(outfile)

    def _add_page(suffix_label: str, title_suffix: str) -> None:
        """Create a single summary page from existing plot images.

        Each page arranges up to six pre-generated PNG plots in a 2×3 grid.
        ``suffix_label`` selects which image set to include (e.g. WITH vs
        INLIER plots), while ``title_suffix`` is appended to the page title to
        clarify the plotted data.
        """
        fig, axs = plt.subplots(2, 3, figsize=(24, 16))
        for suffix, title, (row, col) in plot_types:
            ax = axs[row, col]
            png = os.path.join(config.path, f"{fid}_{suffix_label}_{suffix}.png")
            if os.path.exists(png):
                img = mpimg.imread(png)
                ax.imshow(img)
                ax.axis("off")
                ax.set_title(title, fontsize=22)
            else:
                ax.axis("off")
                ax.set_title(f"{title}\n(nicht gefunden)", fontsize=18)
        plt.suptitle(f"{fid} – Vergleichsplots ({title_suffix})", fontsize=28)
        plt.subplots_adjust(
            left=0.03, right=0.97, top=0.92, bottom=0.08, wspace=0.08, hspace=0.15
        )
        pdf.savefig(fig)
        plt.close(fig)

    _add_page("ALL_WITH", "inkl. Outlier")
    _add_page("ALL_INLIER", "ohne Outlier (Inlier)")

    pdf.close()
    logger.info("[Report] Zusammenfassung gespeichert: %s", outfile)


def build_parts_pdf(
    outdir: str,
    pdf_path: str | None = None,
    include_with: bool = True,
    include_inlier: bool = True,
) -> str:
    if include_with == include_inlier:
        raise ValueError("Bitte genau einen Modus wählen: include_with XOR include_inlier.")
    mode = "WITH" if include_with else "INLIER"
    subtitle = "incl. outliers" if include_with else "excl. outliers"

    part_ids: List[int] = []
    pat5 = re.compile(
        r"^Part_(\d+)_(WITH|INLIER)_(OverlayHistogramm|OverlayGaussFits|OverlayWeibullFits|Boxplot|QQPlot)\.png$"
    )
    patDual = re.compile(r"^Part_(\d+)_DUAL_GroupedBar_Mean_Std\.png$")
    for fn in os.listdir(outdir):
        m = pat5.match(fn)
        if m:
            part_ids.append(int(m.group(1)))
            continue
        m = patDual.match(fn)
        if m:
            part_ids.append(int(m.group(1)))
    part_ids = sorted(set(part_ids))
    if not part_ids:
        logger.warning("[Report] No part PNGs found in %s – nothing to summarize.", outdir)
        return ""

    pdf_path = pdf_path or os.path.join(outdir, "parts_summary.pdf")

    plot_defs = [
        ("OverlayHistogramm", "Histogram"),
        ("OverlayGaussFits", "Gaussian fit"),
        ("OverlayWeibullFits", "Weibull fit"),
        ("Boxplot", "Box plot"),
        ("QQPlot", "Q–Q plot"),
        ("DUAL_GroupedBar_Mean_Std", "Means & Std (WITH vs INLIER)"),
    ]

    with PdfPages(pdf_path) as pdf:
        for i in part_ids:
            fid = f"Part_{i}"
            fig, axs = plt.subplots(2, 3, figsize=(24, 12))
            for idx, (suffix, title) in enumerate(plot_defs):
                r, c = divmod(idx, 3)
                ax = axs[r, c]
                if suffix == "DUAL_GroupedBar_Mean_Std":
                    png = os.path.join(outdir, f"{fid}_DUAL_GroupedBar_Mean_Std.png")
                else:
                    png = os.path.join(outdir, f"{fid}_{mode}_{suffix}.png")
                if os.path.exists(png):
                    img = mpimg.imread(png)
                    ax.imshow(img)
                    ax.axis("off")
                    ax.set_title(f"{title} – {subtitle}", fontsize=12)
                else:
                    ax.axis("off")
                    ax.set_title(f"{title} – {subtitle}\n(missing)", fontsize=12)
            plt.suptitle(f"{fid}", fontsize=20)
            plt.tight_layout(rect=[0, 0, 1, 0.95])
            pdf.savefig(fig)
            plt.close(fig)

    logger.info("[Report] PDF created: %s", pdf_path)
    return pdf_path


def merge_pdfs(pdf_paths: List[str], out_path: str) -> str:
    """Merge multiple PDFs into a single file."""
    try:
        from PyPDF2 import PdfMerger
    except Exception as e:
        raise RuntimeError("PyPDF2 is required for merging PDFs") from e

    merger = PdfMerger()
    for p in pdf_paths:
        if os.path.exists(p):
            merger.append(p)
        else:
            logger.warning("[Report] PDF missing: %s", p)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        merger.write(f)
    merger.close()
    logger.info("[Report] merged PDFs into %s", out_path)
    return out_path
