"""PDF and plot generation utilities for M3C2 reports.

This module gathers helper functions that load distance measurements,
compute statistics, and create visualizations which are combined into
multi-page PDF documents.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Dict, List

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import norm

from m3c2.config.plot_config import PlotConfig, PlotOptions
from ..loaders.distance_loader import load_coordinates_inlier_distances
from ..plotters.overlay_plotter import (
    get_common_range,
    plot_overlay_boxplot,
    plot_overlay_gauss,
    plot_overlay_histogram,
    plot_overlay_qq,
    plot_overlay_violin,
    plot_overlay_weibull,
)
from .data_loader import load_distance_data, resolve_path
from ..plotters.grouped_bar_plotter import plot_grouped_bar_means_stds_dual

logger = logging.getLogger(__name__)


class ReportBuilder:
    """Coordinate data loading and plotting for report creation."""

    def __init__(self, config: PlotConfig, options: PlotOptions) -> None:
        self.config = config
        self.options = options
        self.colors = config.ensure_colors()
        os.makedirs(config.path, exist_ok=True)

    def _load_with_data(self) -> Dict[str, "np.ndarray"]:
        data_with_all: Dict[str, "np.ndarray"] = {}
        for fid in self.config.folder_ids:
            data_with, _ = load_distance_data(
                fid, self.config.filenames, self.config.versions
            )
            if not data_with:
                logger.warning("[Report] Keine WITH-Daten für %s gefunden.", fid)
                continue
            data_with_all.update(data_with)
        return data_with_all

    def _load_inlier_data(self) -> Dict[str, "np.ndarray"]:
        data_inlier_all: Dict[str, "np.ndarray"] = {}
        for fid in self.config.folder_ids:
            for v in self.config.versions:
                label = f"{v}_{fid}"
                base_inl = (
                    f"{v}_Job_0378_8400-110-rad-{fid}_cloud_moved_m3c2_distances_coordinates_inlier_std.txt"
                )
                path_inl = resolve_path(fid, base_inl)
                logger.info("[Report] Lade INLIER: %s", path_inl)
                if not os.path.exists(path_inl):
                    logger.warning("[Report] Datei fehlt (INLIER): %s", path_inl)
                    continue
                try:
                    arr = load_coordinates_inlier_distances(path_inl)
                except (OSError, ValueError) as e:
                    logger.error(
                        "[Report] Laden fehlgeschlagen (INLIER: %s): %s", path_inl, e
                    )
                    continue
                if arr.size:
                    data_inlier_all[label] = arr
        return data_inlier_all

    def overlay_plots(self) -> None:
        data_with_all = self._load_with_data()
        if not data_with_all:
            logger.warning("[Report] Keine Daten gefunden – keine Plots erzeugt.")
            return
        data_inlier_all = self._load_inlier_data()

        data_min, data_max, x = get_common_range(data_with_all)
        fid = "ALLFOLDERS"

        fname = "ALL_WITH"
        gauss_with = {k: norm.fit(v) for k, v in data_with_all.items() if v.size}
        if self.options.plot_hist:
            plot_overlay_histogram(
                fid,
                fname,
                data_with_all,
                self.config.bins,
                data_min,
                data_max,
                self.colors,
                self.config.path,
            )
        if self.options.plot_gauss:
            plot_overlay_gauss(
                fid, fname, data_with_all, gauss_with, x, self.colors, self.config.path
            )
        if self.options.plot_weibull:
            plot_overlay_weibull(
                fid, fname, data_with_all, x, self.colors, self.config.path
            )
        if self.options.plot_box:
            plot_overlay_boxplot(
                fid, fname, data_with_all, self.colors, self.config.path
            )
        if self.options.plot_qq:
            plot_overlay_qq(
                fid, fname, data_with_all, self.colors, self.config.path
            )
        if self.options.plot_grouped_bar:
            plot_grouped_bar_means_stds_dual(
                fid,
                fname,
                data_with_all,
                data_inlier_all,
                self.colors,
                self.config.path,
            )
        if self.options.plot_violin:
            plot_overlay_violin(
                fid, fname, data_with_all, self.colors, self.config.path
            )
        logger.info("[Report] PNGs für %s (WITH) erzeugt.", fid)

        fname = "ALL_INLIER"
        if data_inlier_all:
            gauss_inl = {k: norm.fit(v) for k, v in data_inlier_all.items() if v.size}
            if self.options.plot_hist:
                plot_overlay_histogram(
                    fid,
                    fname,
                    data_inlier_all,
                    self.config.bins,
                    data_min,
                    data_max,
                    self.colors,
                    self.config.path,
                )
            if self.options.plot_gauss:
                plot_overlay_gauss(
                    fid,
                    fname,
                    data_inlier_all,
                    gauss_inl,
                    x,
                    self.colors,
                    self.config.path,
                )
            if self.options.plot_weibull:
                plot_overlay_weibull(
                    fid, fname, data_inlier_all, x, self.colors, self.config.path
                )
            if self.options.plot_box:
                plot_overlay_boxplot(
                    fid, fname, data_inlier_all, self.colors, self.config.path
                )
            if self.options.plot_qq:
                plot_overlay_qq(
                    fid, fname, data_inlier_all, self.colors, self.config.path
                )
            if self.options.plot_grouped_bar:
                plot_grouped_bar_means_stds_dual(
                    fid,
                    fname,
                    data_with_all,
                    data_inlier_all,
                    self.colors,
                    self.config.path,
                )
            if self.options.plot_violin:
                plot_overlay_violin(
                    fid, fname, data_inlier_all, self.colors, self.config.path
                )
            logger.info("[Report] PNGs für %s (INLIER) erzeugt.", fid)
        else:
            logger.warning(
                "[Report] Keine INLIER-Daten gefunden – zweite Seite bleibt leer."
            )


def overlay_plots(config: PlotConfig, options: PlotOptions) -> None:
    """Create combined overlay plots for a set of distance files."""
    ReportBuilder(config, options).overlay_plots()


def summary_pdf(config: PlotConfig) -> None:
    """Assemble a PDF summary from previously generated plot PNGs.

    Parameters
    ----------
    config : PlotConfig
        Configuration describing where plot images are stored. The ``path``
        attribute must point to a directory containing the PNG files and is
        also used as the destination for the resulting PDF.

    Produces
    --------
    A two-page document named ``ALLFOLDERS_comparison_report.pdf`` located in
    ``config.path``. Each page arranges the PNG files
    ``ALLFOLDERS_ALL_WITH_<suffix>.png`` and
    ``ALLFOLDERS_ALL_INLIER_<suffix>.png`` for supported plot types (overlay
    histograms, Gaussian and Weibull fits, box plots, Q-Q plots and grouped
    bar charts) into a 2×3 grid.
    """
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
    """Combine per-part plots into a multi-page PDF.

    Parameters
    ----------
    outdir : str
        Directory containing the PNG plots for each part.
    pdf_path : str, optional
        Destination file for the summary PDF. Defaults to
        ``"parts_summary.pdf"`` inside ``outdir`` if not provided.
    include_with : bool, default True
        Include plots that were generated with outliers ("WITH").
    include_inlier : bool, default True
        Include plots generated without outliers ("INLIER"). Exactly one of
        ``include_with`` or ``include_inlier`` must be ``True``.

    Returns
    -------
    str
        Path to the created PDF or an empty string if no plot images were
        found.
    """
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
    """Merge multiple PDF files into a single document.

    Parameters
    ----------
    pdf_paths : List[str]
        Paths to the individual PDF files to merge. Missing files are skipped
        and a warning is logged.
    out_path : str
        Destination path for the merged PDF. Parent directories are created if
        necessary.

    Returns
    -------
    str
        The filesystem path of the resulting merged PDF.

    Raises
    ------
    RuntimeError
        If :mod:`PyPDF2` is not available.
    """
    try:
        from PyPDF2 import PdfMerger
    except ImportError as e:
        raise RuntimeError(
            "PyPDF2 is required for merging PDFs but is not installed"
        ) from e

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
