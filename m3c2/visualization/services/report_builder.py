"""Utilities for assembling plot images into PDF reports.

This module bundles helper functions that collect previously generated
plots and merge them into PDF documents. The functions were extracted
from :mod:`plot_service` to keep the façade slim.
"""

from __future__ import annotations

import logging
import os
import re
from typing import List

import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from m3c2.config.plot_config import PlotConfig

logger = logging.getLogger(__name__)


def summary_pdf(config: PlotConfig) -> None:
    """Assemble a PDF summary from previously generated plot PNGs.

    Parameters
    ----------
    config : PlotConfig
        Configuration describing where plot images are stored. The ``path``
        attribute must point to a directory containing the PNG files and is
        also used as the destination for the resulting PDF.
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

    Returns the path to the created PDF or an empty string if no plot images
    were found.
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
    """Merge multiple PDF files into a single document."""
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
