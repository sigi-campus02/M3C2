"""High-level façade for plotting and report generation.

The :class:`PlotService` class merely delegates to specialised helper
modules. Overlay plot creation lives in :mod:`overlay_plot_service`
whereas PDF assembly is handled by :mod:`report_builder`.
"""

from __future__ import annotations

import logging
from typing import List

from m3c2.config.plot_config import PlotConfig, PlotOptions

from .overlay_plot_service import overlay_by_index as _overlay_by_index
from .overlay_plot_service import overlay_from_data as _overlay_from_data
from .report_service import ReportBuilder
from .report_builder import (
    build_parts_pdf as _build_parts_pdf,
    merge_pdfs as _merge_pdfs,
    summary_pdf as _summary_pdf,
)

logger = logging.getLogger(__name__)


class PlotService:
    """Facade exposing plot and report helpers."""

    @staticmethod
    def overlay_by_index(
        data_dir: str,
        outdir: str,
        versions=("python",),
        bins: int = 256,
        options: PlotOptions | None = None,
        skip_existing: bool = True,
    ) -> None:
        """Scan ``data_dir`` and create overlay plots grouped by part index."""
        _overlay_by_index(
            data_dir,
            outdir,
            versions=versions,
            bins=bins,
            options=options,
            skip_existing=skip_existing,
        )

    @staticmethod
    def overlay_from_data(
        data_dir: str,
        outdir: str,
        bins: int = 256
    ) -> None:
        _overlay_from_data(
            data_dir,
            outdir,
            bins=bins
        )

    @staticmethod
    def overlay_plots(config: PlotConfig, options: PlotOptions) -> None:
        """Create overlay plots via :class:`~ReportBuilder`."""
        builder = ReportBuilder(config, options)
        builder.build()

    @staticmethod
    def summary_pdf(config: PlotConfig) -> None:
        """Aggregate individual plot images into a comparison PDF."""
        _summary_pdf(config)

    @staticmethod
    def build_parts_pdf(
        outdir: str,
        pdf_path: str | None = None,
        include_with: bool = True,
        include_inlier: bool = True,
    ) -> str:
        """Create a consolidated PDF report for all parts."""
        return _build_parts_pdf(
            outdir,
            pdf_path=pdf_path,
            include_with=include_with,
            include_inlier=include_inlier,
        )

    @staticmethod
    def merge_pdfs(pdf_paths: List[str], out_path: str) -> str:
        """Merge several PDF files into a single document."""
        return _merge_pdfs(pdf_paths, out_path)
