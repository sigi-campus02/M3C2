from __future__ import annotations

import logging
import os
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import norm

from m3c2.config.plot_config import PlotConfig, PlotOptions
from .distance_loader import scan_distance_files_by_index
from .overlay_plotter import (
    get_common_range,
    plot_overlay_boxplot,
    plot_overlay_gauss,
    plot_overlay_histogram,
    plot_overlay_qq,
    plot_overlay_violin,
    plot_overlay_weibull,
)
from .report_builder import (
    overlay_plots as _overlay_plots,
    summary_pdf as _summary_pdf,
    build_parts_pdf as _build_parts_pdf,
    merge_pdfs as _merge_pdfs,
)

logger = logging.getLogger(__name__)


class PlotService:
    """Facade for all plotting and report generation tasks."""

    CASE_ORDER = ("CASE1", "CASE2", "CASE3", "CASE4")

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @staticmethod
    def _labels_by_case_map(case_map: Dict[str, str], case_order: Tuple[str, ...] | None = None) -> List[str]:
        order = case_order or PlotService.CASE_ORDER
        labels: List[str] = []
        for c in order:
            labels.extend([lbl for lbl, cas in case_map.items() if cas == c])
        return labels

    @staticmethod
    def _reorder_data(data: Dict[str, np.ndarray], labels_order: List[str]) -> "OrderedDict[str, np.ndarray]":
        return OrderedDict((lbl, data[lbl]) for lbl in labels_order if lbl in data)

    @staticmethod
    def _colors_by_case(labels_order: List[str], label_to_case: Dict[str, str], case_colors: Dict[str, str]) -> Dict[str, str]:
        return {lbl: case_colors.get(label_to_case.get(lbl, "CASE1"), "#777777") for lbl in labels_order}

    @staticmethod
    def _plot_grouped_bar_means_stds_dual_by_case(
        fid: str,
        data_with: Dict[str, np.ndarray],
        data_inlier: Dict[str, np.ndarray],
        colors: Dict[str, str],
        outdir: str,
        title_text: str = "Means & Std – Incl. vs. Excl. Outliers",
        labels_order: List[str] | None = None,
    ) -> None:
        labels = labels_order or list(dict.fromkeys(list(data_with.keys()) + list(data_inlier.keys())))
        means_with, stds_with, means_inl, stds_inl, bar_colors = [], [], [], [], []
        for lbl in labels:
            arr_w = data_with.get(lbl, np.array([]))
            arr_i = data_inlier.get(lbl, np.array([]))
            m_w = float(np.abs(np.mean(arr_w))) if arr_w.size else np.nan
            s_w = float(np.std(arr_w)) if arr_w.size else np.nan
            m_i = float(np.abs(np.mean(arr_i))) if arr_i.size else np.nan
            s_i = float(np.std(arr_i)) if arr_i.size else np.nan
            means_with.append(m_w)
            stds_with.append(s_w)
            means_inl.append(m_i)
            stds_inl.append(s_i)
            bar_colors.append(colors.get(lbl, "#8aa2ff"))

        x = np.arange(len(labels))
        width = 0.4
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(2, 1, figsize=(max(10, len(labels) * 1.8), 8), sharex=True)
        ax[0].bar(x - width / 2, means_with, width, label="incl. outliers", color=bar_colors)
        ax[0].bar(x + width / 2, means_inl, width, label="excl. outliers", color=bar_colors, alpha=0.55)
        ax[0].set_ylabel("|μ|")
        ax[0].set_title(f"{title_text} – {fid}")
        ax[0].set_ylim(bottom=0)
        ax[0].legend()
        ax[1].bar(x - width / 2, stds_with, width, label="incl. outliers", color=bar_colors)
        ax[1].bar(x + width / 2, stds_inl, width, label="excl. outliers", color=bar_colors, alpha=0.55)
        ax[1].set_ylabel("σ")
        ax[1].set_title(f"Std. deviation – {fid}")
        ax[1].set_xticks(x)
        ax[1].set_xticklabels(labels, rotation=30, ha="right")
        ax[1].set_ylim(bottom=0)
        ax[1].legend()
        plt.tight_layout()
        out = os.path.join(outdir, f"{fid}_DUAL_GroupedBar_Mean_Std.png")
        plt.savefig(out)
        plt.close()
        logger.info("[Report] Saved grouped bar: %s", out)

    # ------------------------------------------------------------------
    # Delegated high-level methods
    # ------------------------------------------------------------------
    @classmethod
    def overlay_by_index(
        cls,
        data_dir: str,
        outdir: str,
        versions=("python",),
        bins: int = 256,
        options: PlotOptions | None = None,
        skip_existing: bool = True,
    ) -> None:
        """Scan *data_dir* and create overlay plots grouped by part index.

        Distance files are discovered with
        :func:`scan_distance_files_by_index` and split into data sets that
        include outliers (``WITH``) and those restricted to inliers
        (``INLIER``).  For each part index the method reorders the data by
        case, assigns stable colours, determines a common range where
        necessary and writes the requested plots to *outdir*.

        Depending on ``options`` histograms with optional Gaussian or Weibull
        fits, box plots, Q–Q plots and grouped bar charts of means and
        standard deviations are generated for the available data sets.  The
        output directory is created if required and the function returns
        silently when no suitable files are found.
        """

        options = options or PlotOptions()
        os.makedirs(outdir, exist_ok=True)
        per_index, case_colors = scan_distance_files_by_index(data_dir, versions=versions)
        if not per_index:
            logger.warning("[Report] Keine Distanzdateien gefunden in %s.", data_dir)
            return
        for i in sorted(per_index.keys()):
            fid = f"Part_{i}"
            data_with = per_index[i]["WITH"]
            if data_with:
                case_map_w = per_index[i]["CASE_WITH"]
                labels_w = cls._labels_by_case_map(case_map_w)
                data_with = cls._reorder_data(data_with, labels_w)
                colors_w = cls._colors_by_case(labels_w, case_map_w, case_colors)
                need_range = options.plot_hist or options.plot_gauss or options.plot_weibull
                if need_range:
                    data_min, data_max, x = get_common_range(data_with)
                gauss_with = {k: norm.fit(v) for k, v in data_with.items()} if options.plot_gauss else {}
                if options.plot_hist:
                    plot_overlay_histogram(
                        fid,
                        "WITH",
                        data_with,
                        bins,
                        data_min,
                        data_max,
                        colors_w,
                        outdir,
                        title_text=f"Histogram – Part {i} / incl. Outliers",
                        labels_order=labels_w,
                    )
                if options.plot_gauss:
                    plot_overlay_gauss(
                        fid,
                        "WITH",
                        data_with,
                        gauss_with,
                        x,
                        colors_w,
                        outdir,
                        title_text=f"Gaussian fit – Part {i} / incl. Outliers",
                        labels_order=labels_w,
                    )
                if options.plot_weibull:
                    plot_overlay_weibull(
                        fid,
                        "WITH",
                        data_with,
                        x,
                        colors_w,
                        outdir,
                        title_text=f"Weibull fit – Part {i} / incl. Outliers",
                        labels_order=labels_w,
                    )
                if options.plot_box:
                    plot_overlay_boxplot(
                        fid,
                        "WITH",
                        data_with,
                        colors_w,
                        outdir,
                        title_text=f"Box plot – Part {i} / incl. Outliers",
                        labels_order=labels_w,
                    )
                if options.plot_qq:
                    plot_overlay_qq(
                        fid,
                        "WITH",
                        data_with,
                        colors_w,
                        outdir,
                        title_text=f"Q–Q plot – Part {i} / incl. Outliers",
                        labels_order=labels_w,
                    )
            data_inl = per_index[i]["INLIER"]
            if data_inl:
                case_map_i = per_index[i]["CASE_INLIER"]
                labels_i = cls._labels_by_case_map(case_map_i)
                data_inl = cls._reorder_data(data_inl, labels_i)
                colors_i = cls._colors_by_case(labels_i, case_map_i, case_colors)
                need_range = options.plot_hist or options.plot_gauss or options.plot_weibull
                if need_range:
                    data_min, data_max, x = get_common_range(data_inl)
                gauss_inl = {k: norm.fit(v) for k, v in data_inl.items()} if options.plot_gauss else {}
                if options.plot_hist:
                    plot_overlay_histogram(
                        fid,
                        "INLIER",
                        data_inl,
                        bins,
                        data_min,
                        data_max,
                        colors_i,
                        outdir,
                        title_text=f"Histogram – Part {i} / excl. Outliers",
                        labels_order=labels_i,
                    )
                if options.plot_gauss:
                    plot_overlay_gauss(
                        fid,
                        "INLIER",
                        data_inl,
                        gauss_inl,
                        x,
                        colors_i,
                        outdir,
                        title_text=f"Gaussian fit – Part {i} / excl. Outliers",
                        labels_order=labels_i,
                    )
                if options.plot_weibull:
                    plot_overlay_weibull(
                        fid,
                        "INLIER",
                        data_inl,
                        x,
                        colors_i,
                        outdir,
                        title_text=f"Weibull fit – Part {i} / excl. Outliers",
                        labels_order=labels_i,
                    )
                if options.plot_box:
                    plot_overlay_boxplot(
                        fid,
                        "INLIER",
                        data_inl,
                        colors_i,
                        outdir,
                        title_text=f"Box plot – Part {i} / excl. Outliers",
                        labels_order=labels_i,
                    )
                if options.plot_qq:
                    plot_overlay_qq(
                        fid,
                        "INLIER",
                        data_inl,
                        colors_i,
                        outdir,
                        title_text=f"Q–Q plot – Part {i} / excl. Outliers",
                        labels_order=labels_i,
                    )
            if options.plot_grouped_bar and per_index[i]["WITH"] and per_index[i]["INLIER"]:
                combined_case_map = {**per_index[i]["CASE_WITH"], **per_index[i]["CASE_INLIER"]}
                labels_dual = cls._labels_by_case_map(combined_case_map)
                colors_dual = cls._colors_by_case(labels_dual, combined_case_map, case_colors)
                cls._plot_grouped_bar_means_stds_dual_by_case(
                    fid=f"Part_{i}",
                    data_with=per_index[i]["WITH"],
                    data_inlier=per_index[i]["INLIER"],
                    colors=colors_dual,
                    outdir=outdir,
                    title_text="Means & Std – WITH vs INLIER",
                    labels_order=labels_dual,
                )

    @staticmethod
    def overlay_plots(config: PlotConfig, options: PlotOptions) -> None:
        """Wrapper exposing :func:`report_builder.overlay_plots`.

        The heavy lifting is done by ``report_builder.overlay_plots``; this
        method merely forwards the call so that overlay plotting is available
        via :class:`PlotService`.
        """
        _overlay_plots(config, options)

    @staticmethod
    def summary_pdf(config: PlotConfig) -> None:
        """Aggregate individual plot images into a comparison PDF.

        Collects the previously generated overlay plots in ``config.path`` and
        merges them into a multi-page PDF summarizing the results for all
        datasets.
        """
        _summary_pdf(config)

    @staticmethod
    def build_parts_pdf(
        outdir: str,
        pdf_path: str | None = None,
        include_with: bool = True,
        include_inlier: bool = True,
    ) -> str:
        """Create a consolidated PDF report for all parts.

        Parameters
        ----------
        outdir:
            Directory containing per-part plots and PDFs.
        pdf_path:
            Optional destination path for the combined PDF. If ``None``, a
            default name within ``outdir`` is used.
        include_with:
            Whether sections including outliers should be appended.
        include_inlier:
            Whether sections with inlier-only data should be appended.

        Returns
        -------
        str
            Path to the generated PDF file.
        """
        return _build_parts_pdf(outdir, pdf_path=pdf_path, include_with=include_with, include_inlier=include_inlier)

    @staticmethod
    def merge_pdfs(pdf_paths: List[str], out_path: str) -> str:
        """Merge several PDF files into a single document.

        Args:
            pdf_paths: List of paths to the PDF files to be merged. Nonexistent
                files are skipped.
            out_path: Destination path where the merged PDF will be stored.

        Returns:
            The path to the merged PDF file.
        """
        return _merge_pdfs(pdf_paths, out_path)
