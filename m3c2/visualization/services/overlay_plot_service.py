"""Overlay plot generation helpers.

This module provides utility functions for creating overlay plots from
M3C2 distance data. The functions were previously part of
:mod:`plot_service` but were extracted to keep the façade lean.
"""

from __future__ import annotations

import logging
import os
from collections import OrderedDict
from typing import Dict, List, Tuple

import numpy as np
from scipy.stats import norm

from m3c2.config.plot_config import PlotOptions
from ..loaders.distance_loader import scan_distance_files_by_index
from ..plotters.overlay_plotter import (
    get_common_range,
    plot_overlay_boxplot,
    plot_overlay_gauss,
    plot_overlay_histogram,
    plot_overlay_qq,
    plot_overlay_weibull,
)

logger = logging.getLogger(__name__)

# Stable ordering of cases when arranging labels
CASE_ORDER = ("CASE1", "CASE2", "CASE3", "CASE4")


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _labels_by_case_map(case_map: Dict[str, str], case_order: Tuple[str, ...] | None = None) -> List[str]:
    """Order labels according to a mapping of labels to cases.

    Parameters
    ----------
    case_map:
        Mapping from label names to a case identifier.
    case_order:
        Optional sequence describing the desired order of cases. If not
        provided, :data:`CASE_ORDER` is used.

    Returns
    -------
    list[str]
        Labels sorted first by the order of their case as specified by
        ``case_order`` and within each case by their insertion order in
        ``case_map``.
    """
    order = case_order or CASE_ORDER
    labels: List[str] = []
    for c in order:
        labels.extend([lbl for lbl, cas in case_map.items() if cas == c])
    return labels


def _reorder_data(data: Dict[str, np.ndarray], labels_order: List[str]) -> "OrderedDict[str, np.ndarray]":
    """Return ``data`` sorted according to ``labels_order``.

    An :class:`~collections.OrderedDict` is constructed whose items follow
    the sequence given by ``labels_order``. Only labels present in
    ``data`` are included, ensuring that downstream plotting receives the
    arrays in a predictable order.
    """
    return OrderedDict((lbl, data[lbl]) for lbl in labels_order if lbl in data)


def _colors_by_case(labels_order: List[str], label_to_case: Dict[str, str], case_colors: Dict[str, str]) -> Dict[str, str]:
    """Map labels to plotting colors based on their assigned case.

    Each ``label`` in ``labels_order`` is looked up in ``label_to_case`` to
    determine the case it belongs to. The corresponding color is then
    retrieved from ``case_colors``. If a label has no case mapping or the
    case has no associated color, a default gray (``"#777777"``) is used.
    """
    return {lbl: case_colors.get(label_to_case.get(lbl, "CASE1"), "#777777") for lbl in labels_order}


def _plot_grouped_bar_means_stds_dual_by_case(
    fid: str,
    data_with: Dict[str, np.ndarray],
    data_inlier: Dict[str, np.ndarray],
    colors: Dict[str, str],
    outdir: str,
    title_text: str = "Means & Std – Incl. vs. Excl. Outliers",
    labels_order: List[str] | None = None,
) -> None:
    """Plot grouped bar charts comparing mean and standard deviation."""
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
    import matplotlib.pyplot as plt  # lazy import for headless environments

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


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def overlay_from_data(data: Dict[str, np.ndarray], outdir: str, bins: int = 256) -> List[str]:
    """Generate overlay plots for arbitrary distance arrays.

    Parameters
    ----------
    data:
        Mapping of labels to numeric arrays.
    outdir:
        Directory where resulting PNG images will be written.
    bins:
        Number of histogram bins. Defaults to ``256``.

    Returns
    -------
    list[str]
        Paths to the generated image files.
    """

    if len(data) < 2:
        raise ValueError("At least two datasets are required for overlay plots")

    os.makedirs(outdir, exist_ok=True)

    labels = list(data.keys())
    colors = {lbl: col for lbl, col in zip(labels, ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"])}

    data_min, data_max, x = get_common_range(data)
    gauss = {k: norm.fit(v) for k, v in data.items()}

    fid, fname = "Part_0", "WITH"
    title = " vs ".join(labels)

    plot_overlay_histogram(
        fid,
        fname,
        data,
        bins,
        data_min,
        data_max,
        colors,
        outdir,
        labels_order=labels,
        title_text=f"Histogram – {title}",
    )
    plot_overlay_gauss(
        fid,
        fname,
        data,
        gauss,
        x,
        colors,
        outdir,
        labels_order=labels,
        title_text=f"Gauss-Fit – {title}",
    )
    plot_overlay_weibull(
        fid,
        fname,
        data,
        x,
        colors,
        outdir,
        labels_order=labels,
        title_text=f"Weibull-Fit – {title}",
    )
    plot_overlay_boxplot(
        fid,
        fname,
        data,
        colors,
        outdir,
        labels_order=labels,
        title_text=f"Boxplot – {title}",
    )
    plot_overlay_qq(
        fid,
        fname,
        data,
        colors,
        outdir,
        labels_order=labels,
        title_text=f"Q-Q-Plot – {title}",
    )

    files = [
        os.path.join(outdir, f"{fid}_{fname}_OverlayHistogramm.png"),
        os.path.join(outdir, f"{fid}_{fname}_OverlayGaussFits.png"),
        os.path.join(outdir, f"{fid}_{fname}_OverlayWeibullFits.png"),
        os.path.join(outdir, f"{fid}_{fname}_Boxplot.png"),
        os.path.join(outdir, f"{fid}_{fname}_QQPlot.png"),
    ]
    return files


def overlay_by_index(
    data_dir: str,
    outdir: str,
    versions=("python",),
    bins: int = 256,
    options: PlotOptions | None = None,
    skip_existing: bool = True,
) -> None:
    """Scan *data_dir* and create overlay plots grouped by part index."""
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
            labels_w = _labels_by_case_map(case_map_w)
            data_with = _reorder_data(data_with, labels_w)
            colors_w = _colors_by_case(labels_w, case_map_w, case_colors)
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
            labels_i = _labels_by_case_map(case_map_i)
            data_inl = _reorder_data(data_inl, labels_i)
            colors_i = _colors_by_case(labels_i, case_map_i, case_colors)
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
            labels_dual = _labels_by_case_map(combined_case_map)
            colors_dual = _colors_by_case(labels_dual, combined_case_map, case_colors)
            _plot_grouped_bar_means_stds_dual_by_case(
                fid=f"Part_{i}",
                data_with=per_index[i]["WITH"],
                data_inlier=per_index[i]["INLIER"],
                colors=colors_dual,
                outdir=outdir,
                title_text="Means & Std – WITH vs INLIER",
                labels_order=labels_dual,
            )
