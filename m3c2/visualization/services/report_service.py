"""Orchestration helpers for building overlay plot reports.

The :class:`ReportBuilder` class glues together data loading and plotting
routines.  Heavy computations and rendering are delegated to specialised
modules which keep this layer focused on the overall workflow.
"""

from __future__ import annotations

import logging
import os
from typing import Dict

import numpy as np
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
from .data_loader import load_data, resolve_path

logger = logging.getLogger(__name__)


class ReportBuilder:
    """Coordinate loading of data and creation of overlay plots."""

    def __init__(self, config: PlotConfig, options: PlotOptions) -> None:
        self.config = config
        self.options = options
        self.colors = config.ensure_colors()
        os.makedirs(config.path, exist_ok=True)

    # ------------------------------------------------------------------
    # High level orchestration
    # ------------------------------------------------------------------
    def build(self) -> None:
        """Generate all requested plots for the configured data set."""

        data_with_all: Dict[str, np.ndarray] = {}
        for fid in self.config.folder_ids:
            data_with, _ = load_data(fid, self.config.filenames, self.config.versions)
            if not data_with:
                logger.warning("[Report] Keine WITH-Daten für %s gefunden.", fid)
                continue
            data_with_all.update(data_with)

        if not data_with_all:
            logger.warning("[Report] Keine Daten gefunden – keine Plots erzeugt.")
            return

        data_inlier_all: Dict[str, np.ndarray] = {}
        for fid in self.config.folder_ids:
            for v in self.config.versions:
                label = f"{v}_{fid}"
                base_inl = (
                    f"{v}_Job_0378_8400-110-rad-{fid}_cloud_comparison_m3c2_distances_coordinates_inlier_std.txt"
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
            plot_overlay_gauss(fid, fname, data_with_all, gauss_with, x, self.colors, self.config.path)
        if self.options.plot_weibull:
            plot_overlay_weibull(fid, fname, data_with_all, x, self.colors, self.config.path)
        if self.options.plot_box:
            plot_overlay_boxplot(fid, fname, data_with_all, self.colors, self.config.path)
        if self.options.plot_qq:
            plot_overlay_qq(fid, fname, data_with_all, self.colors, self.config.path)
        if self.options.plot_grouped_bar:
            logger.warning("Grouped bar plotting is no longer supported in this module")
        if self.options.plot_violin:
            plot_overlay_violin(fid, fname, data_with_all, self.colors, self.config.path)
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
                plot_overlay_gauss(fid, fname, data_inlier_all, gauss_inl, x, self.colors, self.config.path)
            if self.options.plot_weibull:
                plot_overlay_weibull(fid, fname, data_inlier_all, x, self.colors, self.config.path)
            if self.options.plot_box:
                plot_overlay_boxplot(fid, fname, data_inlier_all, self.colors, self.config.path)
            if self.options.plot_qq:
                plot_overlay_qq(fid, fname, data_inlier_all, self.colors, self.config.path)
            if self.options.plot_grouped_bar:
                logger.warning("Grouped bar plotting is no longer supported in this module")
            if self.options.plot_violin:
                plot_overlay_violin(fid, fname, data_inlier_all, self.colors, self.config.path)
            logger.info("[Report] PNGs für %s (INLIER) erzeugt.", fid)
        else:
            logger.warning("[Report] Keine INLIER-Daten gefunden – zweite Seite bleibt leer.")
