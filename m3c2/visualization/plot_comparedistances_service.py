"""Facade for generating comparison plots between reference variants.

This module exposes :class:`PlotServiceCompareDistances` which orchestrates the
loader utilities and the individual plotter modules.  The heavy lifting of
loading data and creating plots is delegated to dedicated modules to keep this
facade light-weight and easy to extend.
"""

from __future__ import annotations

import logging
import os

from m3c2.config.plot_config import PlotConfig, PlotOptionsComparedistances

from .bland_altman_plotter import plot as bland_altman_plot
from .linear_regression_plotter import plot as linear_regression_plot
from .passing_bablok_plotter import plot as passing_bablok_plot

logger = logging.getLogger(__name__)


class PlotServiceCompareDistances:
    """High level service coordinating comparison distance plots."""

    @classmethod
    def overlay_plots(
        cls, config: PlotConfig, options: PlotOptionsComparedistances
    ) -> None:
        """Generate the requested comparison plots.

        Parameters
        ----------
        config:
            Overall plot configuration describing input files and output
            directories.
        options:
            Flags selecting which plot types should be produced.
        """

        os.makedirs(config.path, exist_ok=True)
        folder_ids = config.folder_ids
        ref_variants = config.filenames

        if options.plot_blandaltman:
            logger.info("Generating Bland-Altman plots...")
            bland_altman_plot(folder_ids, ref_variants, outdir=config.path)

        if options.plot_passingbablok:
            logger.info("Generating Passing-Bablok plots...")
            passing_bablok_plot(folder_ids, ref_variants, outdir=config.path)

        if options.plot_linearregression:
            logger.info("Generating Linear Regression plots...")
            linear_regression_plot(folder_ids, ref_variants, outdir=config.path)


__all__ = ["PlotServiceCompareDistances"]

