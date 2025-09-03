"""Facade service for distance comparison plots.

This module serves as a façade orchestrating multiple plot types—
Bland-Altman, Passing-Bablok, and linear regression—to compare distance
measurements.
"""

from __future__ import annotations

import logging
import os

from m3c2.config.plot_config import PlotConfig, PlotOptionsComparedistances
from ..plotters.bland_altman_plotter import bland_altman_plot
from ..plotters.passing_bablok_plotter import passing_bablok_plot
from ..plotters.linear_regression_plotter import linear_regression_plot

logger = logging.getLogger(__name__)


class PlotServiceCompareDistances:
    """High level façade for creating distance comparison plots.

    The service exposes a single entry point that coordinates several
    specialized plotting routines.  Based on the provided
    :class:`~m3c2.config.plot_config.PlotConfig`, it resolves where the
    output should be stored (``config.path``) and which distance datasets
    should be compared (``config.folder_ids`` and ``config.filenames``).

    The generated diagrams are controlled by
    :class:`~m3c2.config.plot_config.PlotOptionsComparedistances`.  Setting
    ``plot_blandaltman``, ``plot_passingbablok`` and/or
    ``plot_linearregression`` toggles the respective Bland–Altman,
    Passing–Bablok, and linear regression plots.
    """

    @classmethod
    def overlay_plots(cls, config: PlotConfig, options: PlotOptionsComparedistances) -> None:
        """Create comparison plots for distance measurements.

        Parameters
        ----------
        config : PlotConfig
            Configuration providing the output ``path`` as well as the
            ``folder_ids`` and ``filenames`` used to locate the data.
        options : PlotOptionsComparedistances
            Flags controlling which optional plots to generate. Enable
            ``plot_blandaltman``, ``plot_passingbablok`` and/or
            ``plot_linearregression`` to create the respective diagrams.
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
