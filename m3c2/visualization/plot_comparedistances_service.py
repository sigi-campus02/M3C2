from __future__ import annotations

import logging
import os

from m3c2.config.plot_config import PlotConfig, PlotOptionsComparedistances
from .bland_altman_plotter import bland_altman_plot
from .passing_bablok_plotter import passing_bablok_plot
from .linear_regression_plotter import linear_regression_plot

logger = logging.getLogger(__name__)


class PlotServiceCompareDistances:
    """Facade orchestrating distance comparison plots."""

    @classmethod
    def overlay_plots(cls, config: PlotConfig, options: PlotOptionsComparedistances) -> None:
        os.makedirs(config.path, exist_ok=True)
        folder_ids = config.folder_ids
        ref_variants = config.filenames

        logger.info(
            "Generating comparison plots for folders %s with reference variants %s",
            folder_ids,
            ref_variants,
        )
        try:
            if options.plot_blandaltman:
                logger.info("Generating Bland-Altman plots...")
                bland_altman_plot(folder_ids, ref_variants, outdir=config.path)
            if options.plot_passingbablok:
                logger.info("Generating Passing-Bablok plots...")
                passing_bablok_plot(folder_ids, ref_variants, outdir=config.path)
            if options.plot_linearregression:
                logger.info("Generating Linear Regression plots...")
                linear_regression_plot(folder_ids, ref_variants, outdir=config.path)
        except Exception:
            logger.exception("Failed to generate comparison plots")
            raise
        else:
            logger.info("Successfully generated comparison plots")
