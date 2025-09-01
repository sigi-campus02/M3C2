"""Create comparison plots between reference variants of point cloud data.

The script configures :class:`PlotServiceCompareDistances` to overlay various
statistical plots (Bland–Altman, Passing–Bablok, and linear regression) for the
specified folders and reference variants.
"""
import logging
import sys
import os

logger = logging.getLogger(__name__)

# Allow absolute imports when the script is executed directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from m3c2.visualization.plot_comparedistances_service import (
    PlotServiceCompareDistances,
)
from m3c2.config.plot_config import PlotConfig, PlotOptionsComparedistances


def main(
    folder_ids: list[str] | None = None,
    ref_variants: list[str] | None = None,
    outdir: str = "outputs",
) -> None:
    """Configure and create distance comparison plots."""

    folder_ids = folder_ids or ["0342-0349"]
    ref_variants = ref_variants or ["ref", "ref_ai"]

    cfg = PlotConfig(
        folder_ids=folder_ids,
        filenames=ref_variants,
        bins=256,
        outdir=outdir,
        project="MARS",
    )
    opts = PlotOptionsComparedistances(
        plot_blandaltman=True,
        plot_passingbablok=True,
        plot_linearregression=True,
    )

    logger.info(
        "Starting plot generation for folders %s with reference variants %s",
        folder_ids,
        ref_variants,
    )
    try:
        PlotServiceCompareDistances.overlay_plots(cfg, opts)
        logger.info("Plot generation completed successfully")
    except Exception:
        logger.exception("Plot generation failed")
        raise


if __name__ == "__main__":
    main()
