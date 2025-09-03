"""Create comparison plots between reference variants of point cloud data.

The script configures :class:`PlotServiceCompareDistances` to overlay various
statistical plots (Bland–Altman, Passing–Bablok, and linear regression) for the
specified folders and reference variants.
"""
import logging
import sys
import os

from m3c2.io.logging_utils import setup_logging

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
    """Configure and create distance comparison plots.

    Parameters
    ----------
    folder_ids : list[str] | None, optional
        Identifiers of point cloud folders to process. Defaults to
        ``["0342-0349"]`` when ``None``.
    ref_variants : list[str] | None, optional
        Reference variants to compare. Defaults to ``["ref", "ref_ai"]`` when
        ``None``.
    outdir : str, optional
        Target directory for generated plots. Defaults to ``"outputs"``.

    Logging
    -------
    Configures logging via :func:`setup_logging` at a level resolved by
    :func:`resolve_log_level`.

    Side Effects
    ------------
    Creates statistical comparison plots and writes them to ``outdir`` while
    emitting log messages.
    """

    # ``setup_logging`` reads the desired log level from configuration and the
    # environment, so no explicit level argument is required here.
    setup_logging()

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

    logger.info("Starting plot generation %s, %s", cfg, opts)
    PlotServiceCompareDistances.overlay_plots(cfg, opts)


if __name__ == "__main__":
    main()
