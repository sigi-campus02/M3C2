"""Entry point for visualization utilities.
This script provides a thin wrapper around the plotting CLI so that
visualisations can be triggered directly via ``python -m main_plotter_onefolder``.
"""

from __future__ import annotations
import logging
from m3c2.config.logging_config import setup_logging
from pathlib import Path

from m3c2.visualization.services.plot_service import PlotService

logger = logging.getLogger(__name__)


def main() -> None:
    """Execute the plotting command line interface."""

    setup_logging()
    logger.info("Starting plotter CLI")

    from m3c2.cli import overlay_report

    parser = overlay_report.build_arg_parser_onefolder()

    args = parser.parse_args()
    logger.debug("Arguments: %s", args)

    folder_arg = args.folder or getattr(args, "folder", None)
    overlay_outdir_arg = args.overlay_outdir or getattr(args, "outdir", None)
    plot_types_arg = args.plot_types or getattr(args, "plot_types", None)

    if folder_arg is None or overlay_outdir_arg is None:
        parser.error("folder and overlay_outdir are required (via CLI or config)")

    logger.info("Generating overlay report in %s", overlay_outdir_arg)
    logger.info("Using distance files: %s", folder_arg)

    folder = Path(folder_arg).expanduser().resolve()
    overlay_outdir = Path(overlay_outdir_arg).expanduser().resolve()

    if not overlay_outdir.exists():
        overlay_outdir.mkdir(parents=True, exist_ok=True)
        logger.info("Created output directory %s", overlay_outdir)

    PlotService.overlay_by_index(
        data_dir=folder,
        outdir=overlay_outdir,
        options=plot_types_arg
    )

if __name__ == "__main__":
    main()

