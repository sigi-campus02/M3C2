"""Entry point for visualization utilities.

This script provides a thin wrapper around the plotting CLI so that
visualisations can be triggered directly via ``python -m main_plotter``.
"""

from __future__ import annotations

import logging

from m3c2.config.logging_config import setup_logging
from pathlib import Path

logger = logging.getLogger(__name__)


def main() -> None:
    """Execute the plotting command line interface."""

    setup_logging()
    logger.info("Starting plotter CLI")

    from m3c2.cli import overlay_report
    parser = overlay_report.build_arg_parser()
    args = parser.parse_args()
    logger.debug(f"Arguments: {args}")
    logger.info(f"Generating overlay report in {args.overlay_outdir}")
    logger.info(f"Using distance files: {args.overlay_files}")

    overlay_outdir = Path(args.overlay_outdir).expanduser().resolve()
    overlay_files = [Path(f).expanduser().resolve() for f in args.overlay_files]
    if not overlay_outdir.exists():
        overlay_outdir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created output directory {overlay_outdir}")

    overlay_report.main(overlay_files, overlay_outdir)


if __name__ == "__main__":
    main()

