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
    parser = overlay_report.build_arg_parser_severalfolders()
    args = parser.parse_args()
    logger.debug("Arguments: %s", args)

    data_dir_arg = args.data_dir or getattr(args, "dir", None)
    folders_arg = args.folders or getattr(args, "folders", None)
    filename_reference_arg = args.filename_reference or getattr(args, "filename_reference", None)
    filename_comparison_arg = args.filename_comparison or getattr(args, "filename_comparison", None)
    overlay_outdir_arg = args.overlay_outdir or getattr(args, "outdir", None)
    options_arg = args.options or getattr(args, "options", None)

    if data_dir_arg is None or folders_arg is None or filename_reference_arg is None or filename_comparison_arg is None or overlay_outdir_arg is None:
        parser.error("data_dir, folders, filename_reference, filename_comparison, and overlay_outdir are required (via CLI or config)")

    logger.info("Generating overlay report in %s", overlay_outdir_arg)
    logger.info("Using folders: %s", folders_arg)
    logger.info("Using reference filename part: %s", filename_reference_arg)
    logger.info("Using comparison filename part: %s", filename_comparison_arg)

    overlay_outdir = Path(overlay_outdir_arg).expanduser().resolve()
    data_dir = Path(data_dir_arg).expanduser().resolve()
    folders = [f.strip() for f in folders_arg.split(",")]
    filename_reference = filename_reference_arg.strip()
    filename_comparison = filename_comparison_arg.strip()

    if not overlay_outdir.exists():
        overlay_outdir.mkdir(parents=True, exist_ok=True)
        logger.info("Created output directory %s", overlay_outdir)

    


if __name__ == "__main__":
    main()

