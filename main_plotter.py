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
    logger.debug("Arguments: %s", args)

    # ``build_arg_parser`` may populate defaults via the config file. Older
    # versions used the keys ``files``/``outdir`` which we still honour as
    # fallbacks to remain backwards compatible.
    overlay_files_arg = args.overlay_files or getattr(args, "files", None)
    overlay_outdir_arg = args.overlay_outdir or getattr(args, "outdir", None)

    if overlay_files_arg is None or overlay_outdir_arg is None:
        parser.error("overlay_files and overlay_outdir are required (via CLI or config)")

    logger.info("Generating overlay report in %s", overlay_outdir_arg)
    logger.info("Using distance files: %s", overlay_files_arg)

    overlay_outdir = Path(overlay_outdir_arg).expanduser().resolve()
    overlay_files = [Path(f).expanduser().resolve() for f in overlay_files_arg]

    if not overlay_outdir.exists():
        overlay_outdir.mkdir(parents=True, exist_ok=True)
        logger.info("Created output directory %s", overlay_outdir)

    overlay_report.main([str(f) for f in overlay_files], str(overlay_outdir))


if __name__ == "__main__":
    main()

