"""Entry point for visualization utilities.

This script provides a thin wrapper around the plotting CLI so that
visualisations can be triggered directly via ``python -m main_plotter``.
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

    parser = overlay_report.build_arg_parser_severalfolders()
    args = parser.parse_args()
    logger.debug("Arguments: %s", args)

    data_dir_arg = args.data_dir
    folders_arg = args.folders
    filenames_arg = args.filenames
    overlay_outdir_arg = args.overlay_outdir
    options_arg = args.options or getattr(args, "options", None)

    if (
        data_dir_arg is None
        or folders_arg is None
        or filenames_arg is None
        or overlay_outdir_arg is None
    ):
        parser.error(
            "data_dir, folders, filenames, and overlay_outdir are required (via CLI or config)"
        )

    logger.info("Generating overlay report in %s", overlay_outdir_arg)
    logger.info("Using folders: %s", folders_arg)
    logger.info("Using filenames: %s", filenames_arg)

    overlay_outdir = Path(overlay_outdir_arg).expanduser().resolve()
    data_dir = Path(data_dir_arg).expanduser().resolve()
    folders = [f.strip() for f in folders_arg]
    filenames = [f.strip() for f in filenames_arg]

    if not overlay_outdir.exists():
        overlay_outdir.mkdir(parents=True, exist_ok=True)
        logger.info("Created output directory %s", overlay_outdir)

    pdfs: list[str] = []
    for folder in folders:
        folder_path = data_dir / folder
        data = {}
        for name in filenames:
            file_path = folder_path / f"python_{name}_m3c2_distances.txt"
            try:
                data[name] = overlay_report.load_distance_file(str(file_path))
            except (FileNotFoundError, ValueError):
                logger.warning("Skipping missing or invalid file %s", file_path)
        if len(data) < 2:
            logger.warning("Not enough distance files in %s, skipping", folder_path)
            continue
        outdir = overlay_outdir / folder
        outdir.mkdir(parents=True, exist_ok=True)
        PlotService.overlay_from_data(data, str(outdir))
        pdf = PlotService.build_parts_pdf(
            str(outdir),
            pdf_path=str(outdir / "report.pdf"),
            include_with=True,
            include_inlier=False,
        )
        if pdf:
            pdfs.append(pdf)

    if pdfs:
        merged = PlotService.merge_pdfs(pdfs, str(overlay_outdir / "combined_report.pdf"))
        logger.info("Merged PDF saved to %s", merged)

    logger.debug("Plot types: %s", options_arg)
    logger.debug("Output directory: %s", overlay_outdir)

    


if __name__ == "__main__":
    main()

