"""Main entry point for the M3C2 pipeline.

Run Pipeline using the command line:

    Option 1. With arguments in config.json file: python -m main
    Option 2. Directly with arguments in the command line:

                python -m main
                --data_dir ./data
                --folders 0342-0349
                --filename_ref ref.ply
                --filename_mov mov.ply
                --project MARS
                --output_format excel
                --outlier_detection_method rmse
                --outlier_multiplicator 3.0
                --log_level INFO
"""

import logging
import os

from m3c2.io.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Execute the command line application."""
    from m3c2.cli.cli import CLIApp

    app = CLIApp()
    args = app.parse_args()
    level = os.environ.get("LOG_LEVEL", args.log_level)
    setup_logging(level=level, log_file="logs/orchestration.log")

    logger.info(
        "Running pipeline for folders %s with reference filename %s",
        args.folders,
        args.filename_ref,
    )
    try:
        app.run(arg=args)
        logger.info("Processing completed successfully")
    except Exception:
        logger.exception("Processing failed")
        raise


if __name__ == "__main__":
    main()

