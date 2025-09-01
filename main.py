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

from m3c2.io.logging_utils import resolve_log_level, setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    """Execute the command line application."""
    setup_logging(level=resolve_log_level())
    logger.info("Starting CLI application")

    from m3c2.cli.cli import CLIApp

    try:
        CLIApp().run()
    except Exception as exc:
        logger.error("Error running CLI application: %s", exc, exc_info=True)
        raise


if __name__ == "__main__":
    main()

