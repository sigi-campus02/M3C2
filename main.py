"""Main entry point for the M3C2 pipeline.

Run the pipeline from the command line using one of the following options:

    Option 1: Provide arguments in the ``config.json`` file and run ``python -m main``
    Option 2: Pass arguments directly on the command line:

        python -m main \
            --data_dir ./data \
            --folders 0342-0349 \
            --filename_ref ref.ply \
            --filename_mov mov.ply \
            --project MARS \
            --output_format excel \
            --outlier_detection_method rmse \
            --outlier_multiplicator 3.0 \
            --log_level INFO
"""

# Imports
import logging

from m3c2.config.logging_config import setup_logging

# Logging
logger = logging.getLogger(__name__)


# Public API
def main() -> None:
    """Execute the command line application."""

    setup_logging()

    logger.info("Starting CLI application")

    from m3c2.cli.cli import CLIApp

    try:
        CLIApp().run()
    except Exception as exc:
        logger.error("Error running CLI application: %s", exc, exc_info=True)
        raise


if __name__ == "__main__":
    main()

