"""GUI entry point for the M3C2 pipeline."""

# Imports
import logging

from m3c2.cli.argparse_gui import run_gui
from m3c2.cli.cli import CLIApp

# Logging
logger = logging.getLogger(__name__)


# Public API
def main() -> None:
    """Launch the GUI application."""

    logger.info("Starting GUI application")

    app = CLIApp()
    run_gui(app.build_parser(), app.run)


if __name__ == "__main__":
    main()

