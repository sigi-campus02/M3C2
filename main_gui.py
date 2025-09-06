"""GUI entry point for the M3C2 pipeline."""

import logging

from m3c2.cli.argparse_gui import run_gui
from m3c2.cli.cli import CLIApp

logger = logging.getLogger(__name__)


def main() -> None:
    """Launch the GUI application."""

    logger.info("Starting GUI application")

    app = CLIApp()
    parser = app.build_parser()

    def dispatch(argv: list[str]) -> None:
        """Run the regular command line application."""

        app.run(argv)

    run_gui(parser, dispatch)


if __name__ == "__main__":
    main()
