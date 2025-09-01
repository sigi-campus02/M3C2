"""Main entry point for the M3C2 pipeline."""

import sys, os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from m3c2.cli.cli import CLIApp


def main() -> None:
    """Execute the command line application."""
    CLIApp().run()


if __name__ == "__main__":
    main()

