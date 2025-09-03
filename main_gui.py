from m3c2.cli.cli import CLIApp
from m3c2.cli.argparse_gui import run_gui


def main():
    app = CLIApp()
    run_gui(app.build_parser(), app.run)


if __name__ == "__main__":
    main()