"""GUI entry point for the M3C2 pipeline."""

# Imports
import logging

from pathlib import Path

from m3c2.cli.argparse_gui import run_gui
from m3c2.cli.cli import CLIApp
from m3c2.cli import overlay_report
from m3c2.visualization.services.plot_service import PlotService

# Logging
logger = logging.getLogger(__name__)


# Public API
def main() -> None:
    """Launch the GUI application."""

    logger.info("Starting GUI application")

    app = CLIApp()
    parser = app.build_parser()

    # Plotting specific arguments
    parser.add_argument(
        "--plot_strategy",
        type=str,
        choices=["specificfile", "onefolder", "severalfolders"],
        help="Strategy to generate plots from existing distance files.",
    )
    parser.add_argument(
        "--overlay_files",
        type=str,
        nargs="+",
        help="List of specific distance files to plot.",
    )
    parser.add_argument(
        "--overlay_outdir",
        type=str,
        help="Directory for output plots and reports.",
    )
    parser.add_argument(
        "--plot_types",
        type=str,
        nargs="+",
        help="Plot types to generate for specific files.",
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="Folder containing distance files to process.",
    )
    parser.add_argument(
        "--options",
        type=str,
        nargs="+",
        help="List of plot options to generate.",
    )
    parser.add_argument(
        "--filenames",
        type=str,
        nargs="+",
        help="Distance file name parts present in each folder.",
    )

    def dispatch(argv: list[str]) -> None:
        args = parser.parse_args(argv)
        if args.stats_singleordistance == "plot":
            strategy = args.plot_strategy or "specificfile"
            if strategy == "specificfile":
                if not args.overlay_files or not args.overlay_outdir:
                    raise ValueError("overlay_files and overlay_outdir are required")
                overlay_report.main(args.overlay_files, args.overlay_outdir)
            elif strategy == "onefolder":
                if not args.folder or not args.overlay_outdir:
                    raise ValueError("folder and overlay_outdir are required")
                PlotService.overlay_by_index(
                    data_dir=args.folder,
                    outdir=args.overlay_outdir,
                    options=args.options,
                )
            elif strategy == "severalfolders":
                if (
                    not args.data_dir
                    or not args.folders
                    or not args.filenames
                    or not args.overlay_outdir
                ):
                    raise ValueError(
                        "data_dir, folders, filenames and overlay_outdir are required"
                    )
                data_dir = Path(args.data_dir).expanduser().resolve()
                overlay_outdir = Path(args.overlay_outdir).expanduser().resolve()
                if not overlay_outdir.exists():
                    overlay_outdir.mkdir(parents=True, exist_ok=True)
                pdfs: list[str] = []
                folders = [f.strip() for f in args.folders]
                filenames = [f.strip() for f in args.filenames]
                for folder in folders:
                    folder_path = data_dir / folder
                    data: dict[str, object] = {}
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
                    PlotService.merge_pdfs(
                        pdfs, str(overlay_outdir / "combined_report.pdf")
                    )
            else:
                raise ValueError(f"Unknown plot strategy: {strategy}")
        else:
            app.run(argv)

    run_gui(parser, dispatch)


if __name__ == "__main__":
    main()

