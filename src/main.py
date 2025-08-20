"""Command line interface for running the M3C2 pipeline.

The script orchestrates the processing of point clouds with a configurable
pipeline.  In addition to the dataset parameters, boolean flags allow
selection of individual pipeline steps:

* ``--run-m3c2`` – execute only the M3C2 algorithm
* ``--stats-distances`` – compute statistics for existing distance files
* ``--stats-single-cloud`` – compute statistics for individual point clouds
* ``--visuals`` – generate histogram and colored point cloud files
* ``--plots`` – create comparison plots
* ``--run-all`` – perform all steps (default if no flag is given)

The basic dataset options are:

* ``folder`` – path to the dataset folder (default: ``data/rocks``)
* ``filename_mov`` – name of the moving point cloud file (default: ``points_100``)
* ``filename_ref`` – name of the reference point cloud file (default: ``points_zshift``)
* ``mov_as_corepoints`` – use the moving cloud as corepoints (default: ``True``)
* ``use_subsampled_corepoints`` – number of corepoints for subsampling;
  ``1`` disables subsampling (default: ``1``)
* ``strategy`` – processing strategy, e.g. ``radius`` (default: ``radius``)
* ``sample_size`` – sample size for parameter estimation (default: ``10000``)
* ``process_python_CC`` – alternative CC for CloudCompare distance files
  (default: ``CC``)
"""

from batch_orchestrator import BatchOrchestrator
from pipeline_config import PipelineConfig
import argparse
import os
from logging_utils import setup_logging
from statistics_service import StatisticsService
from plot_service import PlotService, PlotConfig, PlotOptions

EXTS = [".xyz", ".ply", ".las", ".laz", ".txt", ".csv"]


def _find_existing_file(base_dir: str, stem: str) -> str | None:
    for ext in EXTS:
        path = os.path.join(base_dir, stem + ext)
        if os.path.exists(path):
            return path
    return None


def _single_cloud_stats(folder: str, stems: list[str]) -> None:
    rows = []
    for stem in stems:
        path = _find_existing_file(folder, stem)
        if not path:
            continue
        stats = StatisticsService.cloud_stats_from_file(
            path,
            role="mov",
            area_m2=None,
            radius=0.5,
            k=6,
            sample_size=100_000,
            use_convex_hull=True,
        )
        rows.append(stats)
    if rows:
        StatisticsService.write_cloud_stats(
            rows, out_xlsx="m3c2_stats_clouds.xlsx", sheet_name="CloudStats"
        )


def _generate_plots(folder: str, filenames: list[str]) -> None:
    cfg = PlotConfig(
        folder_id=folder,
        filenames=filenames,
        versions=["python", "CC"],
        bins=256,
        outdir="Plots",
    )
    opts = PlotOptions(
        plot_hist=True,
        plot_gauss=True,
        plot_weibull=True,
        plot_box=True,
        plot_qq=True,
        plot_grouped_bar=True,
    )
    PlotService.overlay_plots(folder, cfg, opts)
    PlotService.summary_pdf(folder, filenames, pdf_name="Plot_Vergleich.pdf", outdir="Plots")


def main(args: argparse.Namespace) -> None:
    """Run the M3C2 pipeline with the provided arguments."""

    cfgs = [
        PipelineConfig(
            args.folder,
            args.filename_mov,
            args.filename_ref,
            args.mov_as_corepoints,
            args.use_subsampled_corepoints,
            args.process_python_CC,
        ),
    ]

    orchestrator = BatchOrchestrator(cfgs, args.strategy, args.sample_size)

    if not (
        args.run_m3c2
        or args.stats_distances
        or args.stats_single_cloud
        or args.visuals
        or args.plots
        or args.run_all
    ):
        args.run_all = True

    if args.run_all:
        orchestrator.run_all()
        _single_cloud_stats(args.folder, [args.filename_mov, args.filename_ref])
        _generate_plots(args.folder, [args.filename_ref])
        return

    orchestrator.run_all(
        run_m3c2=args.run_m3c2,
        run_stats=args.stats_distances,
        run_visuals=args.visuals,
    )

    if args.stats_single_cloud:
        _single_cloud_stats(args.folder, [args.filename_mov, args.filename_ref])
    if args.plots:
        _generate_plots(args.folder, [args.filename_ref])



if __name__ == "__main__":
    setup_logging()
    parser = argparse.ArgumentParser(description="Run the M3C2 pipeline")
    parser.add_argument(
        "--folder",
        default=os.path.join("data", "rocks"),
        help="Path to the dataset folder",
    )
    parser.add_argument(
        "--filename-mov",
        dest="filename_mov",
        default="points_100",
        help="Filename of the moving point cloud",
    )
    parser.add_argument(
        "--filename-ref",
        dest="filename_ref",
        default="points_zshift",
        help="Filename of the reference point cloud",
    )
    parser.add_argument(
        "--mov-as-corepoints",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use the moving cloud as corepoints",
    )
    parser.add_argument(
        "--use-subsampled-corepoints",
        dest="use_subsampled_corepoints",
        type=int,
        default=1,
        help="Number of corepoints for subsampling; 1 disables subsampling",
    )
    parser.add_argument(
        "--strategy",
        default="radius",
        help="Processing strategy to use",
    )
    parser.add_argument(
        "--sample-size",
        dest="sample_size",
        type=int,
        default=10000,
        help="Sample size for parameter estimation",
    )
    parser.add_argument(
        "--process-python-CC",
        dest="process_python_CC",
        default="CC",
        help="Alternative CC for CloudCompare distance files",
    )
    parser.add_argument("--run-m3c2", action="store_true", help="Execute the M3C2 algorithm")
    parser.add_argument(
        "--stats-distances", action="store_true", help="Compute statistics for distance files"
    )
    parser.add_argument(
        "--stats-single-cloud",
        action="store_true",
        help="Compute statistics for individual point clouds",
    )
    parser.add_argument("--visuals", action="store_true", help="Generate visualizations")
    parser.add_argument("--plots", action="store_true", help="Generate plots from statistics")
    parser.add_argument("--run-all", action="store_true", help="Run all pipeline steps")
    parsed_args = parser.parse_args()
    main(parsed_args)
