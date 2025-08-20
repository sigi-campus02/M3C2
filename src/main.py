"""Command line interface for running the M3C2 pipeline.

The script orchestrates the processing of point clouds with a configurable
pipeline.  The following command line arguments are available:

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
    orchestrator.run_all()


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
    parsed_args = parser.parse_args()
    main(parsed_args)
