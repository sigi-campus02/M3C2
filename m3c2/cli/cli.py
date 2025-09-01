"""Command-line interface for the M3C2 pipeline."""
from __future__ import annotations

import argparse
import logging
import json
from pathlib import Path
from typing import Any, List, Optional
from m3c2.io.logging_utils import setup_logging
from m3c2.pipeline.batch_orchestrator import BatchOrchestrator
from m3c2.config.pipeline_config import PipelineConfig

class CLIApp:
    """Command line application wrapper for the pipeline."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_path = (
            Path(config_path)
            if config_path is not None
            else Path(__file__).resolve().parent.parent / "config.json"
        )

    # ------------------------------------------------------------------
    def build_parser(self) -> argparse.ArgumentParser:
        """Build the argument parser for command line and GUI usage."""
        parser = argparse.ArgumentParser(
            description="M3C2 Pipeline Command-Line Interface",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )
        parser.add_argument(
            "--data_dir",
            type=str,
            default="data",
            help="Directory containing point cloud data folders.",
        )
        parser.add_argument(
            "--folders",
            type=str,
            nargs="+",
            default=None,
            help="List of folder IDs to process (e.g., '0342-0349 0817-0821').",
        )
        parser.add_argument(
            "--filename_ref",
            type=str,
            default="ref",
            help="Name of reference cloud file to be compared.",
        )
        parser.add_argument(
            "--filename_mov",
            type=str,
            default="mov",
            help="Name of moving point cloud file.",
        )
        parser.add_argument(
            "--mov_as_corepoints",
            action="store_true",
            help="Use moving point cloud as corepoints.",
        )
        parser.add_argument(
            "--use_subsampled_corepoints",
            type=int,
            default=1,
            help="Subsampling factor for corepoints (1 = no subsampling).",
        )
        parser.add_argument(
            "--sample_size",
            type=int,
            default=10000,
            help="Sample size used for parameter estimation (normal & projection scale).",
        )
        parser.add_argument(
            "--only_stats",
            action="store_true",
            help="Only compute statistics based on existing distance files (no M3C2 processing).",
        )
        parser.add_argument(
            "--stats_singleordistance",
            type=str,
            choices=["single", "distance"],
            default="distance",
            help="Type of statistics to compute: 'single' for single-cloud, 'distance' for distance-based.",
        )
        parser.add_argument(
            "--output_format",
            type=str,
            choices=["excel", "json"],
            default="excel",
            help="Output format for statistics: 'excel' or 'json'.",
        )
        parser.add_argument(
            "--project",
            type=str,
            default="MARS",
            help="Project name used for file and folder naming.",
        )
        parser.add_argument(
            "--normal_override",
            type=float,
            default=None,
            help="Override normal scale parameter.",
        )
        parser.add_argument(
            "--proj_override",
            type=float,
            default=None,
            help="Override projection scale parameter.",
        )
        parser.add_argument(
            "--use_existing_params",
            action="store_true",
            help="Use existing parameters in folder if available.",
        )
        parser.add_argument(
            "--outlier_detection_method",
            type=str,
            choices=["rmse", "iqr", "std", "nmad"],
            default="rmse",
            help="Method for outlier detection.",
        )
        parser.add_argument(
            "--outlier_rmse_multiplicator",
            type=float,
            default=3.0,
            help="Outlier removal threshold as a multiple of used detection method.",
        )
        parser.add_argument(
            "--log_level",
            type=str,
            choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
            default="INFO",
            help="Logging level.",
        )
        return parser


    def parse_args(self, argv: Optional[List[str]] = None) -> argparse.Namespace:
        """Parse command line arguments using the configured parser."""
        parser = self.build_parser()

        # Load defaults from configuration file if available
        if self.config_path.exists():
            try:
                with self.config_path.open("r", encoding="utf-8") as handle:
                    data: dict[str, Any] = json.load(handle)
            except json.JSONDecodeError:
                data = {}
            defaults = data.get("arguments", data)
            if isinstance(defaults, dict):
                parser.set_defaults(**defaults)

        args = parser.parse_args(argv)

        return args

    # ------------------------------------------------------------------

    def create_pipeline_configurations(
            self,
            folder_ids: list[str],
            args: argparse.Namespace
    ) -> list[PipelineConfig]:
        """Create pipeline configurations based on the parsed arguments."""
        configs: List[PipelineConfig] = []
        for folder in folder_ids:
            config = PipelineConfig(
                data_dir=args.data_dir,
                folder_id=folder,
                filename_ref=args.filename_ref,
                filename_mov=args.filename_mov,
                mov_as_corepoints=args.mov_as_corepoints,
                use_subsampled_corepoints=args.use_subsampled_corepoints,
                sample_size=args.sample_size,
                only_stats=args.only_stats,
                stats_singleordistance=args.stats_singleordistance,
                project=args.project,
                normal_override=args.normal_override,
                proj_override=args.proj_override,
                use_existing_params=args.use_existing_params,
                outlier_detection_method=args.outlier_detection_method,
                outlier_rmse_multiplicator=args.outlier_rmse_multiplicator,
                output_format=args.output_format,
                log_level=args.log_level,
            )
            configs.append(config)
        return configs
    

    def run(self, argv: Optional[List[str]] = None) -> int:
        """Main runner: setup logging, validate inputs, build configs, run orchestrator."""
       
        arg = self.parse_args(argv)

        log_file = "logs/orchestration.log"
        setup_logging(level=arg.log_level, log_file=log_file)

        base_dir = Path(arg.data_dir).expanduser().resolve()
        
        if not base_dir.is_dir():
            self.logger.error("The specified data directory does not exist or is not a directory: %s", base_dir)
            return 1
        
        missing_folders = [f for f in arg.folders if not (base_dir / f).is_dir()]
        if missing_folders:
            self.logger.error("The following folders are missing in the data directory: %s", missing_folders)
            return 1
    
        if not arg.filename_ref:
            self.logger.error("No ref filename specified.")
            return 1
        if not arg.filename_mov:
            self.logger.error("No mov filename specified.")
            return 1

        folder_ids = list(arg.folders)

        self.logger.info("Base directories for processing: %s", base_dir)
        self.logger.info("Folder IDs for processing: %s", folder_ids)
        self.logger.info("Reference filename for processing: %s", arg.filename_ref)
        self.logger.info("Moving filename for processing: %s", arg.filename_mov)
    
        # Building pipeline configuration

        configs = self.create_pipeline_configurations(folder_ids, arg)
        
        # Running orchestrator

        orchestrator = BatchOrchestrator(configs, arg.sample_size, arg.output_format)

        try:
            orchestrator.run()
        except Exception as e:
            self.logger.error("Error occurred while running orchestrator: %s", e)
            return 1
        
        self.logger.info("Processing completed successfully.")