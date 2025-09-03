"""Command-line interface for the M3C2 pipeline."""
from __future__ import annotations

import argparse
import logging
import json
from pathlib import Path
from typing import Any, List, Optional

from m3c2.config.logging_config import setup_logging
from m3c2.pipeline.batch_orchestrator import BatchOrchestrator
from m3c2.config.pipeline_config import PipelineConfig

class CLIApp:
    """Command line application wrapper for the pipeline."""

    def __init__(self, config_path: str | Path | None = None) -> None:
        """Initialize the CLI application wrapper.

        Parameters
        ----------
        config_path:
            Optional path to a JSON configuration file. If omitted, the
            default ``config.json`` located two directories above this file
            is used.

        The resolved configuration path is stored on the instance and later
        consulted to provide default argument values when parsing command-line
        options. A logger dedicated to this class is also set up.
        """
        self.logger = logging.getLogger(self.__class__.__name__)
        self.config_path = (
            Path(config_path)
            if config_path is not None
            else Path(__file__).resolve().parents[2] / "config.json"
        )

    def _load_schema_defaults(self) -> dict[str, Any]:
        """Load default argument values from the JSON schema.

        Returns
        -------
        dict[str, Any]
            Mapping of argument names to their default values as defined in
            ``config.schema.json``. If the schema cannot be read, an empty
            dictionary is returned.
        """
        schema_path = self.config_path.with_name("config.schema.json")
        try:
            with schema_path.open("r", encoding="utf-8") as handle:
                schema = json.load(handle)
        except (OSError, json.JSONDecodeError):
            return {}

        arg_props = (
            schema.get("properties", {})
            .get("arguments", {})
            .get("properties", {})
        )
        return {
            key: value["default"]
            for key, value in arg_props.items()
            if "default" in value
        }

    # ------------------------------------------------------------------
    def build_parser(self) -> argparse.ArgumentParser:
        """Build the argument parser for command line and GUI usage."""
        parser = argparse.ArgumentParser(
            description="M3C2 Pipeline Command-Line Interface",
            formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        )

        # Arguments are defined without hardcoded defaults. Defaults are
        # populated from the JSON schema after all arguments have been added.
        parser.add_argument(
            "--data_dir",
            type=str,
            help="Directory containing point cloud data folders.",
        )
        parser.add_argument(
            "--folders",
            type=str,
            nargs="+",
            help="List of folder IDs to process (e.g., '0342-0349 0817-0821').",
        )
        parser.add_argument(
            "--filename_ref",
            type=str,
            help="Name of reference cloud file to be compared.",
        )
        parser.add_argument(
            "--filename_mov",
            type=str,
            help="Name of moving point cloud file.",
        )
        parser.add_argument(
            "--filename_singlecloud",
            type=str,
            help="Name of single statistics file.",
        )
        parser.add_argument(
            "--mov_as_corepoints",
            action=argparse.BooleanOptionalAction,
            help="Use moving point cloud as corepoints.",
        )
        parser.add_argument(
            "--use_subsampled_corepoints",
            type=int,
            help="Subsampling factor for corepoints (1 = no subsampling).",
        )
        parser.add_argument(
            "--sample_size",
            type=int,
            help="Sample size used for parameter estimation (normal & projection scale).",
        )
        parser.add_argument(
            "--scale_strategy",
            type=str,
            choices=["radius"],
            help="Strategy for scanning normal/projection scales.",
        )
        parser.add_argument(
            "--only_stats",
            action=argparse.BooleanOptionalAction,
            help="Only compute statistics based on existing distance files (no M3C2 processing).",
        )
        parser.add_argument(
            "--stats_singleordistance",
            type=str,
            choices=["single", "distance"],
            help="Type of statistics to compute: 'single' for single-cloud, 'distance' for distance-based.",
        )
        parser.add_argument(
            "--output_format",
            type=str,
            choices=["excel", "json"],
            help="Output format for statistics: 'excel' or 'json'.",
        )
        parser.add_argument(
            "--project",
            type=str,
            help="Project name used for file and folder naming.",
        )
        parser.add_argument(
            "--normal_override",
            type=float,
            help="Override normal scale parameter.",
        )
        parser.add_argument(
            "--proj_override",
            type=float,
            help="Override projection scale parameter.",
        )
        parser.add_argument(
            "--use_existing_params",
            action=argparse.BooleanOptionalAction,
            help="Use existing parameters in folder if available.",
        )
        parser.add_argument(
            "--outlier_detection_method",
            type=str,
            choices=["rmse", "iqr", "std", "nmad"],
            help="Method for outlier detection.",
        )
        parser.add_argument(
            "--outlier_multiplicator",
            type=float,
            help="Outlier removal threshold as a multiple of used detection method.",
        )

        # Apply defaults from the configuration schema if available
        defaults = self._load_schema_defaults()
        if defaults:
            parser.set_defaults(**defaults)

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
                defaults.pop("log_level", None)
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
                filename_singlecloud=args.filename_singlecloud,
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
                outlier_multiplicator=args.outlier_multiplicator,
                output_format=args.output_format
            )
            configs.append(config)
        return configs
    

    def run(self, argv: Optional[List[str]] = None) -> int:
        """Main runner: setup logging, validate inputs, build configs, run orchestrator."""
       
        arg = self.parse_args(argv)

        setup_logging()

        base_dir = Path(arg.data_dir).expanduser().resolve()
        
        if not base_dir.is_dir():
            self.logger.error("The specified data directory does not exist or is not a directory: %s", base_dir)
            return 1
        
        if not arg.folders:
            self.logger.error("No folders specified. Use --folders or provide them in the configuration file.")
            return 1
        
        missing_folders = [f for f in arg.folders if not (base_dir / f).is_dir()]

        if missing_folders:
            self.logger.error("The following folders are missing in the data directory: %s", missing_folders)
            return 1
    
        if not arg.filename_ref and not arg.stats_singleordistance == "single":
            self.logger.error("No ref filename specified.")
            return 1
        if not arg.filename_mov and not arg.stats_singleordistance == "single":
            self.logger.error("No mov filename specified.")
            return 1
        if not arg.filename_singlecloud and arg.stats_singleordistance == "single":
            self.logger.error(
                "No single stats filename specified for single-cloud statistics"
            )
            return 1

        folder_ids = list(arg.folders)

        self.logger.info("Base directories for processing: %s", base_dir)
        self.logger.info("Folder IDs for processing: %s", folder_ids)
        self.logger.info("Reference filename for processing: %s", arg.filename_ref)
        self.logger.info("Moving filename for processing: %s", arg.filename_mov)
        self.logger.info(
            "Single statistics filename for processing: %s", arg.filename_singlecloud
        )
    
        # Building pipeline configuration

        configs = self.create_pipeline_configurations(folder_ids, arg)

        # Running orchestrator
        orchestrator = BatchOrchestrator(configs, strategy=arg.scale_strategy)

        try:
            orchestrator.run_all()
        except Exception as e:
            self.logger.error("Error occurred while running orchestrator: %s", e)
            return 1

        self.logger.info("Processing completed successfully.")
        return 0
