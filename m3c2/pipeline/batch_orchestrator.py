"""High level orchestration for running the complete M3C2 workflow.

The :class:`BatchOrchestrator` coordinates configuration handling, parameter
estimation, execution of the core algorithm and subsequent analysis steps.
"""

from __future__ import annotations

import logging
import os
import time
from typing import List
import numpy as np

from m3c2.config.pipeline_config import PipelineConfig
from m3c2.core.statistics import StatisticsService
from m3c2.pipeline.component_factory import PipelineComponentFactory

logger = logging.getLogger(__name__)


class BatchOrchestrator:
    """Run the full M3C2 pipeline for a collection of configurations.

    Parameters
    ----------
    configs : list[PipelineConfig]
        Collection of configuration objects to process.
    sample_size : int, optional
        Number of core points to sample when estimating scales.
    output_format : str
        Format for statistical outputs, ``"excel"`` or ``"json"``.
    """

    def __init__(
        self,
        configs: List[PipelineConfig],
        strategy: str = "radius",
        fail_fast: bool = False,
    ) -> None:
        """Create a new orchestrator instance.

        Parameters
        ----------
        configs : list[PipelineConfig]
            Collection of configuration objects to process.
        strategy : str, optional
            Strategy used by the pipeline components.
        fail_fast : bool, optional
            Abort the batch on unexpected errors if ``True``. If ``False``
            (default), errors are logged and processing continues with the
            next job.
        """

        self.configs = configs
        self.fail_fast = fail_fast
        output_format = configs[0].output_format if configs else "excel"
        self.factory = PipelineComponentFactory(strategy, output_format)
        self.data_loader = self.factory.create_data_loader()
        self.scale_estimator = self.factory.create_scale_estimator()
        self.m3c2_executor = self.factory.create_m3c2_executor()
        # self.outlier_handler = self.factory.create_outlier_handler()
        self.statistics_runner = self.factory.create_statistics_runner()
        self.visualization_runner = self.factory.create_visualization_runner()

        logger.info("=== BatchOrchestrator initialisiert ===")
        logger.info("Konfigurationen: %d Jobs", len(self.configs))


    # Small helper to create consistent file name tags
    @staticmethod
    def _run_tag(cfg: PipelineConfig) -> str:
        """Return a concise tag representing the moving and reference files.

        Parameters
        ----------
        cfg : PipelineConfig
            Configuration for the current job.

        Returns
        -------
        str
            Combination of moving and reference filenames.
        """
        return f"{cfg.filename_mov}-{cfg.filename_ref}"
    
    def run_all(self) -> None:
        """Run the pipeline for each configured dataset.

        Returns
        -------
        None

        Notes
        -----
        Any exception raised during processing of a single job is caught and
        logged. When ``fail_fast`` is ``False`` (default) the batch continues
        with the next job. If ``fail_fast`` is ``True``, unexpected errors
        abort the batch.
        """
        if not self.configs:
            logger.warning("Keine Konfigurationen – nichts zu tun.")
            return

        # Process each configuration in sequence
        for cfg in self.configs:
            try:
                self._run_single(cfg)
            except (IOError, ValueError):
                logger.exception(
                    "[Job] Fehler in Job '%s' (Version %s)",
                    cfg.folder_id,
                    cfg.filename_ref,
                )
            except Exception:
                logger.exception(
                    "[Job] Unerwarteter Fehler in Job '%s' (Version %s)",
                    cfg.folder_id,
                    cfg.filename_ref,
                )
                if self.fail_fast:
                    raise

    def _run_single(self, cfg: PipelineConfig) -> None:
        """Execute the pipeline for a single configuration.

        Parameters
        ----------
        cfg : PipelineConfig
            Configuration for the dataset to process.

        Returns
        -------
        None

        Raises
        ------
        Exception
            Any exception from internal steps is caught and logged.
        """
        logger.info(
            "%s, %s, %s, %s",
            cfg.folder_id,
            cfg.filename_mov,
            cfg.filename_ref,
            cfg.process_python_CC,
        )
        start = time.perf_counter()

        if cfg.stats_singleordistance == "distance":
            self._batch_process_multicloud(cfg)

        elif cfg.stats_singleordistance == "single":
            self._batch_process_singlecloud(cfg)

        else:
            raise ValueError("Unbekannter stats_singleordistance-Wert")

        logger.info(
            "[Job] %s abgeschlossen in %.3fs",
            cfg.folder_id,
            time.perf_counter() - start,
        )

    def _batch_process_multicloud(self, cfg: PipelineConfig):
        """Process statistics for a pair of point clouds.

        Parameters
        ----------
        cfg : PipelineConfig
            Configuration describing the moving and reference clouds as well
            as processing options.

        Returns
        -------
        None

        Notes
        -----
        Depending on ``cfg.only_stats`` either run the full M3C2 distance
        computation followed by statistics generation or skip distance
        calculation and only compute statistics. Results are written to the
        output directory defined in ``cfg``.
        """
        ds, mov, ref, corepoints = self.data_loader.load_data(cfg, mode="multicloud")
        out_base = ds.config.folder
        tag = self._run_tag(cfg)


        #--------------------------------------------------------
        # Process incl. M3C2 Distance calculation

        if not cfg.only_stats:
            self._batch_process_multicloud_full(cfg, mov, ref, corepoints, out_base, tag)

        #--------------------------------------------------------
        # Process only statistics without computing distances
        
        else:
            try:
                logger.info("[Statistics] Berechne Statistiken …")
                self.statistics_runner.compute_statistics(cfg, mov, ref, tag)
            except (IOError, ValueError):
                logger.exception("Fehler bei der Berechnung der Statistik")
            except Exception:
                logger.exception(
                    "Unerwarteter Fehler bei der Berechnung der Statistik"
                )
                raise

    def _batch_process_singlecloud(self, cfg: PipelineConfig):
        """Compute statistics for a single cloud."""

        single_cloud = self.data_loader.load_data(cfg, mode="singlecloud")

        # --- Scale estimation for single cloud statistics parameters
        normal, projection = self.scale_estimator.determine_scales(
                cfg, single_cloud
            )

        try:
            logger.info("[Statistics] Berechne Statistiken …")
            self.statistics_runner.single_cloud_statistics_handler(
                cfg, single_cloud, normal
            )
        except (IOError, ValueError):
            logger.exception("Fehler bei der Berechnung der Statistik")
        except Exception:
            logger.exception(
                "Unerwarteter Fehler bei der Berechnung der Statistik"
            )
            raise

    def _save_params(self, cfg: PipelineConfig, normal: float, projection: float, out_base: str) -> None:
        """Persist determined scale parameters to disk.

        Parameters
        ----------
        cfg : PipelineConfig
            Configuration for naming the output file.
        normal : float
            Chosen normal scale.
        projection : float
            Chosen projection scale.
        out_base : str
            Directory where the parameter file should be written.
        """
        os.makedirs(out_base, exist_ok=True)
        tag = self._run_tag(cfg)
        params_path = os.path.join(out_base, f"{cfg.process_python_CC}_{tag}_m3c2_params.txt")
        with open(params_path, "w") as f:
            f.write(f"NormalScale={normal}\nSearchScale={projection}\n")
        logger.info("[Params] gespeichert: %s", params_path)


    def _handle_existing_params(self, cfg: PipelineConfig, out_base: str, tag: str):

        params_path = os.path.join(
            out_base, f"{cfg.process_python_CC}_{tag}_m3c2_params.txt"
        )
        normal, projection = StatisticsService._load_params(params_path)

        if not np.isnan(normal) and not np.isnan(projection):
            logger.info(
                "[Params] geladen: %s (NormalScale=%.6f, SearchScale=%.6f)",
                params_path,
                normal,
                projection,
            )
            return normal, projection

        return np.nan, np.nan
    

    def _batch_process_multicloud_full(self, cfg: PipelineConfig, mov: str, ref: str, corepoints: np.ndarray, out_base: str, tag: str):
        """Process a multicloud dataset including M3C2 distance calculation, statistics, and visualizations."""

        #--------------------------------------------------------
        # Process incl. M3C2 Distance calculation / Only stats / Only visuals,plots
        #--------------------------------------------------------

        # 1. Calculate / Collect parameters for M3C2 algorithm

        normal = projection = np.nan

        if cfg.use_existing_params:
            
            normal, projection = self._handle_existing_params(cfg, out_base, tag)

            if np.isnan(normal) and np.isnan(projection):
                logger.info("[Params] keine vorhandenen Parameter gefunden, berechne neu")
                normal, projection = self.scale_estimator.determine_scales(
                    cfg, corepoints
                )
        else:
            normal, projection = self.scale_estimator.determine_scales(
                cfg, corepoints
            )
            self._save_params(cfg, normal, projection, out_base)

        
        # 3. Run M3C2 algorithm with collected parameters

        distances = self.m3c2_executor.run_m3c2(
            cfg, mov, ref, corepoints, normal, projection, out_base, tag
        )

        # 4. Generate Histogram & .Ply output files incl. calculated distances
        self.visualization_runner.generate_visuals(
            cfg, mov, distances, out_base, tag
        )


        # 5. Compute Outliers

        # self._handle_outliers(cfg, out_base, tag)
            


    def _handle_outliers(self, cfg: PipelineConfig, out_base: str, tag: str):


        # ----- Remove Outliers -----
        try:
            logger.info("[Outlier] Entferne Ausreißer für %s", cfg.folder_id)
            self.outlier_handler.exclude_outliers(cfg, out_base, tag)
        except (IOError, ValueError):
            logger.exception("Fehler beim Entfernen von Ausreißern")
        except Exception:
            logger.exception(
                "Unerwarteter Fehler beim Entfernen von Ausreißern"
            )
            raise

        # ----- Generate Outlier/Inlier Clouds -----
        try:
            logger.info(
                "[Outlier] Erzeuge .ply Dateien für Outliers / Inliers …"
            )
            self.visualization_runner.generate_clouds_outliers(
                cfg, ds.config.folder, tag
            )
        except (IOError, ValueError):
            logger.exception(
                "Fehler beim Erzeugen von .ply Dateien für Ausreißer / Inlier"
            )
        except Exception:
            logger.exception(
                "Unerwarteter Fehler beim Erzeugen von .ply Dateien für Ausreißer / Inlier"
            )
            raise