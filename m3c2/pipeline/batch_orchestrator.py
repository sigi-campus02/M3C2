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

    def __init__(self, configs: List[PipelineConfig], strategy: str = "radius") -> None:
        """Create a new orchestrator instance."""

        self.configs = configs
        output_format = configs[0].output_format if configs else "excel"
        self.factory = PipelineComponentFactory(strategy, output_format)
        self.data_loader = self.factory.create_data_loader()
        self.scale_estimator = self.factory.create_scale_estimator()
        self.m3c2_executor = self.factory.create_m3c2_executor()
        self.outlier_handler = self.factory.create_outlier_handler()
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
        logged so that subsequent jobs can continue.
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

        ds, mov, ref, corepoints = self.data_loader.load_data(cfg)
        tag = self._run_tag(cfg)

        if cfg.process_python_CC == "python" and not cfg.only_stats:
            out_base = ds.config.folder
            normal = projection = np.nan
            if cfg.use_existing_params:
                params_path = os.path.join(out_base, f"{cfg.process_python_CC}_{tag}_m3c2_params.txt")
                normal, projection = StatisticsService._load_params(params_path)
                if not np.isnan(normal) and not np.isnan(projection):
                    logger.info(
                        "[Params] geladen: %s (NormalScale=%.6f, SearchScale=%.6f)",
                        params_path,
                        normal,
                        projection,
                    )
                else:
                    logger.info("[Params] keine vorhandenen Parameter gefunden, berechne neu")
            if np.isnan(normal) or np.isnan(projection):
                normal, projection = self.scale_estimator.determine_scales(cfg, corepoints)
                self._save_params(cfg, normal, projection, out_base)
            distances, uncertainties, dists_path = self.m3c2_executor.run_m3c2(
                cfg, mov, ref, corepoints, normal, projection, out_base, tag
            )
            self.visualization_runner.generate_visuals(cfg, mov, distances, out_base, tag)

        try:
            logger.info("[Outlier] Entferne Ausreißer für %s", cfg.folder_id)
            self.outlier_handler.exclude_outliers(cfg, dists_path)
        except (IOError, ValueError):
            logger.exception("Fehler beim Entfernen von Ausreißern")
        except Exception:
            logger.exception("Unerwarteter Fehler beim Entfernen von Ausreißern")
            raise

        try:
            logger.info("[Outlier] Erzeuge .ply Dateien für Outliers / Inliers …")
            self.visualization_runner.generate_clouds_outliers(cfg, ds.config.folder, tag)
        except (IOError, ValueError):
            logger.exception("Fehler beim Erzeugen von .ply Dateien für Ausreißer / Inlier")
        except Exception:
            logger.exception("Unerwarteter Fehler beim Erzeugen von .ply Dateien für Ausreißer / Inlier")
            raise

        try:
            logger.info("[Statistics] Berechne Statistiken …")
            self.statistics_runner.compute_statistics(cfg, ref, tag)
        except (IOError, ValueError):
            logger.exception("Fehler bei der Berechnung der Statistik")
        except Exception:
            logger.exception("Unerwarteter Fehler bei der Berechnung der Statistik")
            raise

        logger.info("[Job] %s abgeschlossen in %.3fs", cfg.folder_id, time.perf_counter() - start)

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
