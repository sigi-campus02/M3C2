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
from m3c2.statistics import StatisticsService
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
        if cfg.filename_mov is not None and cfg.filename_ref is None:
            return f"{cfg.filename_mov}-{cfg.filename_ref}"
        else:
            return f"{cfg.filename_singlecloud}"
        
    
    def run_all(self) -> None:
        """Run the pipeline for each configured dataset.

        Returns
        -------
        None

        Notes
        -----
        Any runtime error raised during processing of a single job is caught and
        logged. When ``fail_fast`` is ``False`` (default) the batch continues
        with the next job. If ``fail_fast`` is ``True``, such errors
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
            except RuntimeError:
                logger.exception(
                    "[Job] Unerwarteter Fehler in Job '%s' (Version %s)",
                    cfg.folder_id,
                    cfg.filename_ref,
                )
                if self.fail_fast:
                    raise
            except Exception:
                logger.exception(
                    "[Job] Unbekannter Fehler in Job '%s' (Version %s)",
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
            except RuntimeError:
                logger.exception(
                    "Unerwarteter Fehler bei der Berechnung der Statistik"
                )
                raise

    def _batch_process_singlecloud(self, cfg: PipelineConfig):
        """Compute statistics for an individual point cloud.

        Parameters
        ----------
        cfg : PipelineConfig
            Configuration describing the input cloud and output settings.

        Notes
        -----
        The cloud specified in ``cfg`` is loaded and the normal and
        projection scales are determined using :attr:`scale_estimator`.
        These scales are then passed to
        ``statistics_runner.single_cloud_statistics_handler`` to compute
        the statistical summaries written to the configured output paths.

        Raises
        ------
        IOError, ValueError
            If the input data is missing or the configuration is invalid.
        RuntimeError
            Any unexpected runtime error during scale estimation or statistics
            computation is logged and re-raised.
        """

        single_cloud = self.data_loader.load_data(cfg, mode="singlecloud")


        # --- Scale estimation for single cloud statistics parameters

        if cfg.use_existing_params:
            normal = projection = np.nan
            normal, projection = self._handle_override_params(cfg)

        elif not cfg.use_existing_params:
            logger.info("[Params] keine vorhandenen Parameter gefunden, berechne neu")
            normal, projection = self.scale_estimator.determine_scales(
                cfg, single_cloud
            )
            # --- Save determined parameters
            out_base = os.path.join(cfg.data_dir, cfg.folder_id)

            self._save_params(cfg, normal, projection, out_base)
        
        else:
            logger.error("[Params] Ungültige/Fehlende Parameter in Config")

        # --- Compute statistics

        try:
            logger.info("[Statistics] Berechne Statistiken …")
            self.statistics_runner.single_cloud_statistics_handler(
                cfg, single_cloud, normal
            )
        except (IOError, ValueError):
            logger.exception("Fehler bei der Berechnung der Statistik")
        except RuntimeError:
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
        try:
            with open(params_path, "w") as f:
                f.write(f"NormalScale={normal}\nSearchScale={projection}\n")
        except OSError:
            logger.exception("[Params] speichern fehlgeschlagen: %s", params_path)
            raise
        logger.info("[Params] gespeichert: %s", params_path)


    def _handle_existing_params(self, cfg: PipelineConfig, out_base: str, tag: str):
        """Load previously determined M3C2 scale parameters.

        Parameters
        ----------
        cfg : PipelineConfig
            Configuration used to derive the expected parameter file name via
            ``cfg.process_python_CC``.
        out_base : str
            Directory in which the parameter file is searched.
        tag : str
            Run identifier appended to the file name.

        Returns
        -------
        Tuple[float, float]
            ``(normal_scale, search_scale)`` read from
            ``{out_base}/{cfg.process_python_CC}_{tag}_m3c2_params.txt`` if both
            values are present.  Otherwise ``(numpy.nan, numpy.nan)`` is
            returned.
        """

        # -------------------------------------------------
        # 1. Use parameter override of config if exists
        if cfg.normal_override is not None and cfg.proj_override is not None:
            normal, projection = self._handle_override_params(cfg)
        
        # -------------------------------------------------
        # Use parameters of file if exists
        elif cfg.normal_override is None and cfg.proj_override is None:
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
        
        # -------------------------------------------------
        # Otherwise parameters don't exist
        else:
            logger.info("[Params] keine vorhandenen Parameter gefunden")
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
            
    def _handle_override_params(self, cfg: PipelineConfig):
        """Load previously determined M3C2 scale parameters in config.

        Parameters
        ----------
        cfg : PipelineConfig
            Configuration used to derive the expected parameter configs.

        Returns
        -------
        Tuple[float, float]

        """

        # -------------------------------------------------
        # Use parameter override of config if exists
        if cfg.normal_override is not None and cfg.proj_override is not None:
            normal = cfg.normal_override
            projection = cfg.proj_override
            logger.info(
                "[Params] Überschreibe mit: (NormalScale=%.6f, SearchScale=%.6f)",
                normal,
                projection,
            )
            return normal, projection
        
        # -------------------------------------------------
        # Otherwise parameters don't exist
        else:
            logger.info("[Params] keine vorhandenen Parameter gefunden")
            return np.nan, np.nan
    