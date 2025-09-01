from __future__ import annotations
"""High level orchestration for running the complete M3C2 workflow.

The :class:`BatchOrchestrator` coordinates configuration handling, parameter
estimation, execution of the core algorithm and subsequent analysis steps.
"""

import logging
import os
import time
from typing import List, Tuple
import numpy as np
from m3c2.io.datasource import DataSource
from m3c2.config.pipeline_config import PipelineConfig
from m3c2.core.param_estimator import ParamEstimator
from m3c2.core.statistics import StatisticsService
from m3c2.core.exclude_outliers import exclude_outliers
from m3c2.visualization.visualization_service import VisualizationService
from m3c2.core.m3c2_runner import M3C2Runner
from m3c2.pipeline.strategies import ScaleScan, STRATEGIES
from m3c2.config.datasource_config import DataSourceConfig

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
        """Create a new orchestrator instance.

        Parameters
        ----------
        configs : list[PipelineConfig]
            Configurations describing each job to run.
        strategy : str, optional
            Name of the scale-scanning strategy to use.  Currently only
            ``"radius"`` is implemented but the parameter allows an easy
            extension in the future.
        """

        self.configs = configs
        self.strategy_name = strategy
        # The output format is identical for all configs; take it from the
        # first configuration for convenience.
        self.output_format = configs[0].output_format if configs else "excel"

        # Log basic information about the incoming batch
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
            except Exception:
                logger.exception("[Job] Fehler in Job '%s' (Version %s)", cfg.folder_id, cfg.filename_ref)

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

        # Load input data and determine core points
        ds, mov, ref, corepoints = self._load_data(cfg)

        if cfg.process_python_CC == "python" and not cfg.only_stats:
            # Run M3C2 only when full processing (beyond statistics) is requested
            out_base = ds.config.folder
            tag = self._run_tag(cfg)
            normal = projection = np.nan
            if cfg.use_existing_params:
                # Reuse previously estimated parameters if available
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
                    logger.info(
                        "[Params] keine vorhandenen Parameter gefunden, berechne neu",
                    )
            if np.isnan(normal) or np.isnan(projection):
                # Estimate optimal scales and persist them for later runs
                normal, projection = self._determine_scales(cfg, corepoints)
                self._save_params(cfg, normal, projection, out_base)
            distances, _ = self._run_m3c2(
                cfg, mov, ref, corepoints, normal, projection, out_base
            )

            # Visualise distances including those flagged as outliers
            self._generate_visuals(cfg, mov, distances, out_base)

        try:
            # Generate distance files excluding outliers for further analysis
            logger.info("[Outlier] Entferne Ausreißer für %s", cfg.folder_id)
            self._exclude_outliers(cfg, ds.config.folder)
        except Exception:
            logger.exception("Fehler beim Entfernen von Ausreißern")

        try:
            # Create coloured point clouds showing outliers and inliers
            logger.info("[Outlier] Erzeuge .ply Dateien für Outliers / Inliers …")
            self._generate_clouds_outliers(cfg, ds.config.folder)
        except Exception:
            logger.exception("Fehler beim Erzeugen von .ply Dateien für Ausreißer / Inlier")

        try:
            # Compute summary statistics for the processed dataset
            self._compute_statistics(cfg, ref)
        except Exception:
            logger.exception("Fehler bei der Berechnung der Statistik")

        logger.info("[Job] %s abgeschlossen in %.3fs", cfg.folder_id, time.perf_counter() - start)

    def _load_data(self, cfg: PipelineConfig) -> Tuple[DataSource, object, object, object]:
        """Load point cloud data and core points according to the configuration.

        Parameters
        ----------
        cfg : PipelineConfig
            Configuration specifying file names and loading options.

        Returns
        -------
        tuple
            ``(DataSource, moving, reference, corepoints)`` where ``moving`` and
            ``reference`` are objects understood by downstream services.
        """
        t0 = time.perf_counter()
        # Create a data source that knows how to load the required clouds
        
        ds_config = DataSourceConfig(
            cfg.folder_id,
            cfg.filename_mov,
            cfg.filename_ref,
            cfg.mov_as_corepoints,
            cfg.use_subsampled_corepoints
        )
        ds = DataSource(ds_config)
        mov, ref, corepoints = ds.load_points()

        logger.info(
            "[Load] data/%s: mov=%s, ref=%s, corepoints=%s | %.3fs",
            cfg.folder_id,
            getattr(mov, "cloud", np.array([])).shape if hasattr(mov, "cloud") else "Epoch",
            getattr(ref, "cloud", np.array([])).shape if hasattr(ref, "cloud") else "Epoch",
            np.asarray(corepoints).shape,
            time.perf_counter() - t0,
        )
        return ds, mov, ref, corepoints

    def _determine_scales(self, cfg: PipelineConfig, corepoints) -> Tuple[float, float]:
        """Determine suitable normal and projection scales.

        Parameters
        ----------
        cfg : PipelineConfig
            Configuration possibly providing override values.
        corepoints : array-like
            Points used for scale estimation.

        Returns
        -------
        tuple of float
            Chosen normal and projection scales.
        """
        if cfg.normal_override is not None and cfg.proj_override is not None:
            # Respect user-specified override parameters
            normal, projection = cfg.normal_override, cfg.proj_override
            logger.info("[Scales] Overrides verwendet: normal=%.6f, proj=%.6f", normal, projection)
            return normal, projection

        t0 = time.perf_counter()
        # Instantiate the desired scanning strategy for the current
        # configuration.  ``STRATEGIES`` maps a user-facing name to the
        # corresponding class.  This design keeps the orchestrator agnostic of
        # concrete strategy implementations.
        try:
            strategy_cls = STRATEGIES[self.strategy_name]
        except KeyError as exc:
            raise ValueError(f"Unbekannte Strategie: {self.strategy_name}") from exc

        strategy = strategy_cls(sample_size=cfg.sample_size)
        estimator = ParamEstimator(strategy=strategy)

        avg = estimator.estimate_min_spacing(corepoints)
        logger.info("[Spacing] avg_spacing=%.6f (k=6) | %.3fs", avg, time.perf_counter() - t0)

        t0 = time.perf_counter()
        scans: List[ScaleScan] = estimator.scan_scales(corepoints, avg)
        logger.info("[Scan] %d Skalen evaluiert | %.3fs", len(scans), time.perf_counter() - t0)

        if scans:
            # Log a few of the most promising scales for debugging
            top_valid = sorted(scans, key=lambda s: s.valid_normals, reverse=True)[:5]
            logger.debug("  Top(valid_normals): %s", [(round(s.scale, 6), int(s.valid_normals)) for s in top_valid])
            top_smooth = sorted(scans, key=lambda s: (np.nan_to_num(s.roughness, nan=np.inf)))[:5]
            logger.debug("  Top(min_roughness): %s", [(round(s.scale, 6), float(s.roughness)) for s in top_smooth])

        t0 = time.perf_counter()
        normal, projection = ParamEstimator.select_scales(scans)
        logger.info("[Select] normal=%.6f, proj=%.6f | %.3fs", normal, projection, time.perf_counter() - t0)
        return normal, projection

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


    def _run_m3c2(self, cfg: PipelineConfig, mov, ref, corepoints, normal: float, projection: float, out_base: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run the M3C2 algorithm and persist results to disk.

        Parameters
        ----------
        cfg : PipelineConfig
            Current job configuration.
        mov, ref : object
            Moving and reference point clouds or epochs.
        corepoints : array-like
            Core points at which distances are evaluated.
        normal : float
            Normal scale to use in the computation.
        projection : float
            Projection scale to use in the computation.
        out_base : str
            Directory where result files are written.

        Returns
        -------
        tuple of ndarray
            Arrays of distances and their associated uncertainties.
        """

        tag = self._run_tag(cfg)

        t0 = time.perf_counter()
        runner = M3C2Runner()
        distances, uncertainties = runner.run(mov, ref, corepoints, normal, projection)
        duration = time.perf_counter() - t0
        n = len(distances)
        nan_share = float(np.isnan(distances).sum()) / n if n else 0.0
        logger.info("[Run] Punkte=%d | NaN=%.2f%% | Zeit=%.3fs", n, 100.0 * nan_share, duration)

        dists_path = os.path.join(out_base, f"{cfg.process_python_CC}_{tag}_m3c2_distances.txt")
        np.savetxt(dists_path, distances, fmt="%.6f")
        logger.info("[Run] Distanzen gespeichert: %s (%d Werte, %.2f%% NaN)", dists_path, n, 100.0 * nan_share)

        # Store XYZ coordinates alongside the computed distances
        coords_path = os.path.join(out_base, f"{cfg.process_python_CC}_{tag}_m3c2_distances_coordinates.txt")

        # Handle both Epoch objects (with .cloud) and raw numpy arrays
        if hasattr(mov, "cloud"):
            xyz = np.asarray(mov.cloud)
        else:
            xyz = np.asarray(mov)
        if xyz.shape[0] == distances.shape[0]:
            arr = np.column_stack((xyz, distances))
            header = "x y z distance"
            np.savetxt(coords_path, arr, fmt="%.6f", header=header)
            logger.info(f"[Run] Distanzen mit Koordinaten gespeichert: {coords_path}")
        else:
            logger.warning(f"[Run] Anzahl Koordinaten stimmt nicht mit Distanzen überein: {xyz.shape[0]} vs {distances.shape[0]}")

        uncert_path = os.path.join(out_base, f"{cfg.process_python_CC}_{tag}_m3c2_uncertainties.txt")
        np.savetxt(uncert_path, uncertainties, fmt="%.6f")
        logger.info("[Run] Unsicherheiten gespeichert: %s", uncert_path)

        return distances, uncertainties

    def _exclude_outliers(self, cfg: PipelineConfig, out_base: str) -> None:
        """Remove statistical outliers from the distance results.

        Parameters
        ----------
        cfg : PipelineConfig
            Configuration containing outlier detection settings.
        out_base : str
            Directory where distance files are stored.
        """
        tag = self._run_tag(cfg)
        exclude_outliers(
            data_folder=out_base,
            ref_variant=tag,
            method=cfg.outlier_detection_method,
            outlier_multiplicator=cfg.outlier_multiplicator
        )


    def _compute_statistics(self, cfg: PipelineConfig, ref) -> None:
        """Compute and export statistical summaries for the run.

        Parameters
        ----------
        cfg : PipelineConfig
            Configuration describing which statistics to compute.
        ref : object
            Reference cloud used for statistics depending on mode.
        """
        tag = self._run_tag(cfg)
        if cfg.stats_singleordistance == "distance":
            logger.info(f"[Stats on Distance] Berechne M3C2-Statistiken {cfg.folder_id},{cfg.filename_ref} …")

            if self.output_format == "excel":
                out_path = os.path.join(f"outputs/{cfg.project}_output/{cfg.project}_m3c2_stats_distances.xlsx")
            elif self.output_format == "json":
                out_path = os.path.join(f"outputs/{cfg.project}_output/{cfg.project}_m3c2_stats_distances.json")
            else:
                raise ValueError("Ungültiges Ausgabeformat. Verwenden Sie 'excel' oder 'json'.")

            StatisticsService.compute_m3c2_statistics(
                folder_ids=[cfg.folder_id],
                filename_ref=tag,
                process_python_CC=cfg.process_python_CC,
                out_path=out_path,
                sheet_name="Results",
                output_format=self.output_format,
                outlier_multiplicator=cfg.outlier_multiplicator,
                outlier_method=cfg.outlier_detection_method
            )

        if cfg.stats_singleordistance == "single":
            logger.info(
                f"[Stats on SingleClouds] Berechne M3C2-Statistiken {cfg.folder_id},{cfg.filename_ref} …",
            )

            if self.output_format == "excel":
                out_path = os.path.join(f"outputs/{cfg.project}_output/{cfg.project}_m3c2_stats_clouds.xlsx")
            elif self.output_format == "json":
                out_path = os.path.join(f"outputs/{cfg.project}_output/{cfg.project}_m3c2_stats_clouds.json")
            else:
                raise ValueError("Ungültiges Ausgabeformat. Verwenden Sie 'excel' oder 'json'.")

            StatisticsService.calc_single_cloud_stats(
                folder_ids=[cfg.folder_id],
                filename_mov=cfg.filename_mov,
                filename_ref=cfg.filename_ref,
                out_path=out_path,
                sheet_name="CloudStats",
                output_format=self.output_format
            )

    def _generate_visuals(self, cfg: PipelineConfig, mov, distances: np.ndarray, out_base: str) -> None:
        """Create graphical and point cloud representations of results.

        Parameters
        ----------
        cfg : PipelineConfig
            Configuration for file naming.
        mov : object
            Moving point cloud containing coordinates.
        distances : ndarray
            Array of M3C2 distances corresponding to the moving cloud.
        out_base : str
            Directory to place the generated files in.
        """
        logger.info("[Visual] Erzeuge Visualisierungen …")
        tag = self._run_tag(cfg)
        os.makedirs(out_base, exist_ok=True)

        hist_path = os.path.join(out_base, f"{cfg.process_python_CC}_{tag}_histogram.png")
        ply_path = os.path.join(out_base, f"{cfg.process_python_CC}_{tag}_includenonvalid.ply")
        ply_valid_path = os.path.join(out_base, f"{cfg.process_python_CC}_{tag}.ply")

        VisualizationService.histogram(distances, path=hist_path)
        logger.info("[Visual] Histogram gespeichert: %s", hist_path)

        colors = VisualizationService.colorize(mov.cloud, distances, outply=ply_path)
        logger.info("[Visual] Farb-PLY gespeichert: %s", ply_path)

        try:
            VisualizationService.export_valid(mov.cloud, colors, distances, outply=ply_valid_path)
            logger.info("[Visual] Valid-PLY gespeichert: %s", ply_valid_path)
        except Exception as exc:
            logger.warning("[Visual] Export valid-only übersprungen: %s", exc)

    def _generate_clouds_outliers(self, cfg: PipelineConfig, out_base: str) -> None:
        """Create coloured point clouds for inliers and outliers.

        Parameters
        ----------
        cfg : PipelineConfig
            Configuration providing file naming and outlier settings.
        out_base : str
            Directory where output files are written.
        """
        logger.info("[Visual] Erzeuge .ply Dateien für Outliers / Inliers …")
        os.makedirs(out_base, exist_ok=True)
        tag = self._run_tag(cfg)

        ply_valid_path_outlier = os.path.join(out_base, f"{cfg.process_python_CC}_{tag}_outlier_{cfg.outlier_detection_method}.ply")
        ply_valid_path_inlier = os.path.join(out_base, f"{cfg.process_python_CC}_{tag}_inlier_{cfg.outlier_detection_method}.ply")
        txt_path_outlier = os.path.join(out_base, f"python_{tag}_m3c2_distances_coordinates_outlier_{cfg.outlier_detection_method}.txt")
        txt_path_inlier = os.path.join(out_base, f"python_{tag}_m3c2_distances_coordinates_inlier_{cfg.outlier_detection_method}.txt")

        try:
            VisualizationService.txt_to_ply_with_distance_color(
                txt_path=txt_path_outlier,
                outply=ply_valid_path_outlier
            )
        except Exception as exc:
            logger.warning("[Visual] Export valid-only übersprungen: %s", exc)

        try:
            VisualizationService.txt_to_ply_with_distance_color(
                txt_path=txt_path_inlier,
                outply=ply_valid_path_inlier
            )
        except Exception as exc:
            logger.warning("[Visual] Export valid-only übersprungen: %s", exc)
