# BatchOrchestrator.py
from __future__ import annotations

import logging
import os
import time
from typing import Dict, List

import numpy as np
import pandas as pd
from pipeline_config import PipelineConfig
from datasource import DataSource
from strategies import ScaleStrategy, RadiusScanStrategy, VoxelScanStrategy, ScaleScan
from param_estimator import ParamEstimator
from m3c2_runner import M3C2Runner
from statistics_service import StatisticsService
from visualization_service import VisualizationService


class BatchOrchestrator:
    """
    - Lädt Daten (DataSource)
    - Schätzt/übernimmt Scales (ParamEstimator + Strategy)
    - Run mit py4dgeo (M3C2Runner)
    - Statistiken + Visuals (Services)
    - Loggt detailliert jeden Schritt
    """

    def __init__(
        self,
        configs: List[PipelineConfig],
        strategy: str,
        sample_size: int | None = None,
        log_level: int = logging.INFO,
    ) -> None:
        self.configs = configs
        self.strategy_name = strategy.lower().strip()
        self.sample_size = sample_size
        logging.basicConfig(level=log_level, format="%(asctime)s [%(levelname)s] %(message)s")
        self.strategy: ScaleStrategy = self._resolve_strategy(self.strategy_name, self.sample_size)

        logging.info("=== BatchOrchestrator initialisiert ===")
        logging.info("Konfigurationen: %d Jobs", len(self.configs))
        logging.info("Strategie: %s (sample_size=%s)", type(self.strategy).__name__, str(self.sample_size))

        # Zusatzinfos zur Strategie (falls vorhanden)
        if isinstance(self.strategy, RadiusScanStrategy):
            logging.info("  RadiusScan: multipliers=%s min_points=%s signed=%s",
                         getattr(self.strategy, "multipliers", None),
                         getattr(self.strategy, "min_points", None),
                         getattr(self.strategy, "signed", None))
        if isinstance(self.strategy, VoxelScanStrategy):
            logging.info("  VoxelScan: steps=%s start_pow=%s min_points=%s",
                         getattr(self.strategy, "steps", None),
                         getattr(self.strategy, "start_pow", None),
                         getattr(self.strategy, "min_points", None))

    def _resolve_strategy(self, name: str, sample_size: int | None) -> ScaleStrategy:
        if name in ("radius", "radiusbased", "radius-based"):
            return RadiusScanStrategy(sample_size=sample_size)
        if name in ("voxel", "voxelbased", "voxel-based"):
            return VoxelScanStrategy(sample_size=sample_size)
        raise ValueError(f"Unbekannte Strategie: {name!r}")

    def run_all(self) -> None:
        if not self.configs:
            logging.warning("Keine Konfigurationen – nichts zu tun.")
            return

        stats_rows: List[Dict] = []

        for cfg in self.configs:
            logging.info(f"{cfg.folder_id}, {cfg.filename_mov}, {cfg.filename_ref}")
            t_job0 = time.perf_counter()

            try:
                # 1) Daten laden
                t0 = time.perf_counter()
                ds = DataSource(cfg.folder_id, cfg.filename_mov, cfg.filename_ref, cfg.mov_as_corepoints, cfg.use_subsampled_corepoints)
                mov, ref, corepoints = ds.load_points()

                t1 = time.perf_counter()

                logging.info("[Load] data/%s: mov=%s, ref=%s, corepoints=%s | %.3fs",
                             cfg.folder_id,
                             getattr(mov, "cloud", np.array([])).shape if hasattr(mov, "cloud") else "Epoch",
                             getattr(ref, "cloud", np.array([])).shape if hasattr(ref, "cloud") else "Epoch",
                             np.asarray(corepoints).shape,
                             t1 - t0)

                # 2) Scales bestimmen (Overrides oder Estimation)
                if cfg.normal_override is not None and cfg.proj_override is not None:
                    normal, projection = cfg.normal_override, cfg.proj_override
                    logging.info("[Scales] Overrides verwendet: normal=%.6f, proj=%.6f", normal, projection)
                else:
                    t2 = time.perf_counter()
                    avg = ParamEstimator.estimate_min_spacing(corepoints)
                    logging.info("[Spacing] avg_spacing=%.6f (k=6) | %.3fs", avg, time.perf_counter() - t2)

                    t3 = time.perf_counter()
                    scans: List[ScaleScan] = ParamEstimator.scan_scales(corepoints, self.strategy, avg)
                    t4 = time.perf_counter()
                    logging.info("[Scan] %d Skalen evaluiert | %.3fs", len(scans), t4 - t3)
                    # kleine Übersicht
                    if scans:
                        top_valid = sorted(scans, key=lambda s: s.valid_normals, reverse=True)[:5]
                        logging.debug("  Top(valid_normals): %s",
                                      [(round(s.scale, 6), int(s.valid_normals)) for s in top_valid])
                        top_smooth = sorted(scans, key=lambda s: (np.nan_to_num(s.roughness, nan=np.inf)))[:5]
                        logging.debug("  Top(min_roughness): %s",
                                      [(round(s.scale, 6), float(s.roughness)) for s in top_smooth])

                    t5 = time.perf_counter()
                    normal, projection = ParamEstimator.select_scales(scans)
                    logging.info("[Select] normal=%.6f, proj=%.6f | %.3fs", normal, projection, time.perf_counter() - t5)

                # 2b) Parameter-Datei speichern (data/<fid>/<version>_m3c2_params.txt)  # NEW
                out_base = ds.folder  # "data/<fid>"
                os.makedirs(out_base, exist_ok=True)  # sicherstellen
                params_path = os.path.join(out_base, f"{cfg.filename_ref}_m3c2_params.txt")
                with open(params_path, "w") as f:
                    f.write(f"NormalScale={normal}\nSearchScale={projection}\n")
                logging.info("[Params] gespeichert: %s", params_path)   

                # 3) Run
                t6 = time.perf_counter()
                runner = M3C2Runner()
                distances, uncertainties = runner.run(mov, ref, corepoints, normal, projection)
                t7 = time.perf_counter()
                n = len(distances)
                nan_share = float(np.isnan(distances).sum()) / n if n else 0.0
                logging.info("[Run] Punkte=%d | NaN=%.2f%% | Zeit=%.3fs", n, 100.0 * nan_share, t7 - t6)


                # 3b) Distanzen abspeichern                                
                dists_path = os.path.join(out_base, f"{cfg.filename_ref}_m3c2_distances.txt")
                np.savetxt(dists_path, distances, fmt="%.6f")
                logging.info("[Run] Distanzen gespeichert: %s (%d Werte, %.2f%% NaN)",
                            dists_path, n, 100.0 * nan_share)
                
                uncert_path = os.path.join(out_base, f"{cfg.filename_ref}_m3c2_uncertainties.txt")
                np.savetxt(uncert_path, uncertainties, fmt="%.6f")
                logging.info("[Run] Unsicherheiten gespeichert: %s", uncert_path)

                # 4) Stats
                logging.info("[Stats] Berechne M3C2-Statistiken …")
                StatisticsService.compute_m3c2_statistics(
                    folder_ids=[cfg.folder_id],
                    filename_ref=cfg.filename_ref,
                    out_xlsx="m3c2_stats_all.xlsx",
                    sheet_name="Results",
                )

                # 5) Visuals
                logging.info("[Visual] Erzeuge Visualisierungen …")
                out_base = ds.folder  # -> "data/<fid>"
                os.makedirs(out_base, exist_ok=True)

                hist_path = os.path.join(out_base, f"{cfg.filename_ref}_histogram.png")
                ply_path = os.path.join(out_base, f"{cfg.filename_ref}_colored_cloud.ply")
                ply_valid_path = os.path.join(out_base, f"{cfg.filename_ref}_colored_cloud_validonly.ply")

                VisualizationService.histogram(distances, path=hist_path)
                logging.info("[Visual] Histogram gespeichert: %s", hist_path)

                colors = VisualizationService.colorize(mov.cloud, distances, outply=ply_path)
                logging.info("[Visual] Farb-PLY gespeichert: %s", ply_path)

                # optional: nur gültige Punkte exportieren
                try:
                    VisualizationService.export_valid(mov.cloud, colors, distances, outply=ply_valid_path)
                    logging.info("[Visual] Valid-PLY gespeichert: %s", ply_valid_path)
                except Exception as e:
                    logging.warning("[Visual] Export valid-only übersprungen: %s", e)

                # Job-Timing
                logging.info("[Job] %s abgeschlossen in %.3fs", cfg.folder_id, time.perf_counter() - t_job0)

            except Exception:
                logging.exception("[Job] Fehler in Job '%s' (Version %s)", cfg.folder_id, cfg.filename_ref)
                # Weiter mit dem nächsten Job
                continue
