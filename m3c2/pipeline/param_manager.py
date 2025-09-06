"""Manage saving and loading of M3C2 scale parameters."""

import logging
import os
from typing import Tuple

import numpy as np

from m3c2.config.pipeline_config import PipelineConfig
from m3c2.statistics.distance_stats import _load_params

logger = logging.getLogger(__name__)


class ParamManager:
    """Handle persistence and retrieval of M3C2 parameters."""

    def save_params(
        self, cfg: PipelineConfig, normal: float, projection: float, out_base: str, tag: str
    ) -> None:
        """Persist determined scale parameters to disk."""
        os.makedirs(out_base, exist_ok=True)
        params_path = os.path.join(
            out_base, f"{cfg.process_python_CC}_{tag}_m3c2_params.txt"
        )
        try:
            with open(params_path, "w") as f:
                f.write(f"NormalScale={normal}\nSearchScale={projection}\n")
        except OSError:
            logger.exception("[Params] speichern fehlgeschlagen: %s", params_path)
            raise
        logger.info("[Params] gespeichert: %s", params_path)

    def handle_existing_params(
        self, cfg: PipelineConfig, out_base: str, tag: str
    ) -> Tuple[float, float]:
        """Load previously determined M3C2 scale parameters."""
        if cfg.normal_override is not None and cfg.proj_override is not None:
            return self.handle_override_params(cfg)

        if cfg.normal_override is None and cfg.proj_override is None:
            params_path = os.path.join(
                out_base, f"{cfg.process_python_CC}_{tag}_m3c2_params.txt"
            )
            normal, projection = _load_params(params_path)
            if not np.isnan(normal) and not np.isnan(projection):
                logger.info(
                    "[Params] geladen: %s (NormalScale=%.6f, SearchScale=%.6f)",
                    params_path,
                    normal,
                    projection,
                )
                return normal, projection

        logger.info("[Params] keine vorhandenen Parameter gefunden")
        return np.nan, np.nan

    def handle_override_params(self, cfg: PipelineConfig) -> Tuple[float, float]:
        """Load scale parameter overrides provided via configuration."""
        if cfg.normal_override is not None and cfg.proj_override is not None:
            normal = cfg.normal_override
            projection = cfg.proj_override
            logger.info(
                "[Params] Überschreibe mit: (NormalScale=%.6f, SearchScale=%.6f)",
                normal,
                projection,
            )
            return normal, projection

        logger.info("[Params] keine vorhandenen Parameter gefunden")
        return np.nan, np.nan
