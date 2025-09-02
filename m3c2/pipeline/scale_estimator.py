"""Determine optimal scales for the M3C2 algorithm."""

from __future__ import annotations

import logging
import time
from typing import List, Tuple

import numpy as np

from m3c2.core.param_estimator import ParamEstimator
from m3c2.pipeline.strategies import ScaleScan, STRATEGIES

logger = logging.getLogger(__name__)


class ScaleEstimator:
    """Estimate normal and projection scales using a scanning strategy."""

    def __init__(self, strategy_name: str = "radius") -> None:
        self.strategy_name = strategy_name

    def determine_scales(self, cfg, corepoints) -> Tuple[float, float]:
        """Determine suitable normal and projection scales.

        This method is part of the public pipeline API.
        """
        if cfg.normal_override is not None and cfg.proj_override is not None:
            normal, projection = cfg.normal_override, cfg.proj_override
            logger.info("[Scales] Overrides verwendet: normal=%.6f, proj=%.6f", normal, projection)
            return normal, projection

        t0 = time.perf_counter()
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
            top_valid = sorted(scans, key=lambda s: s.valid_normals, reverse=True)[:5]
            logger.debug(
                "  Top(valid_normals): %s",  # noqa: G004
                [(round(s.scale, 6), int(s.valid_normals)) for s in top_valid],
            )
            top_smooth = sorted(
                scans, key=lambda s: (np.nan_to_num(s.roughness, nan=np.inf))
            )[:5]
            logger.debug(
                "  Top(min_roughness): %s",  # noqa: G004
                [(round(s.scale, 6), float(s.roughness)) for s in top_smooth],
            )
        else:
            raise ValueError("Scan-Strategie lieferte keine Skalen.")

        t0 = time.perf_counter()
        normal, projection = ParamEstimator.select_scales(scans)
        logger.info("[Select] normal=%.6f, proj=%.6f | %.3fs", normal, projection, time.perf_counter() - t0)
        return normal, projection
