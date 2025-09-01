from __future__ import annotations
"""Utility class for loading input data for the M3C2 pipeline."""

import logging
import os
import time
from typing import Tuple

import numpy as np

from m3c2.config.datasource_config import DataSourceConfig
from m3c2.io.datasource import DataSource

logger = logging.getLogger(__name__)


class DataLoader:
    """Load point cloud data and core points according to a configuration."""

    def _load_data(self, cfg) -> Tuple[DataSource, object, object, object]:
        """Load point clouds and core points as specified by ``cfg``."""
        t0 = time.perf_counter()
        ds_config = DataSourceConfig(
            os.path.join(cfg.data_dir, cfg.folder_id),
            cfg.filename_mov,
            cfg.filename_ref,
            cfg.mov_as_corepoints,
            cfg.use_subsampled_corepoints,
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
