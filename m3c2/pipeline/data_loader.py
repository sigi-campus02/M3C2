"""Utility class for loading input data for the M3C2 pipeline."""
from __future__ import annotations

import logging
import os
import time

import numpy as np

from m3c2.config.datasource_config import DataSourceConfig
from m3c2.io.datasource import DataSource

logger = logging.getLogger(__name__)


class DataLoader:
    """Load point cloud data and core points according to a configuration."""


    def load_data(self, cfg, mode: str) -> tuple[DataSource, object, object, object] | object:

        """Load point clouds and core points according to ``cfg``.

        Parameters
        ----------
        cfg : PipelineConfig
            Configuration defining the location of the point clouds and
            which epoch to use for the core points.
        mode : str
            ``"multicloud"`` to load moving and reference epochs or
            ``"singlecloud"`` to load a single epoch.

        Returns
        -------
        tuple or object
            For ``"multicloud"`` returns ``(ds, mov, ref, corepoints)``
            containing the :class:`DataSource` used for loading, the moving
            and reference epochs and a NumPy array of the core point
            coordinates.  For ``"singlecloud"`` returns only the single cloud
            epoch.

        Side Effects
        ------------
        Reads the point cloud files from disk and emits log messages about the
        loaded data.

        Notes
        -----
        This method is part of the public pipeline API.
        """
        if mode == "multicloud":
            return self._load_data_multi(cfg)

        if mode == "singlecloud":
            return self._load_data_single(cfg)

        raise ValueError(f"Unknown mode '{mode}'")


    def _load_data_multi(self, cfg) -> tuple[DataSource, object, object, object]:
        """Load multi-cloud data according to ``cfg``.

        Parameters
        ----------
        cfg : PipelineConfig
            Configuration defining the location of the point clouds and
            which epoch to use for the core points.

        Returns
        -------
        tuple
            ``(ds, mov, ref, corepoints)`` containing the :class:`DataSource`
            used for loading, the moving and reference epochs and a NumPy array
            of the core point coordinates.
        """
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

    def _load_data_single(self, cfg) -> object:
        """Load single-cloud data according to ``cfg``.

        Parameters
        ----------
        cfg : PipelineConfig
            Configuration defining the location of the point clouds and
            which epoch to use for the core points.

        Returns
        -------
        object
            The single cloud epoch.
        """
        t0 = time.perf_counter()

        ds_config = DataSourceConfig(
            os.path.join(cfg.data_dir, cfg.folder_id),
            cfg.filename_singlecloud,
        )
        ds_single = DataSource(ds_config)

        single_cloud = ds_single.load_points_singlecloud()

        logger.info(
            "[Load] data/%s: single_cloud=%s | %.3fs",
            cfg.folder_id,
            getattr(single_cloud, "cloud", np.array([])).shape if hasattr(single_cloud, "cloud") else "Epoch",
            time.perf_counter() - t0,
        )
        return single_cloud
