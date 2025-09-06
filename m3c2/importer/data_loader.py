"""Utility class for loading input data for the M3C2 pipeline."""

from __future__ import annotations

import logging
import os
import time

import numpy as np

from m3c2.config.datasource_config import DataSourceConfig
from m3c2.importer.datasource import DataSource

logger = logging.getLogger(__name__)


class DataLoader:
    """Load point cloud data and core points according to a configuration."""

    def load_data(self, config, mode: str) -> tuple[DataSource, object, object, object] | object:
        """Load point clouds and core points according to ``config``.

        Parameters
        ----------
        config : PipelineConfig
            Configuration defining the location of the point clouds and
            which epoch to use for the core points.
        mode : str
            ``"multicloud"`` to load comparison and reference epochs or
            ``"singlecloud"`` to load a single epoch.

        Returns
        -------
        tuple or object
            For ``"multicloud"`` returns ``(data_source, comparison, reference, corepoints)``
            containing the :class:`DataSource` used for loading, the comparison
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
        # Delegate based on mode. Configuration is passed through to the helper
        # which knows how to interpret it.
        if mode == "multicloud":
            # Return the data source along with comparison, reference and core points
            return self._load_data_multi(config)

        if mode == "singlecloud":
            # Return only the single epoch loaded from the data source
            return self._load_data_single(config)

        # Reject invalid modes early to help debugging
        raise ValueError(f"Unknown mode '{mode}'")

    def _load_data_multi(self, config) -> tuple[DataSource, object, object, object]:
        """Load multi-cloud data according to ``config``.

        Parameters
        ----------
        config : PipelineConfig
            Configuration defining the location of the point clouds and
            which epoch to use for the core points.

        Returns
        -------
        tuple
            ``(data_source, comparison, reference, corepoints)`` containing the :class:`DataSource`
            used for loading, the comparison and reference epochs and a NumPy array
            of the core point coordinates.
        """
        # Start timing the loading operation
        start_time = time.perf_counter()

        # Build a DataSourceConfig from the pipeline configuration
        datasource_config = DataSourceConfig(
            folder=os.path.join(config.data_dir, config.folder_id),
            comparison_basename=config.filename_comparison,
            reference_basename=config.filename_reference,
            use_subsampled_corepoints=config.use_subsampled_corepoints,
        )
        # Create a DataSource using the configuration above
        data_source = DataSource(datasource_config)

        # Load comparison and reference epochs along with core points
        comparison, reference, corepoints = data_source.load_points()

        # Log the shapes of the loaded data and how long it took
        logger.info(
            "[Load] data/%s: comparison=%s, reference=%s, corepoints=%s | %.3fs",
            config.folder_id,
            getattr(comparison, "cloud", np.array([])).shape if hasattr(comparison, "cloud") else "Epoch",
            getattr(reference, "cloud", np.array([])).shape if hasattr(reference, "cloud") else "Epoch",
            np.asarray(corepoints).shape,
            time.perf_counter() - start_time,
        )
        # Return the data source and the loaded epochs
        return data_source, comparison, reference, corepoints

    def _load_data_single(self, config) -> object:
        """Load single-cloud data according to ``config``.

        Parameters
        ----------
        config : PipelineConfig
            Configuration defining the location of the point clouds and
            which epoch to use for the core points.

        Returns
        -------
        object
            The single cloud epoch.
        """
        # Time the loading of the single cloud
        start_time = time.perf_counter()

        # Build the data source configuration for a single cloud
        datasource_config = DataSourceConfig(
            folder=os.path.join(config.data_dir, config.folder_id),
            filename_singlecloud=config.filename_singlecloud,
        )
        # Instantiate the DataSource with the prepared configuration
        single_data_source = DataSource(datasource_config)

        # Load the single cloud epoch
        single_cloud = single_data_source.load_points_singlecloud()

        # Log the shape of the loaded cloud and the elapsed time
        logger.info(
            "[Load] data/%s: single_cloud=%s | %.3fs",
            config.folder_id,
            getattr(single_cloud, "cloud", np.array([])).shape if hasattr(single_cloud, "cloud") else "Epoch",
            time.perf_counter() - start_time,
        )
        # Return the loaded single cloud
        return single_cloud
