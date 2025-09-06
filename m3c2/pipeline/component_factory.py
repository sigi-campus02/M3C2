"""Factory for creating pipeline components used by :class:`BatchOrchestrator`."""

from __future__ import annotations

import logging

from m3c2.importer.data_loader import DataLoader
from m3c2.m3c2_core.param_handler.scale_estimator import ScaleEstimator
from m3c2.m3c2_core.m3c2_executor import M3C2Executor
from m3c2.statistics.outlier_handler import OutlierHandler
from m3c2.statistics.statistics_runner import StatisticsRunner

logger = logging.getLogger(__name__)


class PipelineComponentFactory:
    """Create configured instances of pipeline helper classes."""

    def __init__(self, strategy_name: str, output_format: str) -> None:
        """Configure the factory with strategy and export format.

        Parameters
        ----------
        strategy_name:
            Name of the scale estimation strategy used by created components.
        output_format:
            Output format for statistics produced in later pipeline stages,
            e.g. ``"excel"`` or ``"json"``.
        """
        self.strategy_name = strategy_name
        self.output_format = output_format
        logger.debug(
            "Initialized PipelineComponentFactory with strategy_name=%s, output_format=%s",
            strategy_name,
            output_format,
        )

    def create_data_loader(self) -> DataLoader:
        """Return a new :class:`DataLoader` instance."""

        logger.debug("Creating %s", DataLoader.__name__)
        return DataLoader()

    def create_scale_estimator(self) -> ScaleEstimator:
        """Create a :class:`ScaleEstimator` for the configured strategy.

        The factory's ``strategy_name`` determines which scale scanning
        strategy the estimator will employ to derive the normal and
        projection scales for the M3C2 algorithm.

        Returns
        -------
        ScaleEstimator
            Estimator instance initialised with the selected strategy.
        """
        logger.debug("Creating %s", ScaleEstimator.__name__)
        return ScaleEstimator(strategy_name=self.strategy_name)

    def create_m3c2_executor(self) -> M3C2Executor:
        """Instantiate the :class:`M3C2Executor` used for processing."""
        logger.debug("Creating %s", M3C2Executor.__name__)
        return M3C2Executor()

    def create_outlier_handler(self) -> OutlierHandler:
        """Instantiate an :class:`OutlierHandler` for recomparison statistical outliers.

        The returned handler applies the configured outlier detection method to the
        generated M3C2 results and filters out measurements deemed to be outliers.
        This ensures downstream statistics and visualizations are based on
        cleaned data.
        """
        logger.debug("Creating %s", OutlierHandler.__name__)
        return OutlierHandler()

    def create_statistics_runner(self) -> StatisticsRunner:
        """Create a :class:`StatisticsRunner` honoring the output format.

        The returned runner serializes computed statistics either as Excel
        spreadsheets or JSON files depending on the ``output_format`` specified
        when this factory was instantiated.
        """
        logger.debug("Creating %s", StatisticsRunner.__name__)
        return StatisticsRunner(self.output_format)

