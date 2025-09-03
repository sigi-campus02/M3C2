"""Thin wrapper around the :mod:`py4dgeo` M3C2 implementation.

The runner hides the instantiation details of the external library and
returns only the computed distances and uncertainties.  It exists mainly to
isolate direct dependencies on :mod:`py4dgeo` from higher level orchestration
code.
"""

from __future__ import annotations

import importlib
import logging
from typing import Tuple

import numpy as np


logger = logging.getLogger(__name__)


class M3C2Runner:
    """Execute the M3C2 algorithm for a pair of point clouds."""

    @staticmethod
    def run(
        mov: object,
        ref: object,
        corepoints: np.ndarray,
        normal: float,
        projection: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run the M3C2 algorithm.

        Parameters
        ----------
        mov, ref : :class:`py4dgeo.Epoch`
            The moving and reference epochs to compare.
        corepoints : ndarray
            Array of core point coordinates where distances are evaluated.
        normal : float
            Normal scale used for surface orientation estimation.
        projection : float
            Projection radius used for distance computation.

        Returns
        -------
        distances : ndarray
            Signed distances for each core point.
        uncertainties : ndarray
            Uncertainty estimate for each distance value.

        Raises
        ------
        ImportError
            If the optional dependency :mod:`py4dgeo` is missing.
        py4dgeo.Py4DGEOError or RuntimeError
            Propagated from :mod:`py4dgeo` if the computation fails.
        """
        logger.info(
            "Starting M3C2 run with normal=%s and projection=%s", normal, projection
        )

        # Import py4dgeo lazily to ease testing and optional dependency handling
        try:
            py4dgeo = importlib.import_module("py4dgeo")
        except ImportError as err:  # pragma: no cover - optional dependency
            logger.exception("Failed to import py4dgeo: %s", err)
            raise

        try:
            # Create the py4dgeo object that performs the actual computation
            m3c2 = py4dgeo.M3C2(
                epochs=(mov, ref),
                corepoints=corepoints,
                cyl_radius=projection,
                normal_radii=[normal],
            )
            distances, uncertainties = m3c2.run()
        except (py4dgeo.Py4DGEOError, RuntimeError) as err:  # pragma: no cover - py4dgeo exceptions
            logger.exception("M3C2 run failed: %s", err)
            raise

        nan_ratio = float(np.isnan(distances).mean()) if distances.size else 0.0
        logger.info(
            "Completed M3C2 run: %d distances (NaN ratio %.2f%%)",
            distances.size,
            nan_ratio * 100,
        )
        return distances, uncertainties
