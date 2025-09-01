"""Thin wrapper around the :mod:`py4dgeo` M3C2 implementation.

The runner hides the instantiation details of the external library and
returns only the computed distances and uncertainties.  It exists mainly to
isolate direct dependencies on :mod:`py4dgeo` from higher level orchestration
code.
"""

from __future__ import annotations
import numpy as np
import importlib
from typing import Tuple


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
        Exception
            Propagated from :mod:`py4dgeo` if the computation fails.
        """
        # Import py4dgeo lazily to ease testing and optional dependency handling
        py4dgeo = importlib.import_module("py4dgeo")
        # Create the py4dgeo object that performs the actual computation
        m3c2 = py4dgeo.M3C2(
            epochs=(mov, ref),
            corepoints=corepoints,
            cyl_radius=projection,
            normal_radii=[normal],
        )
        distances, uncertainties = m3c2.run()
        return distances, uncertainties
