"""Execute the M3C2 algorithm and persist results."""

from __future__ import annotations

import logging
import os
import time
from typing import Tuple

import numpy as np

from m3c2.m3c2_core.m3c2_runner import M3C2Runner

logger = logging.getLogger(__name__)


class M3C2Executor:
    """High level interface for executing the M3C2 algorithm.

    The executor exposes :meth:`run_m3c2`, which accepts a configuration
    object, comparison and reference point clouds, an array of core points and the
    normal and projection scales used by the algorithm.  It delegates the
    heavy computation to :class:`~m3c2.core.m3c2_runner.M3C2Runner` while
    handling logging and persistence of the results.

    Inputs
    ------
    ``cfg``
        Configuration object that must provide ``process_python_CC`` to build
        output file names.
    ``comparison`` and ``reference``
        Point cloud data as ``(N, 3)`` arrays or objects exposing a ``cloud``
        attribute with such an array.
    ``corepoints``
        ``(N, 3)`` coordinates of the core points where distances are
        evaluated.
    ``normal`` / ``projection``
        Floating point scale parameters controlling the local surface fitting
        and projection radius.
    ``out_base`` / ``tag``
        Output directory and filename tag used for the generated text files.

    Outputs
    -------
    A tuple of ``numpy.ndarray`` objects representing the distances and their
    uncertainties, followed by the path to the saved distances file.

    Side Effects
    ------------
    Three ASCII files (distances, distances with coordinates and
    uncertainties) are written to ``out_base`` and informative log messages are
    emitted.
    """

    def run_m3c2(self, cfg, comparison, reference, corepoints, normal: float, projection: float, out_base: str, tag: str,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Run the core M3C2 computation and persist results to disk.

        Parameters
        ----------
        cfg : object
            Configuration object. Must provide the attribute
            ``process_python_CC`` which is used to construct output file
            names.
        comparison : array-like or object
            Comparison point cloud. May be provided as an ``(N, 3)`` array or as
            an object exposing a ``cloud`` attribute containing such an
            array.
        reference : array-like or object
            Reference point cloud in the same format as ``comparison``.
        corepoints : array-like
            ``(N, 3)`` coordinates of the core points where distances are
            evaluated.
        normal : float
            Normal scale used for local surface fitting.
        projection : float
            Projection scale for the M3C2 algorithm.
        out_base : str
            Directory to which result files are written.
        tag : str
            Identifier appended to the generated filenames.

        Returns
        -------
        distances : numpy.ndarray
            Signed distance for each core point. Entries may be ``NaN`` when
            a distance could not be computed.
        uncertainties : numpy.ndarray
            One-sigma distance uncertainty for each core point.

        Notes
        -----
        Three ASCII files are written to ``out_base``:

        * ``{cfg.process_python_CC}_{tag}_m3c2_distances.txt`` – distances.
        * ``{cfg.process_python_CC}_{tag}_m3c2_distances_coordinates.txt`` –
          core point coordinates with distances.
        * ``{cfg.process_python_CC}_{tag}_m3c2_uncertainties.txt`` –
          uncertainty values.

        This method is part of the public pipeline API.
        """

        #------------------------------------------
        # M3C2 computation
        t0 = time.perf_counter()
        runner = M3C2Runner()
        distances, uncertainties = runner.run(comparison, reference, corepoints, normal, projection)

        #------------------------------------------
        # Logging for quick overview of distance cloud metrics and computation time
        duration = time.perf_counter() - t0
        n = len(distances)
        nan_share = float(np.isnan(distances).sum()) / n if n else 0.0

        logger.info("[Run] Punkte=%d | NaN=%.2f%% | Zeit=%.3fs", n, 100.0 * nan_share, duration)

        #------------------------------------------
        # Results saving
        dists_path = os.path.join(out_base, f"{cfg.process_python_CC}_{tag}_m3c2_distances.txt")
        np.savetxt(dists_path, distances, fmt="%.6f")
        logger.info("[Run] Distanzen gespeichert: %s (%d Werte, %.2f%% NaN)", dists_path, n, 100.0 * nan_share)

        coords_path = os.path.join(out_base, f"{cfg.process_python_CC}_{tag}_m3c2_distances_coordinates.txt")
        if hasattr(comparison, "cloud"):
            xyz = np.asarray(comparison.cloud)
        else:
            xyz = np.asarray(comparison)
        if xyz.shape[0] == distances.shape[0]:
            arr = np.column_stack((xyz, distances))
            header = "x y z distance"
            np.savetxt(coords_path, arr, fmt="%.6f", header=header)
            logger.info(f"[Run] Distanzen mit Koordinaten gespeichert: {coords_path}")
        else:
            logger.warning(
                f"[Run] Anzahl Koordinaten stimmt nicht mit Distanzen überein: {xyz.shape[0]} vs {distances.shape[0]}"
            )

        uncert_path = os.path.join(out_base, f"{cfg.process_python_CC}_{tag}_m3c2_uncertainties.txt")
        np.savetxt(uncert_path, uncertainties, fmt="%.6f")
        logger.info("[Run] Unsicherheiten gespeichert: %s", uncert_path)

        return distances, uncertainties, dists_path
