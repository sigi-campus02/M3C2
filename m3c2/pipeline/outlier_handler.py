"""Pipeline adapter for excluding M3C2 distance outliers.

This module provides a thin wrapper around
``m3c2.archive_moduls.exclude_outliers.exclude_outliers`` that operates on
pipeline concepts.  It translates the ``data_folder`` and ``ref_variant``
parameters used throughout the pipeline into the distance file path expected
by the archived routine.
"""

from __future__ import annotations

import os

from m3c2.archive_moduls.exclude_outliers import exclude_outliers as _exclude_outliers


def exclude_outliers(
    data_folder: str, ref_variant: str, method: str, outlier_multiplicator: float
) -> None:
    """Remove statistical outliers from a set of distance measurements.

    Parameters
    ----------
    data_folder : str
        Directory containing the distance results.
    ref_variant : str
        Identifier used to compose the distance file name.
    method : str
        Statistical method used to compute the outlier threshold.
    outlier_multiplicator : float
        Multiplier applied to the chosen statistic when deriving the threshold.

    Notes
    -----
    The expected distance file is
    ``{data_folder}/python_{ref_variant}_m3c2_distances_coordinates.txt``.
    The heavy lifting is delegated to
    :func:`m3c2.archive_moduls.exclude_outliers.exclude_outliers`.
    """

    dists_path = os.path.join(
        data_folder, f"python_{ref_variant}_m3c2_distances_coordinates.txt"
    )
    _exclude_outliers(
        dists_path=dists_path, method=method, outlier_multiplicator=outlier_multiplicator
    )


__all__ = ["exclude_outliers"]

