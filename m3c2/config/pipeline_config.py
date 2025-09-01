"""Data container describing a single run of the M3C2 pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PipelineConfig:
    """Configuration parameters for one pair-wise M3C2 comparison.

    Attributes
    ----------
    folder_id:
        Identifier of the data folder containing the point clouds.
    filename_mov:
        File name of the moving point cloud.
    filename_ref:
        File name of the reference point cloud.
    mov_as_corepoints:
        Whether to use the moving cloud as core points.
    use_subsampled_corepoints:
        Number of subsampled core points to use (0 disables
        subsampling).
    only_stats:
        If ``True``, only statistical metrics are computed.
    stats_singleordistance:
        Selects between single cloud or distance statistics.
    project:
        Project name for logging and output.
    normal_override:
        Optional override for the normal scale.
    proj_override:
        Optional override for the projection scale.
    use_existing_params:
        Reuse previously estimated parameters if available.
    outlier_multiplicator:
        Multiplication factor for the outlier threshold.
    outlier_detection_method:
        Method used to detect outliers (e.g., ``"rmse"``).
    process_python_CC:
        Backend implementation to use: ``"python"`` or ``"CC"``.
    """

    folder_id: str
    filename_mov: str
    filename_ref: str
    mov_as_corepoints: bool
    use_subsampled_corepoints: int
    only_stats: bool
    stats_singleordistance: str
    project: str
    normal_override: Optional[float] = None
    proj_override: Optional[float] = None
    use_existing_params: bool = False
    outlier_multiplicator: float = 3.0
    outlier_detection_method: str = "rmse"
    process_python_CC: str = "python"


