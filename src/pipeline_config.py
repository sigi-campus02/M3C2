# pipeline_config.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional

@dataclass(frozen=True)
class PipelineConfig:
    folder_id: str
    filename_mov: str
    filename_ref: str
    mov_as_corepoints: bool
    use_subsampled_corepoints: int
    process_python_CC: str
    only_stats: bool
    stats_singleordistance: str
    normal_override: Optional[float] = None
    proj_override: Optional[float] = None

