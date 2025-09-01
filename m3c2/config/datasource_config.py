
from __future__ import annotations
from dataclasses import dataclass


@dataclass
class DataSourceConfig:
    """Configuration for the data source."""

    folder: str
    mov_basename: str = "mov"
    ref_basename: str = "ref"
    mov_as_corepoints: bool = True
    use_subsampled_corepoints: int = 1