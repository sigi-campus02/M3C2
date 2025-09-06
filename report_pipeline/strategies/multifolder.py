from __future__ import annotations

"""Strategy assembling jobs from multiple folders and filenames."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..domain import DistanceFile, PlotJob, parse_label_group
from .base import JobBuilder


@dataclass
class MultiFolderJobBuilder(JobBuilder):
    """Build jobs from ``filenames`` located in each of ``folders``."""

    data_dir: Path
    folders: Iterable[str]
    filenames: Iterable[str]
    paired: bool = False

    def build_jobs(self) -> list[PlotJob]:
        base = Path(self.data_dir).expanduser().resolve()
        jobs: list[PlotJob] = []
        for folder in self.folders:
            folder_path = base / folder
            for name in self.filenames:
                path = folder_path / name
                if not path.exists():
                    raise FileNotFoundError(f"Distance file not found: {path}")
                label, group = parse_label_group(path)
                item = DistanceFile(path=path, label=label, group=group)
                jobs.append(PlotJob(items=[item]))

        if self.paired and len(jobs) != 2:
            raise ValueError("--paired requires exactly two files")
        return jobs
