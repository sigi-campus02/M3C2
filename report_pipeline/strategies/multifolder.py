from __future__ import annotations

"""Strategy assembling jobs from multiple folders using a file pattern."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from m3c2.cli.overlay_report import load_distance_file

from ..domain import PlotJob, parse_label_group
from .base import JobBuilder


@dataclass
class MultiFolderJobBuilder(JobBuilder):
    """Build jobs from files matching ``pattern`` in multiple folders."""

    folders: Iterable[Path]
    pattern: str = "*.txt"
    paired: bool = False

    def build_jobs(self) -> list[PlotJob]:
        jobs: list[PlotJob] = []
        for folder in self.folders:
            folder_path = Path(folder).expanduser().resolve()
            if not folder_path.is_dir():
                raise FileNotFoundError(f"Folder does not exist: {folder_path}")
            paths = sorted(p for p in folder_path.glob(self.pattern) if p.is_file())
            for path in paths:
                label, group = parse_label_group(path)
                if group is None:
                    group = folder_path.name
                distances = load_distance_file(str(path))
                jobs.append(PlotJob(distances=distances, label=label, group=group))

        if self.paired and len(jobs) != 2:
            raise ValueError("--paired requires exactly two files")
        return jobs
