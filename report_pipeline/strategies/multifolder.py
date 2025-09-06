"""Strategy assembling jobs from multiple folders using a file pattern."""

from __future__ import annotations


from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..domain import DistanceFile, PlotJob, parse_label_group
from .base import JobBuilder


@dataclass
class MultiFolderJobBuilder(JobBuilder):
    """Build jobs from files matching ``pattern`` in multiple folders."""

    folders: Iterable[Path]
    pattern: str = "*.txt"
    paired: bool = False
    plot_type: str = "histogram"

    def build_jobs(self) -> list[PlotJob]:
        folder_paths = [Path(f).expanduser().resolve() for f in self.folders]
        jobs: list[PlotJob] = []

        for folder_path in folder_paths:
            if not folder_path.is_dir():
                raise FileNotFoundError(f"Folder does not exist: {folder_path}")

            paths = sorted(p for p in folder_path.glob(self.pattern) if p.is_file())

            if self.paired and len(paths) != 2:
                raise ValueError("--paired requires exactly two files per folder")

            items = []
            for path in paths:
                label, group = parse_label_group(path)
                items.append(DistanceFile(path=path, label=label, group=group))

            jobs.append(
                PlotJob(items=items, page_title=folder_path.name, plot_type=self.plot_type)
            )

        return jobs