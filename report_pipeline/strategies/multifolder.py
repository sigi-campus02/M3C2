from __future__ import annotations

"""Strategy assembling jobs from multiple folders using a file pattern."""

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

    def build_jobs(self) -> list[PlotJob]:
        """Return a job for every folder containing matching files.

        Files are searched in each configured *folder* using ``pattern``.  All
        files from a folder are collected into a single :class:`PlotJob` whose
        page title is the folder name.  When ``paired`` is set the folder must
        contain exactly two matching files, otherwise a :class:`ValueError` is
        raised.
        """

        jobs: list[PlotJob] = []
        folder_paths = [Path(f).expanduser().resolve() for f in self.folders]

        for folder_path in folder_paths:
            if not folder_path.is_dir():
                raise FileNotFoundError(f"Folder does not exist: {folder_path}")

            paths = sorted(p for p in folder_path.glob(self.pattern) if p.is_file())
            items: list[DistanceFile] = []

            for path in paths:
                label, group = parse_label_group(path)
                items.append(DistanceFile(path=path, label=label, group=group))

            if self.paired and len(items) != 2:
                raise ValueError("--paired requires exactly two files per folder")

            if items:
                jobs.append(PlotJob(items=items, page_title=folder_path.name))

        return jobs
