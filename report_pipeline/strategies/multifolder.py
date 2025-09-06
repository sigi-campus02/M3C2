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
    plot_type: str = "histogram"

    def build_jobs(self) -> list[PlotJob]:
        folder_paths = [Path(f).expanduser().resolve() for f in self.folders]
        groups: dict[tuple[str, str | None], list[DistanceFile]] = {}

        for folder_path in folder_paths:
            if not folder_path.is_dir():
                raise FileNotFoundError(f"Folder does not exist: {folder_path}")
            paths = sorted(p for p in folder_path.glob(self.pattern) if p.is_file())
            for path in paths:
                base_label, base_group = parse_label_group(path)
                item = DistanceFile(path=path, label=folder_path.name, group=base_group)
                groups.setdefault((base_label, base_group), []).append(item)

        jobs: list[PlotJob] = []
        for (label, group), items in groups.items():
            title = label if group is None else f"{label} ({group})"
            jobs.append(PlotJob(items=items, page_title=title, plot_type=self.plot_type))

        if self.paired:
            expected = len(folder_paths)
            for job in jobs:
                if len(job.items) != expected:
                    raise ValueError("--paired requires exactly one file per folder")

        return jobs
