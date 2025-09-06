from __future__ import annotations

"""Strategy building jobs from explicit distance files."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from ..domain import DistanceFile, PlotJob, parse_label_group
from .base import JobBuilder


@dataclass
class FilesJobBuilder(JobBuilder):
    """Create :class:`PlotJob` instances from a list of files."""

    files: Iterable[Path]
    paired: bool = False
    plot_type: str = "histogram"

    def build_jobs(self) -> list[PlotJob]:
        files = [Path(f).expanduser().resolve() for f in self.files]
        if len(files) < 2:
            raise ValueError("At least two files are required")

        groups: dict[str | None, list[DistanceFile]] = {}
        for path in sorted(files):
            if not path.exists():
                raise FileNotFoundError(f"Distance file not found: {path}")
            label, group = parse_label_group(path)
            item = DistanceFile(path=path, label=label, group=group)
            groups.setdefault(group, []).append(item)

        jobs = [PlotJob(items=items, page_title=grp, plot_type=self.plot_type) for grp, items in groups.items()]

        if self.paired:
            for job in jobs:
                if len(job.items) != 2:
                    raise ValueError("--paired requires exactly two files per overlay")
        return jobs
