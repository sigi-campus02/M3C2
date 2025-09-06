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

    def build_jobs(self) -> list[PlotJob]:
        paths = [Path(f).expanduser().resolve() for f in self.files]
        for path in paths:
            if not path.exists():
                raise FileNotFoundError(f"Distance file not found: {path}")

        if self.paired and len(paths) % 2:
            raise ValueError("--paired requires an even number of files")

        def to_item(p: Path) -> DistanceFile:
            label, group = parse_label_group(p)
            return DistanceFile(path=p, label=label, group=group)

        jobs: list[PlotJob] = []
        if self.paired:
            for idx in range(0, len(paths), 2):
                items = [to_item(p) for p in paths[idx : idx + 2]]
                jobs.append(PlotJob(items=items))
        else:
            for p in paths:
                jobs.append(PlotJob(items=[to_item(p)]))
        return jobs
