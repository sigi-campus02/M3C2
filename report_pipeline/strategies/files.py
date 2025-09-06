from __future__ import annotations

"""Strategy building jobs from explicit distance files."""

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from m3c2.cli.overlay_report import load_distance_file

from ..domain import PlotJob, parse_label_group
from .base import JobBuilder


@dataclass
class FilesJobBuilder(JobBuilder):
    """Create :class:`PlotJob` instances from a list of files."""

    files: Iterable[Path]
    paired: bool = False

    def build_jobs(self) -> list[PlotJob]:
        files = [Path(f).expanduser().resolve() for f in self.files]
        if len(files) < 2:
            raise ValueError("At least two files are required")
        if self.paired and len(files) != 2:
            raise ValueError("--paired requires exactly two files")

        jobs: list[PlotJob] = []
        for path in files:
            if not path.exists():
                raise FileNotFoundError(f"Distance file not found: {path}")
            label, group = parse_label_group(path)
            distances = load_distance_file(str(path))
            jobs.append(PlotJob(distances=distances, label=label, group=group))
        return jobs
