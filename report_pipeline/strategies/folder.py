from __future__ import annotations

"""Strategy producing jobs from all distance files in a folder."""

from dataclasses import dataclass
from pathlib import Path

from ..domain import DistanceFile, PlotJob, parse_label_group
from .base import JobBuilder


@dataclass
class FolderJobBuilder(JobBuilder):
    """Build jobs for every distance file within a directory."""

    folder: Path
    pattern: str = "*"
    paired: bool = False

    def build_jobs(self) -> list[PlotJob]:
        folder = Path(self.folder).expanduser().resolve()
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder does not exist: {folder}")

        paths = sorted(p for p in folder.glob(self.pattern) if p.is_file())
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
