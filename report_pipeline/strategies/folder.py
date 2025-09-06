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
    pattern: str = "*.txt"
    paired: bool = False
    group_by_folder: bool = False

    def build_jobs(self) -> list[PlotJob]:
        folder = Path(self.folder).expanduser().resolve()
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder does not exist: {folder}")

        paths = sorted(p for p in folder.glob(self.pattern) if p.is_file())
        if self.paired and len(paths) != 2:
            raise ValueError("--paired requires exactly two files")

        jobs: list[PlotJob] = []
        for path in paths:

            if self.group_by_folder:
                try:
                    relative_parent = path.parent.relative_to(folder)
                    group = relative_parent.parts[0] if relative_parent.parts else folder.name
                except ValueError:
                    group = path.parent.name
                label = path.stem
            else:
                label, group = parse_label_group(path)
            distances = load_distance_file(str(path))
            jobs.append(PlotJob(distances=distances, label=label, group=group))

        return jobs
