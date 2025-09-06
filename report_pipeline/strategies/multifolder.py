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

    folders: Iterable[Path]
    pattern: str
    paired: bool = False

    def build_jobs(self) -> list[PlotJob]:
        folder_paths = [Path(f).expanduser().resolve() for f in self.folders]
        for folder in folder_paths:
            if not folder.is_dir():
                raise FileNotFoundError(f"Folder does not exist: {folder}")

        files_by_name: dict[str, list[DistanceFile]] = {}
        for folder in folder_paths:
            for path in sorted(folder.glob(self.pattern)):
                if not path.is_file():
                    continue
                label, group = parse_label_group(path)
                item = DistanceFile(path=path, label=label, group=group)
                files_by_name.setdefault(path.name, []).append(item)

        jobs: list[PlotJob] = []
        expected = len(folder_paths)
        for name in sorted(files_by_name):
            items = files_by_name[name]
            if len(items) != expected:
                raise FileNotFoundError(
                    f"File '{name}' not found in all folders"
                )
            if self.paired and len(items) != 2:
                raise ValueError("--paired requires exactly two files per pattern")
            jobs.append(PlotJob(items=items))
        return jobs
