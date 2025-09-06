from __future__ import annotations

"""Strategy producing jobs from all distance files in a folder."""

from dataclasses import dataclass
from pathlib import Path

from ..domain import DistanceFile, PlotJob, normalize_group, parse_label_group
from .base import JobBuilder


@dataclass
class FolderJobBuilder(JobBuilder):
    """Build jobs for every distance file within a directory."""

    folder: Path
    pattern: str = "*.txt"
    paired: bool = False
    group_by_folder: bool = False
    plot_type: str = "histogram"

    def build_jobs(self) -> list[PlotJob]:
        folder = Path(self.folder).expanduser().resolve()
        if not folder.is_dir():
            raise FileNotFoundError(f"Folder does not exist: {folder}")

        paths = sorted(p for p in folder.glob(self.pattern) if p.is_file())

        if self.paired:
            groups: dict[str | None, list[DistanceFile]] = {}
            for path in paths:
                if self.group_by_folder:
                    try:
                        relative_parent = path.parent.relative_to(folder)
                        group = (
                            relative_parent.parts[0]
                            if relative_parent.parts
                            else folder.name
                        )
                    except ValueError:
                        group = path.parent.name
                    group = normalize_group(group)
                    label = path.stem
                else:
                    label, group = parse_label_group(path)
                item = DistanceFile(path=path, label=label, group=group)
                groups.setdefault(group, []).append(item)

            jobs = [PlotJob(items=items, page_title=grp, plot_type=self.plot_type) for grp, items in groups.items()]

            for job in jobs:
                if len(job.items) != 2:
                    raise ValueError("--paired requires exactly two files per overlay")
            return jobs

        groups: dict[str | None, list[DistanceFile]] = {}
        for path in paths:
            if self.group_by_folder:
                try:
                    relative_parent = path.parent.relative_to(folder)
                    group = (
                        relative_parent.parts[0]
                        if relative_parent.parts
                        else folder.name
                    )
                except ValueError:
                    group = path.parent.name
                group = normalize_group(group)
                label = path.stem
            else:
                stem = path.stem
                if "__" in stem:
                    label, group = stem.split("__", 1)
                elif "--" in stem:
                    label, group = stem.split("--", 1)
                else:
                    label, group = stem, None
            item = DistanceFile(path=path, label=label, group=group)
            groups.setdefault(group, []).append(item)

        for items in groups.values():
            items.sort(key=lambda df: df.label)
        return [PlotJob(items=items, page_title=grp, plot_type=self.plot_type) for grp, items in groups.items()]
