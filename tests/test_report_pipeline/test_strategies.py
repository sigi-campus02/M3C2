from pathlib import Path
import pytest

from report_pipeline.strategies.folder import FolderJobBuilder
from report_pipeline.strategies.files import FilesJobBuilder
from report_pipeline.strategies.multifolder import MultiFolderJobBuilder


def _labels(job):
    return [item.label for item in job.items]


def _groups(job):
    return [item.group for item in job.items]


def test_folder_job_builder_sorts(tmp_path: Path) -> None:
    (tmp_path / "b__g.txt").write_text("1\n")
    (tmp_path / "a__g.txt").write_text("2\n")
    builder = FolderJobBuilder(folder=tmp_path)
    jobs = builder.build_jobs()
    assert [_labels(j) for j in jobs] == [["a"], ["b"]]
    assert [_groups(j) for j in jobs] == [["g"], ["g"]]


def test_folder_job_builder_pattern_and_paired(tmp_path: Path) -> None:
    for name in ["a1.txt", "a2.txt", "b1.txt", "b2.txt"]:
        (tmp_path / name).write_text("1\n")
    builder = FolderJobBuilder(folder=tmp_path, pattern="a*.txt", paired=True)
    jobs = builder.build_jobs()
    assert len(jobs) == 1
    assert [p.path.name for p in jobs[0].items] == ["a1.txt", "a2.txt"]


def test_files_job_builder_missing_file(tmp_path: Path) -> None:
    existing = tmp_path / "a.txt"
    existing.write_text("1\n")
    missing = tmp_path / "missing.txt"
    builder = FilesJobBuilder(files=[existing, missing])
    with pytest.raises(FileNotFoundError):
        builder.build_jobs()


def test_files_job_builder_paired(tmp_path: Path) -> None:
    paths = []
    for name in ["a.txt", "b.txt", "c.txt", "d.txt"]:
        p = tmp_path / name
        p.write_text("1\n")
        paths.append(p)
    builder = FilesJobBuilder(files=paths, paired=True)
    jobs = builder.build_jobs()
    assert len(jobs) == 2
    assert all(len(job.items) == 2 for job in jobs)


def test_multifolder_job_builder(tmp_path: Path) -> None:
    f1 = tmp_path / "f1"
    f2 = tmp_path / "f2"
    f1.mkdir()
    f2.mkdir()
    for folder in [f1, f2]:
        (folder / "d1__g1.txt").write_text("1\n")
        (folder / "d2__g2.txt").write_text("2\n")
    builder = MultiFolderJobBuilder(folders=[f1, f2], pattern="*.txt")
    jobs = builder.build_jobs()
    assert len(jobs) == 2
    assert [_labels(j) for j in jobs] == [["d1", "d1"], ["d2", "d2"]]
    assert [_groups(j) for j in jobs] == [["g1", "g1"], ["g2", "g2"]]
