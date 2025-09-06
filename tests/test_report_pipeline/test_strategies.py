from pathlib import Path
import pytest

from report_pipeline.strategies.folder import FolderJobBuilder
from report_pipeline.strategies.files import FilesJobBuilder
from report_pipeline.strategies.multifolder import MultiFolderJobBuilder


def test_folder_job_builder_sorts(tmp_path):
    (tmp_path / "b__g2.txt").write_text("1\n")
    (tmp_path / "a__g1.txt").write_text("2\n")
    builder = FolderJobBuilder(folder=tmp_path)
    jobs = builder.build_jobs()
    labels = [job.items[0].label for job in jobs]
    assert labels == ["a", "b"]
    groups = [job.items[0].group for job in jobs]
    assert groups == ["g1", "g2"]


def test_files_job_builder_missing_file(tmp_path):
    existing = tmp_path / "a.txt"
    existing.write_text("1\n")
    missing = tmp_path / "missing.txt"
    builder = FilesJobBuilder(files=[existing, missing])
    with pytest.raises(FileNotFoundError):
        builder.build_jobs()


def test_files_job_builder_requires_two_files(tmp_path):
    a = tmp_path / "a.txt"
    a.write_text("1\n")
    builder = FilesJobBuilder(files=[a])
    with pytest.raises(ValueError):
        builder.build_jobs()


def test_multifolder_job_builder(tmp_path):
    base = tmp_path
    folders = []
    for folder in ["f1", "f2"]:
        sub = base / folder
        sub.mkdir()
        (sub / "d1__g1.txt").write_text("1\n")
        (sub / "d2__g2.txt").write_text("2\n")
        folders.append(sub)
    builder = MultiFolderJobBuilder(folders=folders)
    jobs = builder.build_jobs()
    assert len(jobs) == 2
    titles = [job.page_title for job in jobs]
    assert titles == ["f1", "f2"]
    labels = [[item.label for item in job.items] for job in jobs]
    assert labels == [["d1", "d2"], ["d1", "d2"]]
    groups = [[item.group for item in job.items] for job in jobs]
    assert groups == [["g1", "g2"], ["g1", "g2"]]


def test_multifolder_job_builder_paired_requires_two_files(tmp_path):
    folder = tmp_path / "f1"
    folder.mkdir()
    (folder / "only.txt").write_text("1\n")
    builder = MultiFolderJobBuilder(folders=[folder], paired=True)
    with pytest.raises(ValueError):
        builder.build_jobs()
