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
    assert len(jobs) == 2
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
    builder = MultiFolderJobBuilder(folders=folders, paired=True)
    jobs = builder.build_jobs()
    assert len(jobs) == 2
    titles = [job.page_title for job in jobs]

    assert titles == ["d1 (g)", "d2 (g)"]
    labels = [[item.label for item in job.items] for job in jobs]
    assert labels == [["f1", "f2"], ["f1", "f2"]]


def test_multifolder_job_builder_paired_validation(tmp_path):
    f1 = tmp_path / "f1"
    f2 = tmp_path / "f2"
    f1.mkdir()
    f2.mkdir()
    (f1 / "d1__g1.txt").write_text("1\n")
    # Missing matching file in f2
    builder = MultiFolderJobBuilder(folders=[f1, f2], paired=True)
    with pytest.raises(ValueError):
        builder.build_jobs()

