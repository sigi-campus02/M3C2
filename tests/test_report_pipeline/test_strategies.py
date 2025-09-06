import pytest

from report_pipeline.strategies.folder import FolderJobBuilder
from report_pipeline.strategies.files import FilesJobBuilder
from report_pipeline.strategies.multifolder import MultiFolderJobBuilder


def test_folder_job_builder_sorts(tmp_path):
    (tmp_path / "b__g2.txt").write_text("1\n")
    (tmp_path / "a__g1.txt").write_text("2\n")
    builder = FolderJobBuilder(folder=tmp_path)
    jobs = builder.build_jobs()
    assert len(jobs) == 1
    labels = [item.label for item in jobs[0].items]
    assert labels == ["a", "b"]
    groups = [item.group for item in jobs[0].items]
    assert groups == ["g", "g"]


def test_folder_job_builder_group_by_folder(tmp_path):
    (tmp_path / "alpha").mkdir()
    (tmp_path / "alpha" / "a.txt").write_text("1\n")
    (tmp_path / "alpha" / "sub").mkdir()
    (tmp_path / "alpha" / "sub" / "b.txt").write_text("1\n")
    (tmp_path / "beta").mkdir()
    (tmp_path / "beta" / "c.txt").write_text("1\n")
    builder = FolderJobBuilder(folder=tmp_path, pattern="**/*.txt", group_by_folder=True)
    jobs = sorted(builder.build_jobs(), key=lambda j: j.page_title)
    assert [job.page_title for job in jobs] == ["alpha", "beta"]
    labels = {job.page_title: [item.label for item in job.items] for job in jobs}
    assert labels["alpha"] == ["a", "b"]
    assert labels["beta"] == ["c"]


def test_folder_job_builder_missing_directory(tmp_path):
    missing = tmp_path / "missing"
    builder = FolderJobBuilder(folder=missing)
    with pytest.raises(FileNotFoundError):
        builder.build_jobs()


def test_folder_job_builder_paired_validation(tmp_path):
    for name in ["a__g.txt", "b__g.txt", "c__g.txt"]:
        (tmp_path / name).write_text("1\n")
    builder = FolderJobBuilder(folder=tmp_path, paired=True)
    with pytest.raises(ValueError):
        builder.build_jobs()



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


def test_files_job_builder_paired_validation(tmp_path):
    a = tmp_path / "a__g.txt"
    b = tmp_path / "b__g.txt"
    c = tmp_path / "c__g.txt"
    for f in (a, b, c):
        f.write_text("1\n")
    builder = FilesJobBuilder(files=[a, b, c], paired=True)
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
    assert titles == ["f1", "f2"]
    labels = [[item.label for item in job.items] for job in jobs]
    assert labels == [["d1", "d2"], ["d1", "d2"]]
    groups = [[item.group for item in job.items] for job in jobs]
    assert groups == [["g", "g"], ["g", "g"]]


def test_multifolder_job_builder_paired_validation(tmp_path):
    f1 = tmp_path / "f1"
    f2 = tmp_path / "f2"
    f1.mkdir()
    f2.mkdir()
    # f1 contains only one file while f2 contains two
    (f1 / "d1.txt").write_text("1\n")
    (f2 / "a.txt").write_text("1\n")
    (f2 / "b.txt").write_text("1\n")
    builder = MultiFolderJobBuilder(folders=[f1, f2], paired=True)
    with pytest.raises(ValueError):
        builder.build_jobs()

