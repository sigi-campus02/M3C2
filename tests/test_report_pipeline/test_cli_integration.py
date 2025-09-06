from pathlib import Path

from report_pipeline import cli
from report_pipeline.strategies.folder import FolderJobBuilder
from report_pipeline.strategies.files import FilesJobBuilder
from report_pipeline.strategies.multifolder import MultiFolderJobBuilder


def test_parse_args_creates_correct_builders(tmp_path):
    ns = cli.parse_args(["folder", str(tmp_path)])
    assert isinstance(ns.builder_factory(ns), FolderJobBuilder)
    assert ns.max_per_page == 4

    ns = cli.parse_args(["files", str(tmp_path / "a.txt"), str(tmp_path / "b.txt")])
    assert isinstance(ns.builder_factory(ns), FilesJobBuilder)

    (tmp_path / "f1").mkdir()
    ns = cli.parse_args(["multifolder", "--folders", str(tmp_path / "f1")])
    assert isinstance(ns.builder_factory(ns), MultiFolderJobBuilder)


def test_run_dry_run_returns_none(tmp_path):
    result = cli.run(["folder", str(tmp_path), "--dry-run"])
    assert result is None


def test_run_returns_builder_and_out(tmp_path):
    f1 = tmp_path / "a.txt"
    f2 = tmp_path / "b.txt"
    f1.touch()
    f2.touch()
    result = cli.run(["files", str(f1), str(f2)])
    builder, out = result
    assert isinstance(builder, FilesJobBuilder)
    assert out == Path("report.pdf")
