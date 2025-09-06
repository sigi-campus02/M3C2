from report_pipeline import cli
from report_pipeline import cli
from report_pipeline.strategies.folder import FolderJobBuilder
from report_pipeline.strategies.files import FilesJobBuilder
from report_pipeline.strategies.multifolder import MultiFolderJobBuilder


def test_parse_args_creates_correct_builders(tmp_path):
    ns = cli.parse_args(["folder", str(tmp_path)])
    assert isinstance(ns.builder_factory(ns), FolderJobBuilder)

    ns = cli.parse_args(["files", str(tmp_path / "a.txt"), str(tmp_path / "b.txt")])
    assert isinstance(ns.builder_factory(ns), FilesJobBuilder)

    ns = cli.parse_args(["multifolder", str(tmp_path), "--folders", "f1", "--filenames", "a.txt"])
    assert isinstance(ns.builder_factory(ns), MultiFolderJobBuilder)


def test_run_dry_run_returns_empty_list(tmp_path):
    result = cli.run(["folder", str(tmp_path), "--dry-run"])
    assert result == []
