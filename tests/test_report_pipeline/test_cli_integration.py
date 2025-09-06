from pathlib import Path
import argparse

import pytest

from report_pipeline import cli
from report_pipeline.strategies.folder import FolderJobBuilder
from report_pipeline.strategies.files import FilesJobBuilder
from report_pipeline.strategies.multifolder import MultiFolderJobBuilder


def test_parse_args_creates_correct_builders(tmp_path):
    ns = cli.parse_args(["folder", str(tmp_path)])
    assert isinstance(ns.builder_factory(ns), FolderJobBuilder)
    assert ns.max_per_page == 4
    assert ns.out == Path("overlay_report.pdf")

    ns = cli.parse_args(["files", str(tmp_path / "a.txt"), str(tmp_path / "b.txt")])
    assert isinstance(ns.builder_factory(ns), FilesJobBuilder)
    assert ns.max_per_page == 4
    assert ns.out == Path("overlay_report.pdf")

    (tmp_path / "f1").mkdir()
    ns = cli.parse_args(["multifolder", "--folders", str(tmp_path / "f1")])
    assert isinstance(ns.builder_factory(ns), MultiFolderJobBuilder)
    assert ns.max_per_page == 4
    assert ns.out == Path("overlay_report.pdf")


def test_run_dry_run_returns_none(tmp_path):
    result = cli.run(["folder", str(tmp_path), "--dry-run"])
    assert result is None


def test_files_builder_requires_two_paths(tmp_path):
    ns = cli.parse_args(["files", str(tmp_path / "a.txt")])
    with pytest.raises(argparse.ArgumentTypeError):
        ns.builder_factory(ns)


def test_files_parser_recognises_trailing_options(tmp_path):
    ns = cli.parse_args(
        [
            "files",
            str(tmp_path / "a.txt"),
            str(tmp_path / "b.txt"),
            "--dry-run",
        ]
    )
    assert ns.dry_run
    assert ns.files == [tmp_path / "a.txt", tmp_path / "b.txt"]
