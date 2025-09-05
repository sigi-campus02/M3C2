"""Tests for CLI application parser defaults and orchestration."""

from __future__ import annotations

from unittest.mock import MagicMock

from m3c2.cli.cli import CLIApp


def test_build_parser_defaults() -> None:
    """Ensure parser is built with expected default values.

    Returns
    -------
    None
    """
    parser = CLIApp().build_parser()
    args = parser.parse_args([])
    assert args.data_dir == "data"
    assert args.scale_strategy == "radius"
    assert args.project == "PROJECT"
    assert args.mov_as_corepoints is True
    assert args.only_stats is True


def test_run_invokes_orchestrator(monkeypatch, tmp_path) -> None:
    """Validate that ``CLIApp.run`` invokes ``BatchOrchestrator``.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to patch the orchestrator class.
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.

    Returns
    -------
    None
    """
    # create folder structure expected by CLIApp.run
    folder = tmp_path / "001"
    folder.mkdir()

    orchestrator_cls = MagicMock()
    monkeypatch.setattr("m3c2.cli.cli.BatchOrchestrator", orchestrator_cls)

    app = CLIApp()
    result = app.run(
        [
            "--data_dir",
            str(tmp_path),
            "--folders",
            "001",
            "--stats_singleordistance",
            "distance",
            "--filename_ref",
            "ref",
            "--filename_mov",
            "mov",
        ]
    )

    assert result == 0
    orchestrator_cls.assert_called_once()

    configs_arg, kwargs = orchestrator_cls.call_args
    configs = configs_arg[0]
    assert kwargs["strategy"] == "radius"
    assert len(configs) == 1
    cfg = configs[0]
    assert cfg.data_dir == str(tmp_path)
    assert cfg.folder_id == "001"

    orchestrator_instance = orchestrator_cls.return_value
    orchestrator_instance.run_all.assert_called_once_with()
