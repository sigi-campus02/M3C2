from __future__ import annotations

from unittest.mock import MagicMock
import sys
from pathlib import Path
from unittest.mock import MagicMock

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from m3c2.cli.cli import CLIApp


def test_build_parser_defaults() -> None:
    parser = CLIApp().build_parser()
    args = parser.parse_args([])
    assert args.data_dir == "data"
    assert args.scale_strategy == "radius"


def test_run_invokes_orchestrator(monkeypatch, tmp_path) -> None:
    # create folder structure expected by CLIApp.run
    folder = tmp_path / "001"
    folder.mkdir()

    orchestrator_cls = MagicMock()
    monkeypatch.setattr("m3c2.cli.cli.BatchOrchestrator", orchestrator_cls)

    app = CLIApp()
    result = app.run(["--data_dir", str(tmp_path), "--folders", "001"])

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


def test_log_level_from_env(monkeypatch, tmp_path) -> None:
    folder = tmp_path / "001"
    folder.mkdir()

    orchestrator_cls = MagicMock()
    monkeypatch.setattr("m3c2.cli.cli.BatchOrchestrator", orchestrator_cls)

    captured = {}

    def fake_setup_logging(level=None, **kwargs):
        captured["level"] = level

    monkeypatch.setattr("m3c2.cli.cli.setup_logging", fake_setup_logging)
    monkeypatch.setenv("LOG_LEVEL", "DEBUG")

    app = CLIApp()
    result = app.run(["--data_dir", str(tmp_path), "--folders", "001"])

    assert result == 0
    assert captured["level"] == "DEBUG"
    configs_arg, _ = orchestrator_cls.call_args
    cfg = configs_arg[0][0]
    assert cfg.log_level == "DEBUG"
