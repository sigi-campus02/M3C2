"""Tests for the :mod:`m3c2.gui.argparse_gui` module."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from unittest import mock
import types

sys.path.append(str(Path(__file__).resolve().parents[2]))

from m3c2.cli.argparse_gui import run_gui


def test_run_gui_invokes_parse_args_and_main_func() -> None:
    """Test the integration between GUI argument collection and execution.

    Purpose
    -------
    Ensure that values entered in the GUI are converted into command-line
    arguments and passed to both :func:`ArgumentParser.parse_args` and the
    supplied ``main`` function.

    Expected Behavior
    -----------------
    After simulating user input and triggering the "Start" button, both mocks
    should be invoked once with ``["--flag", "--opt", "value", "positional"]``.
    """
    parser = argparse.ArgumentParser(prog="prog")
    parser.add_argument("--flag", action="store_true")
    parser.add_argument("--opt")
    parser.add_argument("pos")

    button_cmds: dict[str, mock.Mock] = {}
    vars_created: list[object] = []

    class FakeStringVar:
        def __init__(self, value: object | None = None) -> None:
            self.value = value
            vars_created.append(self)

        def get(self) -> object:
            return self.value

        def set(self, value: object) -> None:
            self.value = value

    class FakeBooleanVar(FakeStringVar):
        pass
    def fake_widget(*args, **kwargs):
        widget = mock.MagicMock()
        widget.grid = mock.MagicMock()
        return widget

    def fake_button(root, text: str, command):
        btn = mock.MagicMock()
        btn.grid = mock.MagicMock()
        button_cmds[text] = command
        return btn

    fake_tk = types.SimpleNamespace(
        Tk=lambda: mock.MagicMock(mainloop=lambda: None, destroy=lambda: None, title=lambda *args, **kwargs: None),
        Label=fake_widget,
        Entry=fake_widget,
        Checkbutton=fake_widget,
        OptionMenu=fake_widget,
        Button=fake_button,
        StringVar=FakeStringVar,
        BooleanVar=FakeBooleanVar,
    )

    parse_mock = mock.MagicMock()
    parser.parse_args = parse_mock
    main_mock = mock.MagicMock()

    with mock.patch("m3c2.cli.argparse_gui.tk", fake_tk), mock.patch(
        "m3c2.cli.argparse_gui.messagebox"
    ):
        run_gui(parser, main_mock)

        flag_var, opt_var, pos_var = vars_created
        flag_var.set(True)
        opt_var.set("value")
        pos_var.set("positional")

        button_cmds["Start"]()

    parse_mock.assert_called_once_with(["--flag", "--opt", "value", "positional"])
    main_mock.assert_called_once_with(["--flag", "--opt", "value", "positional"])


def test_run_gui_parses_comma_separated_lists() -> None:
    """Lists entered as comma-separated strings are split into multiple args."""

    parser = argparse.ArgumentParser(prog="prog")
    parser.add_argument("--folders", nargs="+")
    parser.set_defaults(folders=["one", "two"])

    button_cmds: dict[str, mock.Mock] = {}
    vars_created: list[object] = []

    class FakeStringVar:
        def __init__(self, value: object | None = None) -> None:
            self.value = value
            vars_created.append(self)

        def get(self) -> object:
            return self.value

        def set(self, value: object) -> None:
            self.value = value

    class FakeBooleanVar(FakeStringVar):
        pass

    def fake_widget(*args, **kwargs):
        widget = mock.MagicMock()
        widget.grid = mock.MagicMock()
        return widget

    def fake_button(root, text: str, command):
        btn = mock.MagicMock()
        btn.grid = mock.MagicMock()
        button_cmds[text] = command
        return btn

    fake_tk = types.SimpleNamespace(
        Tk=lambda: mock.MagicMock(mainloop=lambda: None, destroy=lambda: None, title=lambda *a, **k: None),
        Label=fake_widget,
        Entry=fake_widget,
        Button=fake_button,
        StringVar=FakeStringVar,
        BooleanVar=FakeBooleanVar,
        Variable=FakeStringVar,
        Widget=object,
    )

    parse_mock = mock.MagicMock()
    parser.parse_args = parse_mock
    main_mock = mock.MagicMock()

    with mock.patch("m3c2.cli.argparse_gui.tk", fake_tk), mock.patch(
        "m3c2.cli.argparse_gui.messagebox"
    ):
        run_gui(parser, main_mock)
        (folders_var,) = vars_created
        assert folders_var.get() == "one, two"
        folders_var.set("A, B, C")
        button_cmds["Start"]()

    expected = ["--folders", "A", "B", "C"]
    parse_mock.assert_called_once_with(expected)
    main_mock.assert_called_once_with(expected)


def test_run_gui_mode_selection_sets_stats_arg() -> None:
    """Ensure that selecting a mode passes the stats argument to the parser."""

    parser = argparse.ArgumentParser(prog="prog")
    parser.add_argument("--stats_singleordistance", choices=["single", "distance"])
    parser.add_argument("--only_stats", action="store_true")
    parser.add_argument("--filename_ref")
    parser.add_argument("--filename_singlecloud")

    button_cmds: dict[str, mock.Mock] = {}
    vars_created: list[object] = []

    class FakeStringVar:
        def __init__(self, value: object | None = None) -> None:
            self.value = value
            vars_created.append(self)

        def get(self) -> object:
            return self.value

        def set(self, value: object) -> None:
            self.value = value

    class FakeBooleanVar(FakeStringVar):
        pass

    def fake_widget(*args, **kwargs):
        widget = mock.MagicMock()
        widget.grid = mock.MagicMock()
        widget.configure = mock.MagicMock()
        return widget

    def fake_button(root, text: str, command):
        btn = mock.MagicMock()
        btn.grid = mock.MagicMock()
        button_cmds[text] = command
        return btn

    fake_tk = types.SimpleNamespace(
        Tk=lambda: mock.MagicMock(mainloop=lambda: None, destroy=lambda: None, title=lambda *args, **kwargs: None),
        Label=fake_widget,
        Entry=fake_widget,
        Checkbutton=fake_widget,
        OptionMenu=fake_widget,
        Radiobutton=fake_widget,
        Button=fake_button,
        StringVar=FakeStringVar,
        BooleanVar=FakeBooleanVar,
        Variable=FakeStringVar,
        Widget=object,
    )

    parse_mock = mock.MagicMock()
    parser.parse_args = parse_mock
    main_mock = mock.MagicMock()

    with mock.patch("m3c2.cli.argparse_gui.tk", fake_tk), mock.patch(
        "m3c2.cli.argparse_gui.messagebox"
    ):
        run_gui(parser, main_mock)

        mode_var, only_stats_var, filename_ref_var, filename_singlecloud_var = vars_created
        mode_var.set("single")
        filename_singlecloud_var.set("cloud.las")

        button_cmds["Start"]()

    expected = [
        "--only_stats",
        "--filename_singlecloud",
        "cloud.las",
        "--stats_singleordistance",
        "single",
    ]
    parse_mock.assert_called_once_with(expected)
    main_mock.assert_called_once_with(expected)


def test_run_gui_boolean_optional_action() -> None:
    """BooleanOptionalAction arguments can be toggled via checkboxes."""

    parser = argparse.ArgumentParser(prog="prog")
    parser.add_argument("--use_existing", action=argparse.BooleanOptionalAction)
    parser.set_defaults(use_existing=False)

    button_cmds: dict[str, mock.Mock] = {}
    vars_created: list[object] = []

    class FakeBooleanVar:
        def __init__(self, value: object | None = None) -> None:
            self.value = value
            vars_created.append(self)

        def get(self) -> object:
            return self.value

        def set(self, value: object) -> None:
            self.value = value

    def fake_widget(*args, **kwargs):
        widget = mock.MagicMock()
        widget.grid = mock.MagicMock()
        return widget

    def fake_button(root, text: str, command):
        btn = mock.MagicMock()
        btn.grid = mock.MagicMock()
        button_cmds[text] = command
        return btn

    fake_tk = types.SimpleNamespace(
        Tk=lambda: mock.MagicMock(mainloop=lambda: None, destroy=lambda: None, title=lambda *args, **kwargs: None),
        Label=fake_widget,
        Checkbutton=fake_widget,
        Button=fake_button,
        BooleanVar=FakeBooleanVar,
    )

    parse_mock = mock.MagicMock()
    parser.parse_args = parse_mock
    main_mock = mock.MagicMock()

    with mock.patch("m3c2.cli.argparse_gui.tk", fake_tk), mock.patch(
        "m3c2.cli.argparse_gui.messagebox"
    ):
        run_gui(parser, main_mock)

        (use_var,) = vars_created
        use_var.set(True)

        button_cmds["Start"]()

    parse_mock.assert_called_once_with(["--use_existing"])
    main_mock.assert_called_once_with(["--use_existing"])
