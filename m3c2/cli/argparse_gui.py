"""GUI launcher for argparse-based scripts.

This utility builds a simple Tkinter interface for all arguments defined in an
``argparse.ArgumentParser``. Users can edit values, start execution and cancel
without running anything.

Example usage integrates the :class:`~m3c2.cli.cli.CLIApp` of the project.
"""
import argparse
import tkinter as tk
from tkinter import messagebox
import logging
import os
import json
from typing import Tuple
from m3c2.config.logging_config import setup_logging
from m3c2.cli.cli import CLIApp


logger = logging.getLogger(__name__)


def run_gui(parser: argparse.ArgumentParser, main_func) -> None:
    """Create a Tkinter GUI for the given parser and execute ``main_func``.

    ``main_func`` receives the list of command-line arguments when the user
    presses the "Start" button. A "Cancel" button closes the window without
    running anything.
    """
    setup_logging()
    root = tk.Tk()
    logger.info("GUI window opened")
    root.title(parser.prog or "Argumente")

    widgets: dict[str, Tuple[tk.Variable, tk.Widget]] = {}
    row = 0

    descriptions = _load_arg_descriptions("config.schema.json")

    mode_action = next(
        (a for a in parser._actions if getattr(a, "dest", "") == "stats_singleordistance"),
        None,
    )
    mode_var: tk.StringVar | None = None

    if mode_action is not None:
        tk.Label(root, text="Modus").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        mode_var = tk.StringVar(value=str(mode_action.default or "single"))
        tk.Radiobutton(
            root,
            text="Distanz Calculation (M3C2)",
            variable=mode_var,
            value="distance",
            command=lambda: _update_mode_fields(mode_var, widgets),
        ).grid(row=row, column=1, sticky="w", padx=5, pady=5)
        row += 1
        tk.Radiobutton(
            root,
            text="Single Cloud Statistiks",
            variable=mode_var,
            value="single",
            command=lambda: _update_mode_fields(mode_var, widgets),
        ).grid(row=row, column=1, sticky="e", padx=5, pady=5)
        row += 1

    bool_action = getattr(argparse, "BooleanOptionalAction", None)

    for action in parser._actions:
        # Skip help actions – they are not user facing parameters
        if isinstance(action, argparse._HelpAction):
            continue
        if mode_action is not None and action is mode_action:
            continue

        tk.Label(root, text=action.dest).grid(row=row, column=0, sticky="w", padx=5, pady=5)
        desc = descriptions.get(action.dest, "")

        if desc:
            tk.Label(root, text=desc, fg="gray", wraplength=350, justify="left").grid(
                row=row, column=2, sticky="w", padx=5, pady=5
            )

        if action.option_strings and action.nargs == 0 and (
            action.const is True
            or (bool_action is not None and isinstance(action, bool_action))
        ):
            var = tk.BooleanVar(value=bool(action.default))
            widget = tk.Checkbutton(root, variable=var)
            widget.grid(row=row, column=1, sticky="w", padx=5, pady=5)
        elif action.choices:
            default = action.default if action.default is not None else next(iter(action.choices))
            var = tk.StringVar(value=str(default))
            widget = tk.OptionMenu(root, var, *action.choices)
            widget.grid(row=row, column=1, sticky="ew", padx=5, pady=5)
        else:
            var = tk.StringVar()
            if action.default not in (None, argparse.SUPPRESS):
                var.set(str(action.default))
            widget = tk.Entry(root, textvariable=var)
            widget.grid(row=row, column=1, sticky="ew", padx=5, pady=5)
        widgets[action.dest] = (var, widget)
        row += 1

    if mode_var is not None:
        _update_mode_fields(mode_var, widgets)

    def on_start() -> None:
        """Handle the start event by collecting arguments and executing the main function.

        The callback gathers values from all GUI widgets, assembles them into an
        ``argv`` list, validates them using the provided ``argparse`` parser and,
        if successful, closes the GUI and invokes ``main_func`` with the
        resulting argument list. Any parsing errors are reported via a message
        box instead of raising ``SystemExit``.
        """
        argv: list[str] = []
        for action in parser._actions:
            if isinstance(action, argparse._HelpAction) or (
                mode_action is not None and action is mode_action
            ):
                continue
            var = widgets[action.dest][0]
            if action.option_strings:
                if isinstance(var, tk.BooleanVar):
                    if bool_action is not None and isinstance(action, bool_action):
                        if var.get() != action.default:
                            opt = action.option_strings[0] if var.get() else action.option_strings[1]
                            argv.append(opt)
                    else:
                        if var.get() != action.default:
                            argv.append(action.option_strings[0])
                else:
                    value = var.get().strip()
                    if action.nargs not in (None, 1):
                        if value:
                            argv.append(action.option_strings[0])
                            argv.extend(value.split())
                    else:
                        if value:
                            argv.extend([action.option_strings[0], value])
            else:  # positional
                value = var.get().strip()
                if action.nargs not in (None, 1):
                    argv.extend(value.split())
                elif value:
                    argv.append(value)

        if mode_action is not None and mode_var is not None:
            argv.extend([mode_action.option_strings[0], mode_var.get()])

        logger.info("Start pressed with arguments: %s", argv)
        try:
            parser.parse_args(argv)
        except SystemExit:
            # argparse reports errors via SystemExit; show message instead
            messagebox.showerror("Ungültige Eingabe", "Bitte Eingaben prüfen.")
            return
        except (ValueError, OSError) as exc:
            logger.exception("Error parsing arguments")
            messagebox.showerror("Fehler", str(exc))
            return

        root.destroy()
        try:
            main_func(argv)
        except (RuntimeError, ValueError) as exc:
            logger.exception("Exception raised by main_func")
            messagebox.showerror("Fehler", str(exc))
        except Exception:
            logger.exception("Unexpected exception raised by main_func")
            raise

    def on_cancel() -> None:
        """Close the GUI without executing the selected command."""
        logger.info("Cancel pressed")
        root.destroy()

    tk.Button(root, text="Start", command=on_start).grid(row=row, column=0, padx=5, pady=10)
    tk.Button(root, text="Abbrechen", command=on_cancel).grid(row=row, column=1, padx=5, pady=10)

    root.mainloop()


def _load_arg_descriptions(schema_path):
    # Absoluten Pfad berechnen
    abs_path = os.path.join(os.path.dirname(__file__), "..", "..", "config.schema.json")
    abs_path = os.path.abspath(abs_path)
    with open(abs_path, "r", encoding="utf-8") as f:
        schema = json.load(f)
    arg_props = schema.get("properties", {}).get("arguments", {}).get("properties", {})
    return {key: value.get("description", "") for key, value in arg_props.items()}


def _update_mode_fields(mode_var: tk.StringVar, widgets: dict[str, Tuple[tk.Variable, tk.Widget]]) -> None:
    """Enable or disable widgets depending on the selected statistics mode."""

    mode = mode_var.get()
    dist_fields = [
        "filename_ref",
        "filename_mov",
        "mov_as_corepoints",
        "use_subsampled_corepoints",
        "outlier_detection_method",
        "outlier_multiplicator",
    ]
    single_fields = ["filename_singlecloud"]

    for name in dist_fields:
        if name in widgets:
            var, widget = widgets[name]
            if mode == "distance":
                widget.configure(state="normal")
            else:
                widget.configure(state="disabled")
                var.set("")

    for name in single_fields:
        if name in widgets:
            var, widget = widgets[name]
            if mode == "single":
                widget.configure(state="normal")
            else:
                widget.configure(state="disabled")
                var.set("")

    if "only_stats" in widgets:
        only_var, only_widget = widgets["only_stats"]
        if mode == "single":
            only_var.set(True)
            only_widget.configure(state="disabled")
        else:
            only_widget.configure(state="normal")

