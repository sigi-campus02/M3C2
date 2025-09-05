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
import re
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

    widgets: dict[str, Tuple[tk.Variable, tk.Widget, tk.Widget, tk.Widget | None]] = {}
    row = 0

    descriptions = _load_arg_descriptions("config.schema.json")

    mode_action = next(
        (a for a in parser._actions if getattr(a, "dest", "") == "stats_singleordistance"),
        None,
    )
    plot_action = next(
        (a for a in parser._actions if getattr(a, "dest", "") == "plot_strategy"),
        None,
    )
    mode_var: tk.StringVar | None = None
    plot_var: tk.StringVar | None = None
    plot_widgets: list[tk.Widget] = []

    if mode_action is not None:
        tk.Label(root, text="Modus").grid(row=row, column=0, sticky="w", padx=5, pady=5)
        mode_var = tk.StringVar(value=str(mode_action.default or "single"))
        tk.Radiobutton(
            root,
            text="Distanz Calculation (M3C2)",
            variable=mode_var,
            value="distance",
            command=lambda: _update_mode_fields(mode_var, widgets, plot_var, plot_widgets),
        ).grid(row=row, column=1, sticky="w", padx=5, pady=5)
        row += 1
        tk.Radiobutton(
            root,
            text="Single Cloud Statistiks",
            variable=mode_var,
            value="single",
            command=lambda: _update_mode_fields(mode_var, widgets, plot_var, plot_widgets),
        ).grid(row=row, column=1, sticky="e", padx=5, pady=5)
        row += 1
        if "plot" in getattr(mode_action, "choices", []):
            tk.Radiobutton(
                root,
                text="Plots from Distances",
                variable=mode_var,
                value="plot",
                command=lambda: _update_mode_fields(mode_var, widgets, plot_var, plot_widgets),
            ).grid(row=row, column=1, sticky="w", padx=5, pady=5)
            row += 1
            if plot_action is not None:

                plot_label = tk.Label(root, text="Plot Strategie")
                plot_label.grid(row=row, column=0, sticky="w", padx=5, pady=5)
                plot_widgets.append(plot_label)
                plot_var = tk.StringVar(value=str(plot_action.default or "specificfile"))
                rb = tk.Radiobutton(
                    root,
                    text="Dateien in Ordner",
                    variable=plot_var,
                    value="onefolder",
                    command=lambda: _update_mode_fields(mode_var, widgets, plot_var, plot_widgets),
                )
                rb.grid(row=row, column=1, sticky="w", padx=5, pady=5)
                plot_widgets.append(rb)
                row += 1
                rb = tk.Radiobutton(
                    root,
                    text="Mehrere Ordner",
                    variable=plot_var,
                    value="severalfolders",
                    command=lambda: _update_mode_fields(mode_var, widgets, plot_var, plot_widgets),
                )
                rb.grid(row=row, column=1, sticky="w", padx=5, pady=5)
                plot_widgets.append(rb)
                row += 1
                rb = tk.Radiobutton(
                    root,
                    text="Spezifische Dateien",
                    variable=plot_var,
                    value="specificfile",
                    command=lambda: _update_mode_fields(mode_var, widgets, plot_var, plot_widgets),
                )
                rb.grid(row=row, column=1, sticky="w", padx=5, pady=5)
                plot_widgets.append(rb)
                row += 1

    bool_action = getattr(argparse, "BooleanOptionalAction", None)

    for action in parser._actions:
        # Skip help actions – they are not user facing parameters
        if isinstance(action, argparse._HelpAction):
            continue
        if mode_action is not None and action is mode_action:
            continue
        if plot_action is not None and action is plot_action:
            continue

        label = tk.Label(root, text=action.dest)
        label.grid(row=row, column=0, sticky="w", padx=5, pady=5)
        desc = descriptions.get(action.dest, "")
        desc_widget: tk.Widget | None = None
        if desc:
            desc_widget = tk.Label(
                root, text=desc, fg="gray", wraplength=350, justify="left"
            )
            desc_widget.grid(row=row, column=2, sticky="w", padx=5, pady=5)

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
                if isinstance(action.default, (list, tuple)):
                    var.set(", ".join(map(str, action.default)))
                else:
                    var.set(str(action.default))
            widget = tk.Entry(root, textvariable=var)
            widget.grid(row=row, column=1, sticky="ew", padx=5, pady=5)
        widgets[action.dest] = (var, widget, label, desc_widget)
        row += 1

    if mode_var is not None:
        _update_mode_fields(mode_var, widgets, plot_var, plot_widgets)

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
            ) or (
                plot_action is not None and action is plot_action
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
                            items = [
                                v.strip()
                                for v in re.split(r"[,\s]+", value)
                                if v.strip()
                            ]
                            argv.extend(items)
                    else:
                        if value:
                            argv.extend([action.option_strings[0], value])
            else:  # positional
                value = var.get().strip()
                if action.nargs not in (None, 1):
                    parts = [v.strip() for v in re.split(r"[,\s]+", value) if v.strip()]
                    argv.extend(parts)
                elif value:
                    argv.append(value)

        if mode_action is not None and mode_var is not None:
            argv.extend([mode_action.option_strings[0], mode_var.get()])
            if (
                mode_var.get() == "plot"
                and plot_action is not None
                and plot_var is not None
            ):
                argv.extend([plot_action.option_strings[0], plot_var.get()])

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


def _update_mode_fields(
    mode_var: tk.StringVar,
    widgets: dict[str, Tuple[tk.Variable, tk.Widget, tk.Widget, tk.Widget | None]],
    plot_var: tk.StringVar | None = None,
    plot_widgets: list[tk.Widget] | None = None,
) -> None:
    """Show only widgets relevant for the selected mode and strategy."""

    mode = mode_var.get()
    dist_fields = [
        "filename_reference",
        "filename_comparison",
        "outlier_detection_method",
        "outlier_multiplicator",
        "only_stats",
    ]
    single_fields = ["filename_singlecloud"]
    plot_specific = ["overlay_files", "overlay_outdir", "plot_types"]
    plot_onefolder = ["folder", "overlay_outdir", "options"]
    plot_several = ["data_dir", "folders", "filenames", "overlay_outdir", "options"]

    common_fields = [
        "data_dir",
        "folders",
        "use_subsampled_corepoints",
        "sample_size",
        "scale_strategy",
        "output_format",
        "project",
        "normal_override",
        "proj_override",
        "use_existing_params",
    ]


    def _hide(
        var: tk.Variable, widget: tk.Widget, label: tk.Widget, desc: tk.Widget | None
    ) -> None:
        widget.grid_remove()
        label.grid_remove()
        if desc is not None:
            desc.grid_remove()
        if isinstance(var, tk.BooleanVar):
            var.set(False)
        else:
            var.set("")

    def _show(widget: tk.Widget, label: tk.Widget, desc: tk.Widget | None) -> None:
        label.grid()
        widget.grid()
        if desc is not None:
            desc.grid()
        try:
            widget.configure(state="normal")
        except tk.TclError:
            pass

    # Distanz-Mode-Felder
    for name in dist_fields:
        if name in widgets:
            var, widget, label, desc = widgets[name]
            if mode == "distance":
                _show(widget, label, desc)
            else:
                _hide(var, widget, label, desc)

    # Single-Mode-Felder
    for name in single_fields:
        if name in widgets:
            var, widget, label, desc = widgets[name]
            if mode == "single":
                _show(widget, label, desc)
            else:
                _hide(var, widget, label, desc)

    # Gemeinsame Felder für Single- und Distanz-Mode
    for name in common_fields:
        if name in widgets:
            var, widget, label, desc = widgets[name]
            if mode == "plot":
                _hide(var, widget, label, desc)
            else:
                _show(widget, label, desc)


    # Plot-Mode-Felder
    all_plot_fields = set(plot_specific + plot_onefolder + plot_several)
    for name in all_plot_fields:
        if name in widgets:

            var, widget, label, desc = widgets[name]

            if mode == "plot":
                strategy = plot_var.get() if plot_var is not None else "specificfile"
                if (
                    (strategy == "specificfile" and name in plot_specific)
                    or (strategy == "onefolder" and name in plot_onefolder)
                    or (strategy == "severalfolders" and name in plot_several)
                ):
                    _show(widget, label, desc)
                else:
                    _hide(var, widget, label, desc)
            elif name not in common_fields:
                _hide(var, widget, label, desc)

    if plot_widgets is not None:
        for w in plot_widgets:
            if mode == "plot":
                w.grid()
                try:
                    w.configure(state="normal")
                except tk.TclError:
                    pass
            else:
                w.grid_remove()

    # only_stats im Single-Mode fix auf True setzen
    if "only_stats" in widgets and mode == "single":
        only_var = widgets["only_stats"][0]
        if isinstance(only_var, tk.BooleanVar):
            only_var.set(True)
        else:
            try:
                only_var.set("1")
            except Exception:
                pass

