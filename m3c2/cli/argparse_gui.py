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

    widgets: dict[str, tk.Variable] = {}
    row = 0
    for action in parser._actions:
        # Skip help actions – they are not user facing parameters
        if isinstance(action, argparse._HelpAction):
            continue

        tk.Label(root, text=action.dest).grid(row=row, column=0, sticky="w", padx=5, pady=5)

        if action.option_strings and action.nargs == 0 and action.const is True:
            var = tk.BooleanVar(value=bool(action.default))
            tk.Checkbutton(root, variable=var).grid(row=row, column=1, sticky="w", padx=5, pady=5)
        elif action.choices:
            default = action.default if action.default is not None else next(iter(action.choices))
            var = tk.StringVar(value=str(default))
            tk.OptionMenu(root, var, *action.choices).grid(row=row, column=1, sticky="ew", padx=5, pady=5)
        else:
            var = tk.StringVar()
            if action.default not in (None, argparse.SUPPRESS):
                var.set(str(action.default))
            tk.Entry(root, textvariable=var).grid(row=row, column=1, sticky="ew", padx=5, pady=5)
        widgets[action.dest] = var
        row += 1

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
            if isinstance(action, argparse._HelpAction):
                continue
            var = widgets[action.dest]
            if action.option_strings:
                if isinstance(var, tk.BooleanVar):
                    if var.get():
                        argv.append(action.option_strings[0])
                else:
                    value = var.get().strip()
                    if action.nargs not in (None, 1):
                        if value:
                            argv.append(action.option_strings[0])
                            argv.extend(value.split())
                    else:
                        argv.extend([action.option_strings[0], value])
            else:  # positional
                value = var.get().strip()
                if action.nargs not in (None, 1):
                    argv.extend(value.split())
                elif value:
                    argv.append(value)
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
