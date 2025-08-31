"""GUI launcher for argparse-based scripts.

This utility builds a simple Tkinter interface for all arguments defined in an
``argparse.ArgumentParser``.  Users can edit values, start execution and cancel
without running anything.

Example usage reimplements the CLI from ``main_generatecloud.py``.
"""
import argparse
import os
import tkinter as tk
from tkinter import messagebox

from main_generatecloud import convert_one, convert_all, logger


def run_gui(parser: argparse.ArgumentParser, main_func) -> None:
    """Create a Tkinter GUI for the given parser and execute ``main_func``.

    ``main_func`` receives the parsed ``argparse.Namespace`` when the user
    presses the "Start" button.  A "Cancel" button closes the window without
    running anything.
    """
    root = tk.Tk()
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
        try:
            args = parser.parse_args(argv)
        except SystemExit:
            # argparse reports errors via SystemExit; show message instead
            messagebox.showerror("Ungültige Eingabe", "Bitte Eingaben prüfen.")
            return
        except Exception as exc:
            messagebox.showerror("Fehler", str(exc))
            return

        root.destroy()
        try:
            main_func(args)
        except Exception as exc:  # final safety net
            messagebox.showerror("Fehler", str(exc))

    def on_cancel() -> None:
        root.destroy()

    tk.Button(root, text="Start", command=on_start).grid(row=row, column=0, padx=5, pady=10)
    tk.Button(root, text="Abbrechen", command=on_cancel).grid(row=row, column=1, padx=5, pady=10)

    root.mainloop()


# ---------------------------------------------------------------------------
# Example integration with existing CLI (main_generatecloud.py)
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Erzeuge .ply aus *_m3c2_distances_coordinates*.txt")
    parser.add_argument("paths", nargs="+", help="Ordner oder Dateien (TXT). Bei Ordnern wird rekursiv gesucht.")
    parser.add_argument("--overwrite", action="store_true", help="Vorhandene .ply überschreiben")
    return parser


def main(args: argparse.Namespace) -> None:
    txts: list[str] = []
    dirs: list[str] = []
    for p in args.paths:
        if os.path.isdir(p):
            dirs.append(os.path.abspath(p))
        elif os.path.isfile(p):
            if p.endswith(".txt"):
                txts.append(os.path.abspath(p))
            else:
                logger.warning("Ignoriere Datei (keine .txt): %s", p)
        else:
            logger.warning("Pfad nicht gefunden: %s", p)

    for t in txts:
        try:
            convert_one(t, overwrite=args.overwrite)
        except Exception as exc:
            logger.warning("Fehler bei %s: %s", t, exc)

    if dirs:
        convert_all(dirs, overwrite=args.overwrite)


if __name__ == "__main__":
    run_gui(build_parser(), main)
