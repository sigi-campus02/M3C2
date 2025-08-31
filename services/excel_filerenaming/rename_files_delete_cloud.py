#!/usr/bin/env python3
# remove_cloud_token.py
"""Remove the ``python_`` token from file and directory names.

This script provides a command-line utility to strip a specific token from
filenames. It can optionally traverse directories recursively and rename
folders as well as files.
"""

import argparse, os
from pathlib import Path

def transform(name: str) -> str:
    """Remove all occurrences of ``python_`` from a filename or directory name."""
    return name.replace("python_", "")

def iter_paths(base: Path, recursive: bool, include_dirs: bool):
    """Yield paths to files and optionally directories under ``base``.

    Traverses bottom-up when ``recursive`` is ``True`` so directory renames
    remain valid while iterating.
    """
    if recursive:
        for root, dirs, files in os.walk(base, topdown=False):
            for f in files:
                yield Path(root) / f
            if include_dirs:
                for d in dirs:
                    yield Path(root) / d
    else:
        for p in base.iterdir():
            if p.is_file() or (include_dirs and p.is_dir()):
                yield p

def main():
    """Command-line interface for removing ``python_`` tokens from names."""
    ap = argparse.ArgumentParser(
        description='Entfernt alle Vorkommen von "_cloud" aus Datei-/Ordnernamen.'
    )
    ap.add_argument("path", nargs="?", default=".", help="Basisordner (Standard: .)")
    ap.add_argument("-r", "--recursive", action="store_true", help="Auch Unterordner bearbeiten")
    ap.add_argument("--include-dirs", action="store_true", help="Auch Ordnernamen umbenennen")
    ap.add_argument("-n", "--dry-run", action="store_true", help="Nur anzeigen, nichts ändern")
    args = ap.parse_args()

    base = Path(args.path).resolve()
    changed = skipped = 0

    # Iterate over collected paths and rename them as needed.
    for p in iter_paths(base, args.recursive, args.include_dirs):
        new_name = transform(p.name)
        if new_name == p.name or not new_name:
            continue
        target = p.with_name(new_name)
        if target.exists() and target != p:
            print(f"SKIP (Ziel existiert): {p} -> {target}")
            skipped += 1
            continue
        print(f"{p} -> {target}")
        if not args.dry_run:
            p.rename(target)
        changed += 1

    print(f"Fertig. {'(Dry-Run) ' if args.dry_run else ''}Umbenannt: {changed}, Übersprungen: {skipped}")

if __name__ == "__main__":
    # Execute the CLI when run as a script.
    main()
