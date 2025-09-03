#!/usr/bin/env python3
# remove_cloud_token.py
"""CLI tool for removing `_cloud` tokens from file and directory names.

The script walks a base directory and renames entries by deleting the
`_cloud` token from their names. It supports recursive traversal,
optionally includes directory names in the renaming process and offers a
dry-run mode to preview changes without modifying the filesystem.
"""

import argparse, os, logging
from pathlib import Path

from m3c2.io.logging_utils import setup_logging

logger = logging.getLogger(__name__)

def transform(name: str) -> str:
    """Remove all occurrences of the ``python_`` marker from a name.

    The function strips every ``python_`` substring from the provided file or
    directory name.  The remainder of the name, including any extension, is
    returned unchanged.  If the marker constitutes the entire original name an
    empty string is returned, allowing the caller to skip renaming.
    """
    return name.replace("python_", "")

def iter_paths(base: Path, recursive: bool, include_dirs: bool):
    """Liefer Pfade: Dateien immer, Ordner nur wenn include_dirs=True. Bottom-up bei Rekursion."""
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
    """Entry point for the ``delete_filename`` CLI.

    The command removes every occurrence of ``python_`` from file and directory
    names.  It can operate on a single directory or recursively on a directory
    tree and supports a dry-run mode to preview changes.

    Usage
    -----
    python -m m3c2.io.filename_services.delete_filename [PATH]
        [-r] [--include-dirs] [-n]

    Parameters
    ----------
    PATH : str, optional
        Base directory to process. Defaults to the current directory.
    -r, --recursive : bool
        Recurse into subdirectories.
    --include-dirs : bool
        Apply renaming to directory names as well.
    -n, --dry-run : bool
        Show actions without performing any renaming.
    """
    ap = argparse.ArgumentParser(
        description='Entfernt alle Vorkommen von "_cloud" aus Datei-/Ordnernamen.'
    )
    ap.add_argument("path", nargs="?", default=".", help="Basisordner (Standard: .)")
    ap.add_argument("-r", "--recursive", action="store_true", help="Auch Unterordner bearbeiten")
    ap.add_argument("--include-dirs", action="store_true", help="Auch Ordnernamen umbenennen")
    ap.add_argument("-n", "--dry-run", action="store_true", help="Nur anzeigen, nichts ändern")
    args = ap.parse_args()

    # Initialize logging using configuration defaults and environment values.
    setup_logging()

    base = Path(args.path).resolve()
    changed = skipped = 0

    for p in iter_paths(base, args.recursive, args.include_dirs):
        new_name = transform(p.name)
        if new_name == p.name or not new_name:
            continue
        target = p.with_name(new_name)
        if target.exists() and target != p:
            logger.warning(f"SKIP (Ziel existiert): {p} -> {target}")
            skipped += 1
            continue
        logger.info(f"{p} -> {target}")
        if not args.dry_run:
            p.rename(target)
        changed += 1

    logger.info(
        f"Fertig. {'(Dry-Run) ' if args.dry_run else ''}Umbenannt: {changed}, Übersprungen: {skipped}"
    )

if __name__ == "__main__":
    main()
