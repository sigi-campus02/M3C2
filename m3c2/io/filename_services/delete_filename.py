#!/usr/bin/env python3
# remove_cloud_token.py
import argparse, os, logging
from pathlib import Path

from m3c2.io.logging_utils import resolve_log_level, setup_logging

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
    ap = argparse.ArgumentParser(
        description='Entfernt alle Vorkommen von "_cloud" aus Datei-/Ordnernamen.'
    )
    ap.add_argument("path", nargs="?", default=".", help="Basisordner (Standard: .)")
    ap.add_argument("-r", "--recursive", action="store_true", help="Auch Unterordner bearbeiten")
    ap.add_argument("--include-dirs", action="store_true", help="Auch Ordnernamen umbenennen")
    ap.add_argument("-n", "--dry-run", action="store_true", help="Nur anzeigen, nichts ändern")
    args = ap.parse_args()

    setup_logging(level=resolve_log_level())

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
