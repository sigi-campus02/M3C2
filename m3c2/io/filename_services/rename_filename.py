#!/usr/bin/env python3
# rename_group_prefixes.py
import re, argparse, os, logging
from pathlib import Path

from m3c2.io.logging_utils import resolve_log_level, setup_logging

logger = logging.getLogger(__name__)

# Matcht einen "Cloud-Block":  (_|^-)(1|2)-<idx>(-AI)?_cloud
BLOCK = re.compile(
    r'(?P<pre>(^|[_-]))(?P<grp>[12])-(?P<idx>\d+)(?P<ai>-AI)?(?P<cloud>_cloud)'
)

def transform(name: str) -> str:
    def repl(m):
        """Map group numbers to letters inside ``*_cloud`` blocks.

        The regex match ``m`` contains a numeric group identifier ``grp`` and
        the surrounding parts of a ``*_cloud`` block. ``1`` is replaced with
        ``a`` and ``2`` with ``b`` while keeping the rest of the match
        unchanged.
        """
        mapped = 'a' if m.group('grp') == '1' else 'b'
        return f"{m.group('pre')}{mapped}-{m.group('idx')}{m.group('ai') or ''}{m.group('cloud')}"
    return BLOCK.sub(repl, name)

def iter_paths(base: Path, recursive: bool):
    """Liefert Dateien und Ordner. Bei Rekursion bottom-up (sicher für Ordner-Renames)."""
    if recursive:
        for root, dirs, files in os.walk(base, topdown=False):
            for f in files:
                yield Path(root) / f
            for d in dirs:
                yield Path(root) / d
    else:
        for p in base.iterdir():
            yield p  # Dateien und Ordner

def main():
    ap = argparse.ArgumentParser(
        description="Ersetzt in *_cloud-Blöcken die Gruppenkennung: 1-* -> a-*, 2-* -> b-* (in Dateien und Ordnern)."
    )
    ap.add_argument("path", nargs="?", default=".", help="Basisordner (Standard: .)")
    ap.add_argument("-r", "--recursive", action="store_true", help="Auch Unterordner bearbeiten")
    ap.add_argument("-n", "--dry-run", action="store_true", help="Nur anzeigen, nichts ändern")
    args = ap.parse_args()

    setup_logging(level=resolve_log_level())

    base = Path(args.path).resolve()
    changed = skipped = 0

    for p in iter_paths(base, args.recursive):
        new_name = transform(p.name)
        if new_name == p.name:
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
