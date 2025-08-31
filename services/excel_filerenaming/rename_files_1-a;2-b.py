#!/usr/bin/env python3
# rename_group_prefixes.py
"""Rename numeric group prefixes in file and directory names to alphabetic ones.

The script scans file or directory names containing patterns like
``1-<index>_cloud`` or ``2-<index>_cloud`` and converts them to
``a-<index>_cloud`` and ``b-<index>_cloud`` respectively.
"""

import re, argparse, os
from pathlib import Path

# Match a "cloud" block:  (_|^-)(1|2)-<idx>(-AI)?_cloud
BLOCK = re.compile(
    r'(?P<pre>(^|[_-]))(?P<grp>[12])-(?P<idx>\d+)(?P<ai>-AI)?(?P<cloud>_cloud)'
)

def transform(name: str) -> str:
    """Replace numeric group prefixes with alphabetic ones in a name.

    Parameters
    ----------
    name: str
        Original file or directory name.

    Returns
    -------
    str
        Name with group identifiers converted.
    """
    def repl(m):
        mapped = 'a' if m.group('grp') == '1' else 'b'
        return f"{m.group('pre')}{mapped}-{m.group('idx')}{m.group('ai') or ''}{m.group('cloud')}"
    return BLOCK.sub(repl, name)

def iter_paths(base: Path, recursive: bool):
    """Yield files and directories from ``base`` respecting recursion.

    Traverses bottom-up when ``recursive`` is ``True`` so that directory
    renames do not invalidate child paths.
    """
    if recursive:
        for root, dirs, files in os.walk(base, topdown=False):
            for f in files:
                yield Path(root) / f
            for d in dirs:
                yield Path(root) / d
    else:
        for p in base.iterdir():
            yield p  # files and directories

def main():
    """Command-line interface for batch renaming of group prefixes."""
    ap = argparse.ArgumentParser(
        description="Ersetzt in *_cloud-Blöcken die Gruppenkennung: 1-* -> a-*, 2-* -> b-* (in Dateien und Ordnern)."
    )
    ap.add_argument("path", nargs="?", default=".", help="Basisordner (Standard: .)")
    ap.add_argument("-r", "--recursive", action="store_true", help="Auch Unterordner bearbeiten")
    ap.add_argument("-n", "--dry-run", action="store_true", help="Nur anzeigen, nichts ändern")
    args = ap.parse_args()

    base = Path(args.path).resolve()
    changed = skipped = 0

    # Iterate through all candidate paths and apply the renaming.
    for p in iter_paths(base, args.recursive):
        new_name = transform(p.name)
        if new_name == p.name:
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
    main()
