from __future__ import annotations

"""Command line interface for the simple report pipeline.

This module exposes a minimal argument parser with three subcommands for
collecting distance files that will later be turned into plots and bundled
into a PDF report.  The subcommands map directly to the job building
strategies defined in :mod:`report_pipeline.strategies`:

``folder``
    Build jobs from all distance files found in a single directory.
``multifolder``
    Combine selected file names from multiple folders located beneath a common
    base directory.
``files``
    Use an explicit list of distance files.

Each subcommand shares a common set of options controlling the report
generation (``--out``, ``--title`` …).  The parser stores a ``builder_factory``
callable on the resulting namespace which can be used to construct an
appropriate :class:`~report_pipeline.strategies.base.JobBuilder` instance.

The :func:`run` convenience function demonstrates how the parsed options can be
used to create jobs.  It intentionally performs only a dry orchestration and
returns the generated jobs, leaving the actual plotting/report creation to
higher level code.
"""

from pathlib import Path
from typing import Callable, Iterable, Sequence
import argparse

from .domain import PlotJob
from .strategies.base import JobBuilder
from .strategies.folder import FolderJobBuilder
from .strategies.multifolder import MultiFolderJobBuilder
from .strategies.files import FilesJobBuilder

# Type alias for the factory stored on each subparser
BuilderFactory = Callable[[argparse.Namespace], JobBuilder]


# ---------------------------------------------------------------------------
# Parser construction
# ---------------------------------------------------------------------------

def _add_shared_options(parser: argparse.ArgumentParser) -> None:
    """Attach options common to all subcommands to *parser*."""

    parser.add_argument(
        "--out",
        type=Path,
        default=Path("report"),
        help="Output directory for generated plots and report.",
    )
    parser.add_argument(
        "--title",
        type=str,
        help="Title displayed on the generated report.",
    )
    parser.add_argument(
        "--max-per-page",
        dest="max_per_page",
        type=int,
        help="Maximum number of plots per PDF page.",
    )
    parser.add_argument(
        "--color-mapping",
        dest="color_mapping",
        type=Path,
        help="Optional JSON file mapping labels to colors.",
    )
    parser.add_argument(
        "--legend",
        action="store_true",
        help="Include a legend in the plots.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse arguments but do not read files or generate plots.",
    )


def build_parser() -> argparse.ArgumentParser:
    """Return the top-level argument parser for the report pipeline."""

    parser = argparse.ArgumentParser(
        description="Generate comparison reports for distance files.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ------------------------------------------------------------------
    # folder subcommand
    # ------------------------------------------------------------------
    folder_parser = subparsers.add_parser(
        "folder", help="Process all distance files within a folder."
    )
    folder_parser.add_argument("folder", type=Path, help="Directory containing distance files")
    folder_parser.add_argument(
        "--paired",
        action="store_true",
        help="Expect exactly two files; raise an error otherwise.",
    )
    _add_shared_options(folder_parser)
    folder_parser.set_defaults(
        builder_factory=lambda ns: FolderJobBuilder(
            folder=ns.folder, paired=ns.paired
        )
    )

    # ------------------------------------------------------------------
    # multifolder subcommand
    # ------------------------------------------------------------------
    multifolder_parser = subparsers.add_parser(
        "multifolder", help="Load specific files from multiple folders."
    )
    multifolder_parser.add_argument(
        "data_dir", type=Path, help="Base directory containing the folders"
    )
    multifolder_parser.add_argument(
        "--folders",
        nargs="+",
        required=True,
        help="Folder names located below the base directory.",
    )
    multifolder_parser.add_argument(
        "--filenames",
        nargs="+",
        required=True,
        help="File names to load from each folder.",
    )
    multifolder_parser.add_argument(
        "--paired",
        action="store_true",
        help="Expect exactly two files overall; raise an error otherwise.",
    )
    _add_shared_options(multifolder_parser)
    multifolder_parser.set_defaults(
        builder_factory=lambda ns: MultiFolderJobBuilder(
            data_dir=ns.data_dir,
            folders=ns.folders,
            filenames=ns.filenames,
            paired=ns.paired,
        )
    )

    # ------------------------------------------------------------------
    # files subcommand
    # ------------------------------------------------------------------
    files_parser = subparsers.add_parser(
        "files", help="Process an explicit list of distance files."
    )
    files_parser.add_argument(
        "files",
        nargs="+",
        type=Path,
        help="Paths to distance files that should be plotted.",
    )
    files_parser.add_argument(
        "--paired",
        action="store_true",
        help="Expect exactly two files; raise an error otherwise.",
    )
    _add_shared_options(files_parser)
    files_parser.set_defaults(
        builder_factory=lambda ns: FilesJobBuilder(
            files=ns.files, paired=ns.paired
        )
    )

    return parser


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Parse ``argv`` using the report pipeline argument parser."""

    parser = build_parser()
    return parser.parse_args(argv)


def run(argv: Sequence[str] | None = None) -> list[PlotJob]:
    """Parse ``argv`` and return plotting jobs.

    When the ``--dry-run`` option is supplied no filesystem interaction takes
    place and an empty list is returned.  The function is intended mainly for
    tests and higher level orchestration code.
    """

    ns = parse_args(argv)
    factory: BuilderFactory = ns.builder_factory
    if ns.dry_run:
        return []
    builder = factory(ns)
    return builder.build_jobs()


__all__ = ["build_parser", "parse_args", "run"]
