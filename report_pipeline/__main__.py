"""Command-line entry point for the report pipeline.

Running ``python -m report_pipeline`` or the corresponding console script
invokes this module.  It parses command-line arguments via
:mod:`report_pipeline.cli` and executes the lightweight report pipeline,
producing a PDF report from distance files.
"""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

from . import cli
from .orchestrator import ReportOrchestrator
from .plotting import figure_factory
from .pdf import writer as pdf_writer


class _Plotter:
    """Adapter exposing ``make_overlay`` for :class:`ReportOrchestrator`."""

    def __init__(
        self,
        max_per_page: int | None,
        color_mapping: str,
        title: str | None,
        plot_type: str,
    ) -> None:
        self.max_per_page = max_per_page or 6
        self.color_mapping = color_mapping
        self.title = title
        self.plot_type = plot_type

    def make_overlay(
        self,
        items,
        title: str | None = None,
        plot_type: str | None = None,
        show_legend: bool = False,
    ):
        return figure_factory.make_overlay(
            items,
            title=title or self.title,
            max_per_page=self.max_per_page,
            color_strategy=self.color_mapping,
            plot_type=plot_type or self.plot_type,
            show_legend=show_legend,
        )


class _PDFWriter:
    """Persist figures as a PDF report."""

    def write(self, figures, out_path: Path, title: str):
        out_path.parent.mkdir(parents=True, exist_ok=True)
        return pdf_writer.write(figures, out_path, title)


def main(argv: Sequence[str] | None = None) -> None:
    """Run the report pipeline based on ``argv`` arguments."""

    ns = cli.parse_args(argv)
    if ns.dry_run:
        return
    builder = ns.builder_factory(ns)
    plotter = _Plotter(ns.max_per_page, ns.color_mapping, ns.title, ns.plot_type)
    pdf_writer_instance = _PDFWriter()
    orchestrator = ReportOrchestrator(plotter, pdf_writer_instance, builder)
    pdf_path = orchestrator.run(ns.out, ns.title or "")
    print(pdf_path)


if __name__ == "__main__":  # pragma: no cover - CLI entry point
    main()
