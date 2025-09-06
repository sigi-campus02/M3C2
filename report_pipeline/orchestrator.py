from __future__ import annotations

"""Lightweight orchestration for building PDF reports from plot jobs.

The :class:`ReportOrchestrator` coordinates the interaction between a plotter
object and a PDF writer.  Given a sequence of :class:`~report_pipeline.domain.PlotJob`
instances it requests plots from the plotter, collects the resulting figures and
finally delegates to the PDF writer to persist the figures as a report.
"""

from pathlib import Path
from typing import Iterable

from .domain import PlotJob


class ReportOrchestrator:
    """Orchestrate plot creation and PDF generation for a series of jobs."""

    def __init__(self, plotter, pdf_writer) -> None:
        """Create a new orchestrator.

        Parameters
        ----------
        plotter:
            Object providing a ``make_overlay`` method returning a figure.
        pdf_writer:
            Object providing a ``write`` method accepting a sequence of figures
            and returning the path to the generated PDF report.
        """

        self.plotter = plotter
        self.pdf_writer = pdf_writer

    def run(self, jobs: Iterable[PlotJob]) -> Path:
        """Generate figures for *jobs* and write them to a PDF report.

        Each job's ``items`` attribute is passed to ``plotter.make_overlay``
        along with the job's ``page_title``.  The created figures are then
        handed to ``pdf_writer.write`` which returns the path to the generated
        PDF.
        """

        figures: list = []
        for job in jobs:
            fig = self.plotter.make_overlay(job.items, title=job.page_title)
            figures.append(fig)
        pdf_path = self.pdf_writer.write(figures)
        return pdf_path
