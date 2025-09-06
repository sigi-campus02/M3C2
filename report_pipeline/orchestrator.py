"""Lightweight orchestration for building PDF reports from plot jobs.

The :class:`ReportOrchestrator` coordinates the interaction between a plotter
object and a PDF writer.  Given a sequence of :class:`~report_pipeline.domain.PlotJob`
instances it requests plots from the plotter, collects the resulting figures and
finally delegates to the PDF writer to persist the figures as a report.
"""

from __future__ import annotations


from pathlib import Path

from .strategies.base import JobBuilder


class ReportOrchestrator:
    """Orchestrate plot creation and PDF generation for a series of jobs."""

    def __init__(self, plotter, pdf_writer, builder: JobBuilder) -> None:
        """Create a new orchestrator.

        Parameters
        ----------
        plotter:
            Object providing a ``make_overlay`` method returning a figure.
        pdf_writer:
            Object providing a ``write`` method accepting a sequence of figures,
            an output path and a document title, returning the path to the
            generated PDF report.
        builder:
            Instance capable of creating :class:`~report_pipeline.domain.PlotJob`
            objects via :meth:`~report_pipeline.strategies.base.JobBuilder.build_jobs`.
        """

        self.plotter = plotter
        self.pdf_writer = pdf_writer
        self.builder = builder

    def run(self, out_path: Path, title: str) -> Path:
        """Generate figures for the builder's jobs and write them to a PDF report."""

        jobs = self.builder.build_jobs()
        all_figures: list = []

        # Detect whether the builder exposes a legend flag.
        try:
            show_legend = object.__getattribute__(self.builder, "legend")
        except AttributeError:
            show_legend = False

        for job in jobs:
            plot_type = getattr(job, "plot_type", "histogram")

            # Branch based on the desired plot type for the current job.
            if plot_type in {"histogram", "gauss", "weibull", "boxplot", "qq", "violin"}:
                current_figures = self.plotter.make_overlay(
                    job.items,
                    title=job.page_title,
                    plot_type=plot_type,
                    show_legend=show_legend,
                )
            elif plot_type == "bland-altman":
                current_figures = self.plotter.make_bland_altman(job.items, title=job.page_title)
            elif plot_type == "linear-regression":
                current_figures = self.plotter.make_linear_regression(job.items, title=job.page_title)
            elif plot_type == "passing-bablok":
                current_figures = self.plotter.make_passing_bablok(job.items, title=job.page_title)
            elif plot_type == "grouped-bar":
                current_figures = self.plotter.make_grouped_bar(job.items, title=job.page_title)
            else:
                raise ValueError(f"Unknown plot type: {plot_type}")

            # Aggregate figures from this job into the full report collection.
            all_figures.extend(current_figures)

        # Write the collected figures to the output PDF.
        pdf_path = self.pdf_writer.write(all_figures, out_path, title)
        return pdf_path
