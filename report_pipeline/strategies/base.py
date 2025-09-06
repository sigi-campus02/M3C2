"""Protocols for job building strategies."""

from __future__ import annotations


from typing import Protocol

from ..domain import PlotJob


class JobBuilder(Protocol):
    """Protocol for objects that create :class:`~report_pipeline.domain.PlotJob` instances."""

    def build_jobs(self) -> list[PlotJob]:
        """Return a list of jobs describing plots to generate."""
        ...