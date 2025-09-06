from __future__ import annotations

"""Factory functions creating matplotlib figures for distance overlays.

The module focuses on a single high level function :func:`make_overlay` which
accepts a list of :class:`~report_pipeline.domain.DistanceFile` instances and
creates one or more overlay histograms.  The function is intentionally
lightweight – it merely loads the underlying distance series and visualises
them using matplotlib's default colour cycle.  When ``max_per_page`` is
smaller than the number of items, multiple figures are created.  The caller is
responsible for further processing, for instance writing the figures to a PDF
report.
"""

from itertools import islice
from typing import Iterable, Iterator

import matplotlib.pyplot as plt
from matplotlib.figure import Figure

from ..domain import DistanceFile
from .loader import load_distance_series


def _chunked(seq: Iterable[DistanceFile], size: int) -> Iterator[list[DistanceFile]]:
    """Yield lists of *size* items taken from *seq* until exhausted."""

    it = iter(seq)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk


def make_overlay(
    items: list[DistanceFile],
    title: str | None = None,
    max_per_page: int = 6,
    color_strategy: str = "cycle",
) -> list[Figure]:
    """Create overlay histogram figures for ``items``.

    Parameters
    ----------
    items:
        List of :class:`DistanceFile` objects describing the data to plot.
    title:
        Optional title applied to the first generated figure.
    max_per_page:
        Maximum number of data series per figure.  Items beyond this limit are
        rendered on additional pages.
    color_strategy:
        Currently only ``"cycle"`` and ``"group"`` are recognised.  The
        strategy influences colour assignment.  With ``"group"`` items sharing
        the same ``group`` attribute receive identical colours.

    Returns
    -------
    list[matplotlib.figure.Figure]
        One or more matplotlib figures containing overlay histograms.
    """

    figures: list[Figure] = []

    # Determine colour assignment strategy
    if color_strategy == "group":
        groups: dict[str | None, str] = {}
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        for idx, g in enumerate({item.group for item in items}):
            groups[g] = color_cycle[idx % len(color_cycle)]
    else:
        groups = {}
        color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    for chunk_index, chunk in enumerate(_chunked(items, max_per_page)):
        fig, ax = plt.subplots()
        if title and chunk_index == 0:
            ax.set_title(title)

        for i, item in enumerate(chunk):
            data = load_distance_series(item.path)
            color = (
                groups.get(item.group)
                if color_strategy == "group"
                else color_cycle[i % len(color_cycle)]
            )
            ax.hist(data, bins=30, histtype="step", label=item.label, color=color)

        ax.set_xlabel("Distance")
        ax.set_ylabel("Count")
        ax.legend()
        fig.tight_layout()
        figures.append(fig)

    return figures
