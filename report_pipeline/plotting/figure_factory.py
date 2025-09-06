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
from hashlib import sha256
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
    color_mapping: str = "auto",
    legend: bool = False,
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
    color_mapping:
        Strategy influencing colour assignment.  ``"auto"`` attempts to pick a
        sensible default, ``"by_label"`` assigns identical colours to matching
        labels and ``"by_folder"`` groups by the parent folder name.  Colour
        selection is deterministic across calls.
    legend:
        When ``True`` a legend is added to each figure.

    Returns
    -------
    list[matplotlib.figure.Figure]
        One or more matplotlib figures containing overlay histograms.
    """

    figures: list[Figure] = []

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    # Determine key function for colour assignment
    if color_mapping == "by_label":
        keyfunc = lambda item: item.label
    elif color_mapping == "by_folder":
        keyfunc = lambda item: str(item.path.parent)
    elif color_mapping == "auto":
        labels = [i.label for i in items]
        if len(set(labels)) == len(labels):
            keyfunc = lambda item: item.label
        else:
            keyfunc = lambda item: str(item.path.parent)
    else:
        raise ValueError(f"Unknown color strategy: {color_mapping}")

    def _colour(key: str) -> str:
        digest = sha256(key.encode("utf8")).hexdigest()
        return color_cycle[int(digest, 16) % len(color_cycle)]

    for chunk_index, chunk in enumerate(_chunked(items, max_per_page)):
        fig, ax = plt.subplots()
        if title and chunk_index == 0:
            ax.set_title(title)

        for item in chunk:
            data = load_distance_series(item.path)
            color = _colour(keyfunc(item))
            ax.hist(data, bins=30, histtype="step", label=item.label, color=color)

        ax.set_xlabel("Distance")
        ax.set_ylabel("Count")
        if legend:
            ax.legend()
        fig.tight_layout()
        figures.append(fig)

    return figures
