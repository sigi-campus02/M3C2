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

import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from scipy.stats import norm, weibull_min, probplot

from ..domain import DistanceFile
from .loader import load_distance_series

logger = logging.getLogger(__name__)

def _chunked(seq: Iterable[DistanceFile], size: int) -> Iterator[list[DistanceFile]]:
    """Yield lists of *size* items taken from *seq* until exhausted."""

    it = iter(seq)
    while True:
        chunk = list(islice(it, size))
        if not chunk:
            break
        yield chunk


def _overlay_histogram(ax, data, colors) -> None:
    for label, arr in data.items():
        ax.hist(arr, bins=30, histtype="step", label=label, color=colors.get(label))
    ax.set_xlabel("Distance")
    ax.set_ylabel("Count")


def _overlay_gauss(ax, data, colors) -> None:
    if not data:
        return
    all_vals = np.concatenate(list(data.values()))
    x = np.linspace(float(np.min(all_vals)), float(np.max(all_vals)), 500)
    for label, arr in data.items():
        mu = float(np.mean(arr))
        std = float(np.std(arr))
        ax.plot(
            x,
            norm.pdf(x, mu, std),
            color=colors.get(label),
            linestyle="--" if label.lower() != "cc" else "-",
            linewidth=2,
            label=rf"{label} Gauss (\mu={mu:.4f}, \sigma={std:.4f})",
        )
    ax.set_xlabel("M3C2 distance")
    ax.set_ylabel("Density")


def _overlay_weibull(ax, data, colors) -> None:
    if not data:
        return
    all_vals = np.concatenate(list(data.values()))
    x = np.linspace(float(np.min(all_vals)), float(np.max(all_vals)), 500)
    for label, arr in data.items():
        try:
            a, loc, b = weibull_min.fit(arr)
            ax.plot(
                x,
                weibull_min.pdf(x, a, loc=loc, scale=b),
                color=colors.get(label),
                linestyle="--" if label.lower() != "cc" else "-",
                linewidth=2,
                label=rf"{label} Weibull (a={a:.2f}, b={b:.4f})",
            )
        except (ValueError, RuntimeError) as e:
            logger.warning("Weibull fit failed for %s: %s", label, e)
    ax.set_xlabel("M3C2 distance")
    ax.set_ylabel("Density")


def _overlay_boxplot(ax, data, colors) -> None:
    try:
        import seaborn as sns

        records = [pd.DataFrame({"Version": label, "Distanz": arr}) for label, arr in data.items()]
        if not records:
            return
        df = pd.concat(records, ignore_index=True)
        order = list(df["Version"].unique())
        palette = {v: colors.get(v) for v in order}
        sns.boxplot(data=df, x="Version", y="Distanz", hue="Version", palette=palette, legend=False, order=order, ax=ax)
    except Exception:
        labels = list(data.keys())
        arrs = [data[v] for v in labels]
        b = ax.boxplot(arrs, labels=labels, patch_artist=True)
        for patch, v in zip(b["boxes"], labels):
            c = colors.get(v, "#aaaaaa")
            patch.set_facecolor(c)
            patch.set_alpha(0.5)
    ax.set_xlabel("Version")
    ax.set_ylabel("M3C2 distance")


def _overlay_qq(ax, data, colors) -> None:
    for label, arr in data.items():
        (osm, osr), (slope, intercept, r) = probplot(arr, dist="norm")
        ax.plot(osm, osr, marker="o", linestyle="", label=label, color=colors.get(label))
        ax.plot(osm, slope * osm + intercept, color=colors.get(label), linestyle="--", alpha=0.7)
    ax.set_xlabel("Theoretical quantiles")
    ax.set_ylabel("Ordered values")


def _overlay_violin(ax, data, colors) -> None:
    try:
        import seaborn as sns

        records = [pd.DataFrame({"Version": label, "Distanz": arr}) for label, arr in data.items()]
        if not records:
            return
        df = pd.concat(records, ignore_index=True)
        palette = {v: colors.get(v) for v in df["Version"].unique()}
        sns.violinplot(data=df, x="Version", y="Distanz", palette=palette, cut=0, inner="quartile", ax=ax)
    except Exception as e:
        labels = list(data.keys())
        arrs = [data[v] for v in labels]
        parts = ax.violinplot(arrs, showmeans=False, showmedians=True)
        for body, v in zip(parts["bodies"], labels):
            c = colors.get(v, "#aaaaaa")
            body.set_facecolor(c)
            body.set_alpha(0.7)
        ax.set_xticks(range(1, len(labels) + 1))
        ax.set_xticklabels(labels)
        logger.warning("Violinplot fallback used due to missing seaborn: %s", e)
    ax.set_xlabel("Version")
    ax.set_ylabel("M3C2 distance")


def make_overlay(
    items: list[DistanceFile],
    title: str | None = None,
    max_per_page: int = 6,
    color_strategy: str = "auto",
    plot_type: str = "histogram",
) -> list[Figure]:
    """Create overlay figures for ``items``.

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
        Strategy influencing colour assignment.  ``"auto"`` attempts to pick a
        sensible default, ``"by_label"`` assigns identical colours to matching
        labels and ``"by_folder"`` groups by the parent folder name.  Colour
        selection is deterministic across calls.
    plot_type:
        Type of overlay plot to generate.  Supported values are ``"histogram"``,
        ``"gauss"``, ``"weibull"``, ``"boxplot"``, ``"qq"`` and ``"violin"``.

    Returns
    -------
    list[matplotlib.figure.Figure]
        One or more matplotlib figures containing overlay plots of the
        requested type.
    """

    figures: list[Figure] = []

    color_cycle = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])

    # Determine key function for colour assignment
    if color_strategy == "by_label":
        keyfunc = lambda item: item.label
    elif color_strategy == "by_folder":
        keyfunc = lambda item: str(item.path.parent)
    elif color_strategy == "auto":
        labels = [i.label for i in items]
        if len(set(labels)) == len(labels):
            keyfunc = lambda item: item.label
        else:
            keyfunc = lambda item: str(item.path.parent)
    else:
        raise ValueError(f"Unknown color strategy: {color_strategy}")

    def _colour(key: str) -> str:
        digest = sha256(key.encode("utf8")).hexdigest()
        return color_cycle[int(digest, 16) % len(color_cycle)]

    for chunk_index, chunk in enumerate(_chunked(items, max_per_page)):
        fig, ax = plt.subplots()
        if title and chunk_index == 0:
            ax.set_title(title)

        data: dict[str, np.ndarray] = {}
        colors: dict[str, str] = {}
        for item in chunk:
            arr = load_distance_series(item.path)
            data[item.label] = arr
            colors[item.label] = _colour(keyfunc(item))

        if plot_type == "histogram":
            _overlay_histogram(ax, data, colors)
        elif plot_type == "gauss":
            _overlay_gauss(ax, data, colors)
        elif plot_type == "weibull":
            _overlay_weibull(ax, data, colors)
        elif plot_type == "boxplot":
            _overlay_boxplot(ax, data, colors)
        elif plot_type == "qq":
            _overlay_qq(ax, data, colors)
        elif plot_type == "violin":
            _overlay_violin(ax, data, colors)
        else:
            raise ValueError(f"Unknown plot type: {plot_type}")

        ax.legend()
        fig.tight_layout()
        figures.append(fig)

    return figures
