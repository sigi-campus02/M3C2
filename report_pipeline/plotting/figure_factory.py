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


def _square_limits(x: np.ndarray, y: np.ndarray, pad: float = 0.05) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return square axis limits covering the ``x`` and ``y`` data."""

    x_min, x_max = float(np.min(x)), float(np.max(x))
    y_min, y_max = float(np.min(y)), float(np.max(y))
    v_min = min(x_min, y_min)
    v_max = max(x_max, y_max)
    cx = cy = (v_min + v_max) / 2.0
    half = max((x_max - x_min), (y_max - y_min)) / 2.0
    half = half * (1.0 + pad) if half > 0 else 1.0
    return (cx - half, cx + half), (cy - half, cy + half)


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
    show_legend: bool = True,
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
    show_legend:
        Whether to render a legend on the created axes.

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
        if show_legend:
            ax.legend()
        fig.tight_layout()
        figures.append(fig)

    return figures


def make_bland_altman(items: list[DistanceFile], title: str | None = None) -> list[Figure]:
    """Create a Bland–Altman plot for exactly two distance files."""

    if len(items) != 2:
        raise ValueError("Bland–Altman plot requires exactly two files")
    a = load_distance_series(items[0].path)
    b = load_distance_series(items[1].path)
    n = min(len(a), len(b))
    a, b = a[:n], b[:n]
    if n == 0:
        return []

    mean_vals = (a + b) / 2.0
    diff_vals = a - b
    mean_diff = float(np.mean(diff_vals))
    std_diff = float(np.std(diff_vals, ddof=1))
    upper = mean_diff + 1.96 * std_diff
    lower = mean_diff - 1.96 * std_diff

    fig, ax = plt.subplots()
    if title:
        ax.set_title(title)
    ax.scatter(mean_vals, diff_vals, alpha=0.3)
    ax.axhline(mean_diff, color="red", linestyle="--", label=f"Mean diff {mean_diff:.4f}")
    ax.axhline(upper, color="green", linestyle="--", label=f"+1.96 SD {upper:.4f}")
    ax.axhline(lower, color="green", linestyle="--", label=f"-1.96 SD {lower:.4f}")
    ax.set_xlabel(items[0].label)
    ax.set_ylabel(items[1].label)
    ax.legend()
    fig.tight_layout()
    return [fig]


def make_linear_regression(items: list[DistanceFile], title: str | None = None) -> list[Figure]:
    """Create an OLS linear regression plot for two distance files."""

    if len(items) != 2:
        raise ValueError("Linear regression requires exactly two files")
    x = load_distance_series(items[0].path)
    y = load_distance_series(items[1].path)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    if n < 3:
        return []

    xbar = float(np.mean(x))
    ybar = float(np.mean(y))
    Sxx = float(np.sum((x - xbar) ** 2))
    if Sxx == 0.0:
        return []
    Sxy = float(np.sum((x - xbar) * (y - ybar)))
    b = Sxy / Sxx
    a = ybar - b * xbar
    resid = y - (a + b * x)
    SSE = float(np.sum(resid ** 2))
    s2 = SSE / (n - 2)
    se_b = float(np.sqrt(s2 / Sxx))
    se_a = float(np.sqrt(s2 * (1.0 / n + (xbar ** 2) / Sxx)))
    from scipy.stats import t

    tcrit = float(t.ppf(0.975, df=n - 2))
    b_L, b_U = b - tcrit * se_b, b + tcrit * se_b
    a_L, a_U = a - tcrit * se_a, a + tcrit * se_a

    fig, ax = plt.subplots()
    if title:
        ax.set_title(title)
    ax.scatter(x, y, alpha=0.35, s=12, label="Data")
    (xl, xu), (yl, yu) = _square_limits(x, y, pad=0.05)
    xx = np.array([xl, xu], dtype=float)
    ax.plot(xx, xx, linestyle="--", color="grey", label="y = x")
    ax.plot(xx, a + b * xx, color="red", label=f"OLS: y = {a:.4f} + {b:.4f} x")
    ax.plot(xx, a_U + b_U * xx, linestyle="--", alpha=0.7, label=f"CI upper")
    ax.plot(xx, a_L + b_L * xx, linestyle="--", alpha=0.7, label=f"CI lower")
    ax.fill_between(xx, a_L + b_L * xx, a_U + b_U * xx, alpha=0.12)
    ax.set_xlim(xl, xu)
    ax.set_ylim(yl, yu)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(items[0].label)
    ax.set_ylabel(items[1].label)
    ax.legend(frameon=False)
    fig.tight_layout()
    return [fig]


def make_passing_bablok(items: list[DistanceFile], title: str | None = None) -> list[Figure]:
    """Create a Passing–Bablok regression plot for two distance files."""

    if len(items) != 2:
        raise ValueError("Passing–Bablok requires exactly two files")
    x = load_distance_series(items[0].path)
    y = load_distance_series(items[1].path)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    if n < 2:
        return []

    S = []
    for i in range(n - 1):
        x_i, y_i = x[i], y[i]
        for j in range(i + 1, n):
            x_j, y_j = x[j], y[j]
            if (x_i == x_j) and (y_i == y_j):
                continue
            if x_i == x_j:
                S.append(np.inf if (y_i > y_j) else -np.inf)
                continue
            g = (y_i - y_j) / (x_i - x_j)
            if g == -1:
                continue
            S.append(g)
    if not S:
        return []
    S = np.array(S, dtype=float)
    S.sort()
    N = int(len(S))
    K = int((S < -1).sum())
    if N % 2 != 0:
        idx = int((N + 1) / 2 + K) - 1
        b = float(S[idx])
    else:
        idx = int(N / 2 + K) - 1
        b = float(0.5 * (S[idx] + S[idx + 1]))
    a = float(np.median(y - b * x))
    from scipy import stats as st

    C = 0.95
    gamma = 1 - C
    q = 1 - (gamma / 2.0)
    w = float(st.norm.ppf(q))
    C_gamma = w * np.sqrt((n * (n - 1) * (2 * n + 5)) / 18.0)
    M1 = int(np.round((N - C_gamma) / 2.0))
    M2 = int(N - M1 + 1)
    b_L = float(S[M1 + K - 1])
    b_U = float(S[M2 + K - 1])
    a_L = float(np.median(y - b_U * x))
    a_U = float(np.median(y - b_L * x))

    fig, ax = plt.subplots()
    if title:
        ax.set_title(title)
    ax.scatter(x, y, alpha=0.35, s=12, label="Data")
    (xl, xu), (yl, yu) = _square_limits(x, y, pad=0.05)
    xx = np.array([xl, xu], dtype=float)
    ax.plot(xx, xx, linestyle="--", color="grey", label="y = x")
    ax.plot(xx, a + b * xx, color="red", label=f"PB: y = {a:.4f} + {b:.4f} x")
    ax.plot(xx, a_U + b_U * xx, linestyle="--", alpha=0.7, label="CI upper")
    ax.plot(xx, a_L + b_L * xx, linestyle="--", alpha=0.7, label="CI lower")
    ax.fill_between(xx, a_L + b_L * xx, a_U + b_U * xx, alpha=0.12)
    ax.set_xlim(xl, xu)
    ax.set_ylim(yl, yu)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xlabel(items[0].label)
    ax.set_ylabel(items[1].label)
    ax.legend(frameon=False)
    fig.tight_layout()
    return [fig]


def make_grouped_bar(items: list[DistanceFile], title: str | None = None) -> list[Figure]:
    """Create grouped bar plots comparing WITH and INLIER data per label."""

    data_with: dict[str, np.ndarray] = {}
    data_inl: dict[str, np.ndarray] = {}
    for item in items:
        arr = load_distance_series(item.path)
        if (item.group or "").lower() == "inlier":
            data_inl[item.label] = arr
        else:
            data_with[item.label] = arr

    all_labels = sorted(set(data_with.keys()) | set(data_inl.keys()))
    means_with: list[float] = []
    stds_with: list[float] = []
    means_inl: list[float] = []
    stds_inl: list[float] = []
    for lbl in all_labels:
        arr_w = data_with.get(lbl, np.array([]))
        arr_i = data_inl.get(lbl, np.array([]))
        means_with.append(float(np.abs(np.mean(arr_w))) if arr_w.size else np.nan)
        stds_with.append(float(np.std(arr_w)) if arr_w.size else np.nan)
        means_inl.append(float(np.abs(np.mean(arr_i))) if arr_i.size else np.nan)
        stds_inl.append(float(np.std(arr_i)) if arr_i.size else np.nan)

    x = np.arange(len(all_labels))
    width = 0.4
    fig, ax = plt.subplots(2, 1, figsize=(max(10, len(all_labels) * 1.8), 8), sharex=True)
    if title:
        fig.suptitle(title)
    ax[0].bar(x - width / 2, means_with, width, label="WITH")
    ax[0].bar(x + width / 2, means_inl, width, label="INLIER", alpha=0.55)
    ax[0].set_ylabel("Mean (|μ|)")
    ax[0].set_ylim(bottom=0)
    ax[0].legend()

    ax[1].bar(x - width / 2, stds_with, width, label="WITH")
    ax[1].bar(x + width / 2, stds_inl, width, label="INLIER", alpha=0.55)
    ax[1].set_ylabel("Std (σ)")
    ax[1].set_xticks(x)
    ax[1].set_xticklabels(all_labels, rotation=30, ha="right")
    ax[1].set_ylim(bottom=0)
    ax[1].legend()

    fig.tight_layout()
    return [fig]
