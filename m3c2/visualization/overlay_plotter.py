from __future__ import annotations

import logging
import os
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, weibull_min, probplot

logger = logging.getLogger(__name__)


def get_common_range(data: Dict[str, np.ndarray]) -> Tuple[float, float, np.ndarray]:
    """Determine a common value range for multiple data arrays."""
    all_vals = np.concatenate(list(data.values())) if data else np.array([])
    data_min, data_max = (
        (float(np.min(all_vals)), float(np.max(all_vals))) if all_vals.size else (0.0, 1.0)
    )
    x = np.linspace(data_min, data_max, 500)
    return data_min, data_max, x


def plot_overlay_histogram(
    fid: str,
    fname: str,
    data: Dict[str, np.ndarray],
    bins: int,
    data_min: float,
    data_max: float,
    colors: Dict[str, str],
    outdir: str,
    title_text: str | None = None,
    labels_order: List[str] | None = None,
) -> None:
    """Plot histograms of several datasets in a single figure.

    Args:
        fid: Identifier of the dataset, used in the output filename.
        fname: Name of the file being analysed, used in plot titles.
        data: Mapping of labels to data arrays to be plotted.
        bins: Number of bins for the histograms.
        data_min: Lower bound of the shared value range.
        data_max: Upper bound of the shared value range.
        colors: Mapping of labels to colour codes for the plots.
        outdir: Directory in which the figure will be saved.
        title_text: Optional custom title for the figure.
        labels_order: Optional explicit order of labels; defaults to ``data`` keys.

    Saves:
        ``<outdir>/<fid>_<fname>_OverlayHistogramm.png`` – the overlay histogram
        showing the distributions of all provided datasets.
    """
    plt.figure(figsize=(10, 6))
    labels = labels_order or list(data.keys())
    for v in labels:
        arr = data[v]
        plt.hist(
            arr,
            bins=bins,
            range=(data_min, data_max),
            density=True,
            histtype="step",
            linewidth=2,
            label=v,
            color=colors.get(v),
        )
    plt.title(title_text or f"Histogramm – {fid}/{fname}")
    plt.xlabel("M3C2 distance")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{fid}_{fname}_OverlayHistogramm.png"))
    plt.close()


def plot_overlay_gauss(
    fid: str,
    fname: str,
    data: Dict[str, np.ndarray],
    gauss_params: Dict[str, Tuple[float, float]],
    x: np.ndarray,
    colors: Dict[str, str],
    outdir: str,
    title_text: str | None = None,
    labels_order: List[str] | None = None,
) -> None:
    """Plot Gaussian probability density functions for multiple datasets.

    Parameters
    ----------
    fid : str
        Identifier of the feature or dataset group, used in the plot title
        and output filename.
    fname : str
        Name of the input file for display in the figure title and filename.
    data : Dict[str, np.ndarray]
        Mapping of labels to the underlying samples. The labels define which
        Gaussian parameters are plotted.
    gauss_params : Dict[str, Tuple[float, float]]
        Pre-computed Gaussian parameters ``(mean, std)`` for each label in
        ``data``.
    x : np.ndarray
        Common x-range on which the Gaussian curves are evaluated.
    colors : Dict[str, str]
        Mapping of labels to colors used for the curves.
    outdir : str
        Directory where the resulting image is stored.
    title_text : str | None, optional
        Custom text for the plot title. When ``None`` a default title is used.
    labels_order : List[str] | None, optional
        Custom order in which to plot the labels. When ``None`` the dictionary
        order of ``data`` is used.

    The function overlays the Gaussian PDF for each dataset on a single set of
    axes, allowing visual comparison of their fitted distributions.
    """
    plt.figure(figsize=(10, 6))
    labels = labels_order or list(data.keys())
    for v in labels:
        if v in gauss_params:
            mu, std = gauss_params[v]
            plt.plot(
                x,
                norm.pdf(x, mu, std),
                color=colors.get(v),
                linestyle="--" if v.lower() != "cc" else "-",
                linewidth=2,
                label=rf"{v} Gauss ($\mu$={mu:.4f}, $\sigma$={std:.4f})",
            )
    plt.title(title_text or f"Overlay Gauss-Fits – {fid}/{fname}")
    plt.xlabel("M3C2 distance")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{fid}_{fname}_OverlayGaussFits.png"))
    plt.close()


def plot_overlay_weibull(
    fid: str,
    fname: str,
    data: Dict[str, np.ndarray],
    x: np.ndarray,
    colors: Dict[str, str],
    outdir: str,
    title_text: str | None = None,
    labels_order: List[str] | None = None,
) -> None:
    """Fit Weibull distributions and plot their probability densities.

    Each array in ``data`` is fitted with :func:`scipy.stats.weibull_min.fit`
    to estimate the shape, location, and scale parameters of a Weibull
    distribution.  The resulting probability density functions are evaluated
    on ``x`` and plotted together.  The combined overlay is written to
    ``{fid}_{fname}_OverlayWeibullFits.png`` inside ``outdir``.
    """
    weibull_params: Dict[str, Tuple[float, float, float]] = {}
    for v, arr in data.items():
        try:
            a, loc, b = weibull_min.fit(arr)
            weibull_params[v] = (float(a), float(loc), float(b))
        except Exception as e:
            logger.warning("[Report] Weibull-Fit fehlgeschlagen (%s/%s, %s): %s", fid, fname, v, e)

    plt.figure(figsize=(10, 6))
    labels = labels_order or list(weibull_params.keys())
    for v in labels:
        if v in weibull_params:
            a, loc, b = weibull_params[v]
            plt.plot(
                x,
                weibull_min.pdf(x, a, loc=loc, scale=b),
                color=colors.get(v),
                linestyle="--" if v.lower() != "cc" else "-",
                linewidth=2,
                label=rf"{v} Weibull (a={a:.2f}, b={b:.4f})",
            )
    plt.title(title_text or f"Overlay Weibull-Fits – {fid}/{fname}")
    plt.xlabel("M3C2 distance")
    plt.ylabel("Density")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{fid}_{fname}_OverlayWeibullFits.png"))
    plt.close()


def plot_overlay_boxplot(
    fid: str,
    fname: str,
    data: Dict[str, np.ndarray],
    colors: Dict[str, str],
    outdir: str,
    title_text: str | None = None,
    labels_order: List[str] | None = None,
) -> None:
    """Create a box plot comparing distributions from multiple versions.

    The function reorganizes the ``data`` dictionary into a pandas ``DataFrame``
    with one column identifying the version and another holding the distance
    values.  If :mod:`seaborn` is installed, this DataFrame is passed to
    :func:`seaborn.boxplot` and the resulting figure is coloured using the
    provided ``colors`` mapping.  When :mod:`seaborn` is unavailable, a manual
    matplotlib implementation is used where each box is coloured individually.
    In both cases the figure is saved to ``outdir`` with a name derived from
    ``fid`` and ``fname``.
    """
    try:
        import seaborn as sns

        records = [pd.DataFrame({"Version": v, "Distanz": arr}) for v, arr in data.items()]
        if not records:
            return
        df = pd.concat(records, ignore_index=True)
        order = labels_order or list(df["Version"].unique())
        palette = {v: colors.get(v) for v in order}
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x="Version", y="Distanz", palette=palette, legend=False, order=order)
        plt.title(title_text or f"Boxplot – {fid}/{fname}")
        plt.xlabel("Version")
        plt.ylabel("M3C2 distance")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{fid}_{fname}_Boxplot.png"))
        plt.close()
    except Exception:
        labels = labels_order or list(data.keys())
        arrs = [data[v] for v in labels]
        plt.figure(figsize=(10, 6))
        b = plt.boxplot(arrs, labels=labels, patch_artist=True)
        for patch, v in zip(b["boxes"], labels):
            c = colors.get(v, "#aaaaaa")
            patch.set_facecolor(c)
            patch.set_alpha(0.5)
        plt.title(title_text or f"Boxplot – {fid}/{fname}")
        plt.xlabel("Version")
        plt.ylabel("M3C2 distance")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{fid}_{fname}_Boxplot.png"))
        plt.close()


def plot_overlay_qq(
    fid: str,
    fname: str,
    data: Dict[str, np.ndarray],
    colors: Dict[str, str],
    outdir: str,
    title_text: str | None = None,
    labels_order: List[str] | None = None,
) -> None:
    """Create an overlaid quantile-quantile (Q–Q) plot.

    Each dataset in ``data`` is compared against a theoretical normal
    distribution using :func:`scipy.stats.probplot`.  The function plots the
    ordered sample values (``osr``) against the theoretical quantiles (``osm``)
    and overlays a least-squares fit line defined by the returned ``slope`` and
    ``intercept``.  Deviations from this line indicate departures from
    normality.

    Parameters
    ----------
    fid: str
        Identifier of the file group used for labelling the output file.
    fname: str
        Name of the current file being processed.
    data: Dict[str, np.ndarray]
        Mapping from dataset label to the array of sample values to evaluate.
    colors: Dict[str, str]
        Colors to use for each dataset label.
    outdir: str
        Directory where the resulting plot image will be saved.
    title_text: str, optional
        Custom title for the plot; defaults to ``"Q-Q-Plot – {fid}/{fname}"``.
    labels_order: List[str], optional
        Explicit order in which datasets should be drawn.
    """
    plt.figure(figsize=(10, 6))
    labels = labels_order or list(data.keys())
    for v in labels:
        arr = data[v]
        (osm, osr), (slope, intercept, r) = probplot(arr, dist="norm")
        plt.plot(osm, osr, marker="o", linestyle="", label=v, color=colors.get(v))
        plt.plot(osm, slope * osm + intercept, color=colors.get(v), linestyle="--", alpha=0.7)
    plt.title(title_text or f"Q-Q-Plot – {fid}/{fname}")
    plt.xlabel("Theoretical quantiles")
    plt.ylabel("Ordered values")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"{fid}_{fname}_QQPlot.png"))
    plt.close()


def plot_overlay_violin(
    fid: str,
    fname: str,
    data: Dict[str, np.ndarray],
    colors: Dict[str, str],
    outdir: str,
    title_text: str | None = None,
    labels_order: List[str] | None = None,
) -> None:
    """Generate a violin plot for comparing M3C2 distance distributions.

    The provided arrays are combined into a single DataFrame and plotted using
    :mod:`seaborn`'s :func:`violinplot`. The resulting figure is written to
    ``outdir`` with a filename based on ``fid`` and ``fname``.
    """
    try:
        import seaborn as sns

        records = [pd.DataFrame({"Version": v, "Distanz": arr}) for v, arr in data.items()]
        if not records:
            return
        df = pd.concat(records, ignore_index=True)
        palette = {v: colors.get(v) for v in df["Version"].unique()}
        plt.figure(figsize=(10, 6))
        sns.violinplot(data=df, x="Version", y="Distanz", palette=palette, cut=0, inner="quartile")
        plt.title(title_text or f"Violinplot – {fid}/{fname}")
        plt.xlabel("Version")
        plt.ylabel("M3C2 distance")
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, f"{fid}_{fname}_Violinplot.png"))
        plt.close()
    except Exception as e:
        logger.warning("[Report] Violinplot fehlgeschlagen (%s/%s): %s", fid, fname, e)
