"""Configuration objects for plot generation utilities."""

from __future__ import annotations

import os.path
from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass(frozen=True)
class PlotOptions:
    """Flags controlling which plot types are generated.

    Attributes
    ----------
    plot_hist:
        Generate histogram plots of the distance distributions.
    plot_gauss:
        Overlay Gaussian fit curves on the histogram.
    plot_weibull:
        Overlay Weibull fit curves on the histogram.
    plot_box:
        Produce box plots of the distances.
    plot_qq:
        Generate quantile-quantile plots.
    plot_grouped_bar:
        Create grouped bar plots showing mean and standard deviation.
    plot_violin:
        Produce violin plots of the distance distributions.
    """

    plot_hist: bool = True
    plot_gauss: bool = True
    plot_weibull: bool = True
    plot_box: bool = True
    plot_qq: bool = True
    plot_grouped_bar: bool = True
    plot_violin: bool = True


@dataclass(frozen=True)
class PlotOptionsComparedistances:
    """Flags for plots used when comparing distance distributions."""

    plot_blandaltman: bool = True
    plot_passingbablok: bool = True
    plot_linearregression: bool = True


@dataclass
class PlotConfig:
    """Comprehensive configuration for generating result plots.

    Attributes
    ----------
    folder_ids:
        List of folder identifiers that contain the distance files.
    filenames:
        Basenames of the distance files to plot.
    project:
        Name of the project used for output directories.
    outdir:
        Base directory for generated plots and artifacts.
    versions:
        Optional list of version prefixes to combine with ``filenames``.
    bins:
        Number of histogram bins.
    colors:
        Optional mapping from label to color in hex representation.
    path:
        Full output path constructed during post initialisation.
    """

    folder_ids: List[str]
    filenames: List[str]
    project: str
    outdir: str
    versions: Optional[List[str]] = None
    bins: int = 256
    colors: Dict[str, str] = field(default_factory=dict)
    path: str = field(init=False)

    def __post_init__(self) -> None:
        """Create the target directory path for generated plots."""

        self.path = os.path.join(
            self.outdir, f"{self.project}_output", f"{self.project}_plots"
        )

    def labels(self) -> List[str]:
        """Return combined labels from versions and filenames."""

        # Maintain order such that plots are comparable across runs
        return [f"{v}_{f}" for v in self.versions for f in self.filenames]

    def ensure_colors(self) -> Dict[str, str]:
        """Return a mapping of labels to colors, generating defaults if needed."""

        if self.colors:
            return dict(self.colors)
        default_palette = [
            "#1f77b4",
            "#ff7f0e",
            "#2ca02c",
            "#d62728",
            "#9467bd",
            "#8c564b",
            "#e377c2",
            "#7f7f7f",
            "#bcbd22",
            "#17becf",
        ]
        lbls = self.labels()
        return {lbls[i]: default_palette[i % len(default_palette)] for i in range(len(lbls))}
