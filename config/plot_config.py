from __future__ import annotations

import os.path
from dataclasses import dataclass, field
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class PlotOptions:
    plot_hist: bool = True
    plot_gauss: bool = True
    plot_weibull: bool = True
    plot_box: bool = True
    plot_qq: bool = True
    plot_grouped_bar: bool = True
    plot_violin: bool = True


@dataclass(frozen=True)
class PlotOptionsComparedistances:
    plot_blandaltman: bool = True
    plot_passingbablok: bool = True
    plot_linearregression: bool = True


@dataclass
class PlotConfig:
    folder_ids: List[str]
    filenames: List[str]
    versions: List[str]
    project: str
    outdir: str
    path: os.path.join(outdir, f"{project}_output", f"{project}_plots")
    bins: int = 256
    colors: Dict[str, str] = field(default_factory=dict)

    def labels(self) -> List[str]:
        # Reihenfolge der vier Kurven, z.B. ["python_ref","python_ref_ai","CC_ref","CC_ref_ai"]
        return [f"{v}_{f}" for v in self.versions for f in self.filenames]

    def ensure_colors(self) -> Dict[str, str]:
        if self.colors:
            return dict(self.colors)
        default_palette = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
            "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
        ]
        lbls = self.labels()
        return {lbls[i]: default_palette[i % len(default_palette)] for i in range(len(lbls))}