"""Create comparison plots between reference variants of point cloud data.

The script configures :class:`PlotServiceCompareDistances` to overlay various
statistical plots (Bland–Altman, Passing–Bablok, and linear regression) for the
specified folders and reference variants.
"""
import logging
logger = logging.getLogger(__name__)

import sys
import os

# Allow absolute imports when the script is executed directly.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from m3c2.config.plot_config import PlotConfig, PlotOptionsComparedistances
from m3c2.visualization.plot_comparedistances_service import (
    PlotServiceCompareDistances,
)

# Select the folders and reference data variants to compare.
# folder_ids = ["0342-0349", "0817-0821", "0910-0913", "1130-1133", "1203-1206", "1306-1311"]
# ref_variants = ["ref", "ref_ai"]

folder_ids = ["0342-0349"]
ref_variants = ["ref", "ref_ai"]

# Configuration object describing input files and output location.
cfg = PlotConfig(
    folder_ids=folder_ids,
    filenames=ref_variants,
    bins=256,
    outdir="outputs",
    project="MARS",
)

# Options specifying which statistical comparison plots to create.
opts = PlotOptionsComparedistances(
    plot_blandaltman=True,
    plot_passingbablok=True,
    plot_linearregression=True,
)

logging.info(f"Starting plot generation {cfg}, {opts}")
# Generate the overlay plots according to the configuration and options.
PlotServiceCompareDistances.overlay_plots(cfg, opts)

