import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from services.plot_service import PlotConfig, PlotOptions, PlotService


# Mehrere Folder: jeder Folder wird eine Seite im PDF
folder_ids = ["1-1", "1-2", "1-3", "1-4", "1-5", "1-6", "1-7"]

# Vier Kurven insgesamt: je Version x je filename -> python_ref, python_ref_ai, CC_ref, CC_ref_ai
versions = ["python"]
filenames = ["Job_0378_8400-110-rad-1-1-AI_cloud"]

cfg = PlotConfig(
    folder_ids=folder_ids,
    filenames=filenames,
    versions=versions,
    bins=256,
    outdir="outputs",
    project="MARS_Multi_Illumination"
)

opts = PlotOptions(
    plot_hist=True, plot_gauss=True, plot_weibull=True,
    plot_box=True, plot_qq=True, plot_grouped_bar=True, plot_violin=True
)

# erzeugt ALLE Plots für alle filenames; Python & CC werden je filename überlagert
PlotService.overlay_plots(cfg, opts)

# optional: ein PDF pro filename erzeugen
PlotService.summary_pdf(cfg)