# example_report.py
from plot_service import PlotService, PlotConfig, PlotOptions
import os

folder_id = ["rocks"]
filenames = ["points_40", "points_100", "points_80", "points_overlap2", "points_zshift"]

cfg = PlotConfig(folder=folder_id, filenames=filenames, bins=256)
opts = PlotOptions(plot_hist=True, plot_gauss=True, plot_weibull=True,
                   plot_box=True, plot_qq=True, plot_grouped_bar=True, outdir="Plots")

for f in filenames:
    cfg = PlotConfig(folder=folder_id, filenames=[f], bins=256)
    PlotService.overlay_plots("rocks", cfg, opts)

PlotService.summary_pdf(filenames, pdf_name="Plot_Vergleich.pdf", outdir="Plots")
