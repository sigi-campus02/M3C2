# example_report.py
from plot_service import PlotService, PlotConfig, PlotOptions

folder_id = "rocks"  # ein Ordnername, keine Liste
versions = ["python", "CC"]
filenames = ["points_40", "points_80", "points_overlap2", "points_zshift"]

cfg = PlotConfig(
    folder_id=folder_id,
    filenames=filenames,
    versions=versions,
    bins=256,
    outdir="Plots",
)

opts = PlotOptions(
    plot_hist=True, plot_gauss=True, plot_weibull=True,
    plot_box=True, plot_qq=True, plot_grouped_bar=True
)

# erzeugt ALLE Plots für alle filenames; Python & CC werden je filename überlagert
PlotService.overlay_plots(folder_id, cfg, opts)

# optional: ein PDF pro filename erzeugen
PlotService.summary_pdf(folder_id, filenames, pdf_name="Plot_Vergleich.pdf", outdir="Plots")
