from services.plot_service import PlotService, PlotConfig, PlotOptions

# Mehrere Folder: jeder Folder wird eine Seite im PDF
folder_ids = ["0342-0349", "0817-0821", "0910-0913", "1130-1133", "1203-1206", "1306-1311"]

# Vier Kurven insgesamt: je Version x je filename -> python_ref, python_ref_ai, CC_ref, CC_ref_ai
versions = ["python"]
filenames = ["ref", "ref_ai"]  

cfg = PlotConfig(
    folder_ids=folder_ids,
    filenames=filenames,
    versions=versions,
    bins=256,
    outdir="outputs",
    project="MARS"
)

opts = PlotOptions(
    plot_hist=False, plot_gauss=False, plot_weibull=False,
    plot_box=False, plot_qq=False, plot_grouped_bar=False, plot_violin=True
)

# erzeugt ALLE Plots für alle filenames; Python & CC werden je filename überlagert
PlotService.overlay_plots(cfg, opts)

# optional: ein PDF pro filename erzeugen
PlotService.summary_pdf(cfg)
