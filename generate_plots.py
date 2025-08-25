# example_report.py
from plot_service import PlotService, PlotConfig, PlotOptions

# TUNSPEKT FOLDERS: "TUNSPEKT_Altone(mov)-Faro(ref)", "TUNSPEKT_Handheld(mov)-Faro(ref)", "TUNSPEKT_Mavic(mov)-Faro(ref)"
# MARS FOLDERS:     "0342-0349", "0817-0821", "0910-0913", "1130-1133", "1203-1206", "1306-1311"
# MARS REF VARIANTS: "ref", "ref_ai"
# TUNSPEKT FOLDERS: "TUNSPEKT_Altone(mov)-Faro(ref)", "TUNSPEKT_Handheld(mov)-Faro(ref)", "TUNSPEKT_Mavic(mov)-Faro(ref)"
# MARS FOLDERS:     "0342-0349", "0817-0821", "0910-0913", "1130-1133", "1203-1206", "1306-1311"
# MARS REF VARIANTS: "ref", "ref_ai"

# Mehrere Folder: jeder Folder wird eine Seite im PDF
folder_ids = ["TUNSPEKT Labordaten_all"] 

# Vier Kurven insgesamt: je Version x je filename -> python_ref, python_ref_ai, CC_ref, CC_ref_ai
versions = ["python"]
filenames = ["altone-faro", "handheld-faro", "mavic-faro"]  


cfg = PlotConfig(
    folder_ids=folder_ids,
    filenames=filenames,
    versions=versions,
    bins=256,
    outdir="Plots_TUNSPEKT", # "Plots_TUNSPEKT", "Plots_Mars"
)

opts = PlotOptions(
    plot_hist=True, plot_gauss=True, plot_weibull=True,
    plot_box=True, plot_qq=True, plot_grouped_bar=True
)

# erzeugt ALLE Plots für alle filenames; Python & CC werden je filename überlagert
PlotService.overlay_plots(cfg, opts)

# optional: ein PDF pro filename erzeugen
PlotService.summary_pdf(cfg, pdf_name="TUNSPEKT_Plot_Vergleich.pdf") #"TUNSPEKT_Plot_Vergleich.pdf", "MARS_Plot_Vergleich.pdf"
