from services.plot_service_comparedistances import PlotServiceCompareDistances, PlotConfig, PlotOptionsComparedistances
import logging

logger = logging.getLogger(__name__)

# folder_ids = ["0342-0349", "0817-0821", "0910-0913", "1130-1133", "1203-1206", "1306-1311"]
# ref_variants = ["ref", "ref_ai"]

folder_ids = ["rocks"]
ref_variants = ["points_40", "points_80"]

cfg = PlotConfig(
    folder_ids=folder_ids,
    filenames=ref_variants,
    bins=256,
    outdir="../outputs",
    project="ROCKS"
)

opts = PlotOptionsComparedistances(
    plot_blandaltman=True,
    plot_passingbablok=True,
    plot_linearregression=True
)

PlotServiceCompareDistances.overlay_plots(cfg, opts)



