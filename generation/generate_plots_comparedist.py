from services.plot_service_comparedistances import PlotServiceCompareDistances
import logging

logger = logging.getLogger(__name__)

folder_ids = ["0342-0349", "0817-0821", "0910-0913", "1130-1133", "1203-1206", "1306-1311"]
ref_variants = ["ref", "ref_ai"]
outdir = "Plots_TUNSPEKT_BlandAltman"

PlotServiceCompareDistances.bland_altman_plot(
    folder_ids=folder_ids,
    ref_variants=ref_variants,
    outdir=outdir,
)

# PlotServiceCompareDistances.passing_bablok_plot(
#     folder_ids=folder_ids,
#     ref_variants=ref_variants,
#     outdir=outdir,
# )

# PlotServiceCompareDistances.linear_regression_plot(
#     folder_ids=folder_ids,
#     ref_variants=ref_variants,
#     outdir=outdir,
# )