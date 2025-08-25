from statistics_service_comparedistances import StatisticsCompareDistances
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)
import numpy as np

# TUNSPEKT FOLDERS: "TUNSPEKT_Altone(mov)-Faro(ref)", "TUNSPEKT_Handheld(mov)-Faro(ref)", "TUNSPEKT_Mavic(mov)-Faro(ref)"
# MARS FOLDERS:     "0342-0349", "0817-0821", "0910-0913", "1130-1133", "1203-1206", "1306-1311"
# MARS REF VARIANTS: "ref", "ref_ai"

# ref_variants = ["altone-faro", "handheld-faro", "mavic-faro"]

# folder_ids = ["TUNSPEKT Labordaten_all"]
# ref_variants = ["altone-faro", "handheld-faro"]
# outdir = "Plots_TUNSPEKT_PassingBablok"

folder_ids = ["0342-0349", "0817-0821", "0910-0913", "1130-1133", "1203-1206", "1306-1311"]
ref_variants = ["ref", "ref_ai"]
outdir = "Plots_MARS_PassingBablok"

# StatisticsCompareDistances.bland_altman_plot(
#     folder_ids=folder_ids,
#     ref_variants=ref_variants,
#     outdir=outdir,
# )

# StatisticsCompareDistances.passing_bablok_plot(
#     folder_ids=folder_ids,
#     ref_variants=ref_variants,
#     outdir=outdir,
# )

# StatisticsCompareDistances.linear_regression_plot(
#     folder_ids=folder_ids,
#     ref_variants=ref_variants,
#     outdir=outdir,
# )