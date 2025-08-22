from statistics_service_comparedistances import StatisticsCompareDistances
import logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# "0342-0349", "0817-0821", "0910-0913", "1130-1133", "1203-1206", "1306-1311"

folder_ids = ["0342-0349"]
ref_variants = ["ref", "ref_ai"]
outdir = "Plots_PassingBablok"


try:    
    StatisticsCompareDistances.passing_bablok_plot(
        folder_ids=folder_ids,
        ref_variants=ref_variants,
        outdir=outdir,
    )
except Exception as e:
    logger.error(f"Error processing folder {folder_ids}: {e}")



