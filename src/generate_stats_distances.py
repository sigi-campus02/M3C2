from statistics_service import StatisticsService
import logging

logging.basicConfig(level=logging.INFO)

folder_ids = ["0342-0349", "0817-0821", "0910-0913", "1130-1133", "1203-1206", "1306-1311"]
versions = [f"08-11-v{i}" for i in range(1, 6)]

for v in versions:
    logging.info("Berechne Statistiken für Version: %s", v)
    StatisticsService.compute_m3c2_statistics(
        folder_ids=folder_ids,
        version=v,
        out_xlsx="m3c2_stats_all.xlsx",
        sheet_name="Results",
    )