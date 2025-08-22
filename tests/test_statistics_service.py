import sys
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from statistics_service import StatisticsService


def test_calc_stats_inlier_outlier_split():
    distances = np.concatenate([np.zeros(100), np.array([100.0])])
    stats = StatisticsService.calc_stats(distances)

    assert stats["Outlier Count"] == 1
    assert stats["Inlier Count"] == 100
    assert stats["Mean Inlier"] == 0.0
    assert stats["Max Inlier"] == 0.0
    assert stats["Mean Outlier"] == 100.0
    assert stats["Valid Count Inlier"] == 100


def test_append_df_to_excel_preserves_formatting(tmp_path):
    import pandas as pd
    from openpyxl import Workbook, load_workbook

    # Ausgangs-Workbook mit formatiertem Header anlegen
    xlsx = tmp_path / "test.xlsx"
    wb = Workbook()
    ws = wb.active
    ws.title = "Results"
    ws.append(["Timestamp", "Total Points"])
    ws["A1"].font = ws["A1"].font.copy(bold=True)
    wb.save(xlsx)

    # Neuen Datensatz anhängen
    df_new = pd.DataFrame({"Total Points": [1]})
    StatisticsService._append_df_to_excel(df_new, str(xlsx))

    # Überprüfen, dass Formatierung erhalten bleibt und Zeile angehängt wurde
    wb2 = load_workbook(xlsx)
    ws2 = wb2["Results"]
    assert ws2["A1"].font.bold is True
    assert ws2.max_row == 2
    assert ws2["B2"].value == 1
