import os
from typing import List, Optional
from statistics_service import StatisticsService
import logging
# logging konfigurieren
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

# Falls du die Kommentar-Funktion aus meiner Antwort übernommen hast:
# from stats_cloud_headers import add_cloud_header_comments

folder_id = os.path.join("data", "rocks")
filenames = ["points_40", "points_100", "points_80", "points_overlap2", "points_zshift"]

# Welche Dateiendungen sollen gesucht werden?
EXTS = [".xyz", ".ply", ".las", ".laz", ".txt", ".csv"]

def find_existing_file(base_dir: str, stem: str, exts: List[str]) -> Optional[str]:
    for ext in exts:
        p = os.path.join(base_dir, stem + ext)
        logging.info(f"[cloud-stats] Suche nach Datei: {p}")
        if os.path.exists(p):
            return p
    return None

rows = []
for stem in filenames:
    path = find_existing_file(folder_id, stem, EXTS)
    if not path:
        print(f"[cloud-stats] Datei für '{stem}' nicht gefunden in {folder_id}")
        continue
    logging.info(f"[cloud-stats] Verarbeite Datei: {path}")
    stats = StatisticsService.cloud_stats_from_file(
        path,
        role="mov",          # wird bei Dateien ignoriert (nur bei Ordnern relevant)
        area_m2=None,        # None → Fläche automatisch (Convex Hull/Bounding-Box)
        radius=0.5,          # Nachbarschaftsradius (m) für lokale Dichte/Rauigkeit
        k=6,                 # k-NN für mittlere Abstände
        sample_size=100_000, # Subsample für lokale Metriken
        use_convex_hull=True # bessere Flächenabschätzung als BBox (falls verfügbar)
    )
    rows.append(stats)

# Excel schreiben/appen
StatisticsService.write_cloud_stats(
    rows,
    out_xlsx="m3c2_stats_clouds.xlsx",
    sheet_name="CloudStats"
)

# (Optional) Header-Kommentare anhängen
# add_cloud_header_comments("cloud_stats.xlsx", sheet_name="CloudStats")
print("Fertig: cloud_stats.xlsx → Sheet 'CloudStats'")
