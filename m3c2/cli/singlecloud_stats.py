"""Compute statistics for a single point cloud pair using `StatisticsService`.

The parameters below define the moving and reference point clouds as well as
sampling and output options. The resulting metrics are written to an Excel
workbook or JSON file depending on ``output_format``.
"""
from m3c2.core.statistics import StatisticsService

# Paths and identifiers for the point cloud dataset to analyse.
folder_ids = ["data/TUNSPEKT Labordaten_all"]
filename_mov = "HandheldRoi"
filename_ref = "MavicRoi"
# Optional area of interest in square meters; ``None`` uses full extent.
area_m2 = None
# Parameters controlling the M3C2 calculation.
radius = 1.0
k = 6
sample_size = 100_000
use_convex_hull = True
# Output configuration.
out_path = "../outputs/TUNSPEKT_output/TUNSPEKT_m3c2_stats_clouds.xlsx"
sheet_name = "CloudStats"
output_format = "excel"

# Compute statistics for the specified clouds and persist the results.
StatisticsService.calc_single_cloud_stats(
    folder_ids=folder_ids,
    filename_mov=filename_mov,
    filename_ref=filename_ref,
    area_m2=area_m2,
    radius=radius,
    k=k,
    sample_size=sample_size,
    use_convex_hull=use_convex_hull,
    out_path=out_path,
    sheet_name=sheet_name,
    output_format=output_format,
)

