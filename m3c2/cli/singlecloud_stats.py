"""Compute statistics for a single point cloud pair using ``StatisticsService``."""

from m3c2.core.statistics import StatisticsService


def main(
    folder_ids: list[str] | None = None,
    filename_mov: str = "HandheldRoi",
    filename_ref: str = "MavicRoi",
    area_m2: float | None = None,
    radius: float = 1.0,
    k: int = 6,
    sample_size: int | None = 100_000,
    use_convex_hull: bool = True,
    out_path: str = "../outputs/TUNSPEKT_output/TUNSPEKT_m3c2_stats_clouds.xlsx",
    sheet_name: str = "CloudStats",
    output_format: str = "excel",
) -> None:
    """Invoke :func:`StatisticsService.calc_single_cloud_stats` with defaults."""

    folder_ids = folder_ids or ["data/TUNSPEKT Labordaten_all"]

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


if __name__ == "__main__":
    main()

