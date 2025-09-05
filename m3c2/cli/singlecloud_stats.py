"""Compute statistics for a single point cloud pair."""

import logging

from m3c2.statistics.single_cloud_service import calc_single_cloud_stats
from m3c2.config.logging_config import setup_logging


logger = logging.getLogger(__name__)


def main(
    folder_ids: list[str] | None = None,
    filename_comparison: str = "HandheldRoi",
    filename_reference: str = "MavicRoi",
    area_m2: float | None = None,
    radius: float = 1.0,
    k: int = 6,
    sample_size: int | None = 100_000,
    use_convex_hull: bool = True,
    out_path: str = "../outputs/TUNSPEKT_output/TUNSPEKT_m3c2_stats_clouds.xlsx",
    sheet_name: str = "CloudStats",
    output_format: str = "excel",
) -> None:
    """Invoke :func:`m3c2.statistics.single_cloud_service.calc_single_cloud_stats` with defaults."""

    # ``setup_logging`` determines the log level internally.
    setup_logging()
    logger.info(
        "Parameters received: folder_ids=%s, filename_comparison=%s, filename_reference=%s, "
        "area_m2=%s, radius=%s, k=%s, sample_size=%s, use_convex_hull=%s, "
        "out_path=%s, sheet_name=%s, output_format=%s",
        folder_ids,
        filename_comparison,
        filename_reference,
        area_m2,
        radius,
        k,
        sample_size,
        use_convex_hull,
        out_path,
        sheet_name,
        output_format,
    )

    folder_ids = folder_ids or ["data/TUNSPEKT Labordaten_all"]
    logger.info("Processing folders: %s", folder_ids)

    try:
        calc_single_cloud_stats(
            folder_ids=folder_ids,
            filename_singlecloud=filename_comparison,
            area_m2=area_m2,
            radius=radius,
            k=k,
            sample_size=sample_size,
            use_convex_hull=use_convex_hull,
            out_path=out_path,
            sheet_name=sheet_name,
            output_format=output_format,
        )
    except (OSError, ValueError):
        logger.exception("Statistics computation failed")
    except Exception:
        logger.exception("Unexpected error during statistics computation")
        raise
    else:
        logger.info("Statistics computation completed successfully")


if __name__ == "__main__":
    main()

