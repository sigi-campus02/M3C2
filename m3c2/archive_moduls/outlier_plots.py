"""Visualize distance distributions including various outlier filtering methods.

The script reads distance files for different processing variants and produces
boxplots comparing all distances with several inlier-selection techniques."""
import logging
import os
import numpy as np
import matplotlib.pyplot as plt

from m3c2.io.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def main(
    base: str = "data/0342-0349/",
    variants: list[tuple[str, str]] | None = None,
    inlier_suffixes: list[tuple[str, str]] | None = None,
    outdir: str | None = None,
) -> None:
    """Create boxplots comparing full and inlier-filtered distances."""

    # Logging configuration relies on environment variables or configuration
    # files; no explicit level argument is necessary.
    setup_logging()

    variants = variants or [
        ("ref", "python_ref_m3c2_distances.txt"),
        ("ref_ai", "python_ref_ai_m3c2_distances.txt"),
    ]
    inlier_suffixes = inlier_suffixes or [
        ("std", "Inlier _STD"),
        ("rmse", "Inlier _RMSE"),
        ("nmad", "Inlier _NMAD"),
        ("iqr", "Inlier _IQR"),
    ]
    outdir = outdir or os.path.join("outputs", "MARS_output", "Plots_MARS_Outlier")

    for variant, dist_file in variants:
        logger.info("Starting variant %s", variant)
        data_list: list[np.ndarray] = []
        labels: list[str] = []

        file_path = os.path.join(base, dist_file)
        try:
            data = np.loadtxt(file_path)
            if data.ndim == 0 or data.size == 0:
                logger.warning("File %s is empty", file_path)
                data = np.array([])
            else:
                data = data[~np.isnan(data)]
                if data.size > 0:
                    data_list.append(data)
                    labels.append("All Distances")
                else:
                    logger.warning("File %s has no valid data", file_path)
        except Exception:
            logger.warning("File %s is missing or unreadable", file_path)

        for suffix, label in inlier_suffixes:
            file_path = os.path.join(
                base, f"python_{variant}_m3c2_distances_coordinates_inlier_{suffix}.txt"
            )
            try:
                arr = np.loadtxt(file_path, skiprows=1)
                data = arr[:, -1]
                data = data[~np.isnan(data)]
                if data.size > 0:
                    data_list.append(data)
                    labels.append(label)
                else:
                    logger.warning("File %s is empty", file_path)
            except Exception:
                logger.warning("File %s is missing or unreadable", file_path)

        plt.figure(figsize=(8, 6))
        plt.boxplot(data_list, labels=labels)
        plt.ylabel("Distanz")
        basename = base.strip("/").split("/")[-1]
        plt.title(f"Vergleich der Distanzverteilungen ({basename}) {variant}")
        plt.grid(True)
        plt.tight_layout()

        os.makedirs(outdir, exist_ok=True)
        plt.savefig(os.path.join(outdir, f"{basename}_OutlierComparison_{variant}.png"))
        plt.close()
        logger.info("Completed variant %s", variant)


if __name__ == "__main__":
    main()

