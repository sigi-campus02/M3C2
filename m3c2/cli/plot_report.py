# generation/generate_plots.py
"""Create summary PDFs from pre-generated plot components.

The script compiles individual plot images produced for the Multi-Illumination
project into two consolidated PDF reports—one including all outliers and one
using only inlier data. The heavy lifting is delegated to :class:`PlotService`.
"""
import os, sys, logging

logger = logging.getLogger(__name__)

# Determine repository root so that package imports work when executed directly.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from m3c2.visualization.services.plot_service import PlotService
from m3c2.config.logging_config import setup_logging
from m3c2.config.plot_config import PlotOptions

# Input and output directories for the TUNSPEKT Labordaten_all dataset.
DATA_DIR = os.path.join(ROOT, "data", "TUNSPEKT Labordaten_all")
OUT_DIR = os.path.join(ROOT, "outputs", "TUNSPEKT Labordaten_all", "plots")


def main(data_dir: str = DATA_DIR, out_dir: str = OUT_DIR) -> tuple[str, str]:
    """Generate summary PDF reports for already generated plots."""

    # Configure logging using defaults from configuration/environment.
    setup_logging()
    logger.info("Generating summary PDF reports from %s to %s", data_dir, out_dir)

    # Example configuration for generating additional grouped plots.
    # only_grouped = PlotOptions(
    #     plot_hist=True, plot_gauss=True, plot_weibull=True,
    #     plot_box=True, plot_qq=True, plot_grouped_bar=True, plot_violin=False,
    # )

    # Example call to generate overlay plots for each index. Commented out to
    # prevent accidental regeneration of existing PNGs.
    # PlotService.overlay_by_index(
    #     data_dir, out_dir,
    #     versions=("python",),
    #     bins=256,
    #     options=only_grouped,
    #     skip_existing=True,
    # )

    try:
        pdf_incl = PlotService.build_parts_pdf(
            out_dir,
            pdf_path=os.path.join(out_dir, "parts_summary_incl_outliers.pdf"),
            include_with=True,
            include_inlier=False,
        )
        pdf_excl = PlotService.build_parts_pdf(
            out_dir,
            pdf_path=os.path.join(out_dir, "parts_summary_excl_outliers.pdf"),
            include_with=False,
            include_inlier=True,
        )

        logger.info("PDF (incl. outliers): %s", pdf_incl)
        logger.info("PDF (excl. outliers): %s", pdf_excl)
        return pdf_incl, pdf_excl
    except (OSError, RuntimeError) as exc:
        logger.error("Failed to generate summary PDFs: %s", exc, exc_info=True)
    except Exception as exc:
        logger.exception(
            "Unexpected error during summary PDF generation: %s", exc
        )
        raise


if __name__ == "__main__":
    main()

