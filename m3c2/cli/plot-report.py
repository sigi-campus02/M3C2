# generation/generate_plots.py
"""Create summary PDFs from pre-generated plot components.

The script compiles individual plot images produced for the Multi-Illumination
project into two consolidated PDF reports—one including all outliers and one
using only inlier data. The heavy lifting is delegated to :class:`PlotService`.
"""
import os, sys, logging

# Determine repository root so that package imports work when executed directly.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from m3c2.visualization.plot_service import PlotService
from m3c2.io.logging_utils import setup_logging
from m3c2.config.plot_config import PlotOptions

# Input and output directories for the Multi-Illumination dataset.
DATA_DIR = os.path.join(ROOT, "data", "Multi-Illumination")
OUT_DIR  = os.path.join(ROOT, "outputs", "MARS_Multi_Illumination", "plots")

if __name__ == "__main__":
    setup_logging()

    # Example configuration for generating additional grouped plots.
    # only_grouped = PlotOptions(
    #     plot_hist=True, plot_gauss=True, plot_weibull=True,
    #     plot_box=True, plot_qq=True, plot_grouped_bar=True, plot_violin=False,
    # )

    # Example call to generate overlay plots for each index. Commented out to
    # prevent accidental regeneration of existing PNGs.
    # PlotService.overlay_by_index(
    #     DATA_DIR, OUT_DIR,
    #     versions=("python",),
    #     bins=256,
    #     options=only_grouped,
    #     skip_existing=True,   # existing PNGs are not overwritten
    # )

    # Build PDF reports that summarize the parts with and without outliers.
    pdf_incl = PlotService.build_parts_pdf(
        OUT_DIR,
        pdf_path=os.path.join(OUT_DIR, "parts_summary_incl_outliers.pdf"),
        include_with=True,
        include_inlier=False,
    )
    pdf_excl = PlotService.build_parts_pdf(
        OUT_DIR,
        pdf_path=os.path.join(OUT_DIR, "parts_summary_excl_outliers.pdf"),
        include_with=False,
        include_inlier=True,
    )

    print(f"PDF (incl. outliers): {pdf_incl}")
    print(f"PDF (excl. outliers): {pdf_excl}")

