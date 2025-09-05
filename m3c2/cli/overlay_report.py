"""Generate overlay plots and bundle them into a PDF report.

This command line utility compares the distributions from two distance files
by creating a set of overlay plots.  The resulting images are fed into the
existing :mod:`report_builder` facilities via :class:`~m3c2.visualization.services.plot_service.PlotService`
to assemble a consolidated PDF report.

Example
-------
    >>> from m3c2.cli.overlay_report import main
    >>> main("distances_run1.txt", "distances_run2.txt")

The function above will create PNG plots in the directory ``overlay_report``
and a combined PDF called ``overlay_report/report.pdf``.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Dict, List
import numpy as np

from m3c2.visualization.loaders.distance_loader import load_1col_distances
from m3c2.visualization.services.plot_service import PlotService

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_distance_file(path: str) -> np.ndarray:
    """Return numeric distances contained in *path*.

    Parameters
    ----------
    path:
        Path to a text or CSV file containing one column of numeric values.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If the file cannot be parsed as numeric data.
    """

    if not os.path.exists(path):
        raise FileNotFoundError(f"Distance file not found: {path}")

    ext = os.path.splitext(path)[1].lower()
    try:
        if ext == ".csv":
            data = np.genfromtxt(path, delimiter=",")
        else:
            data = load_1col_distances(path)
    except Exception as exc:  # pragma: no cover - depends on numpy internals
        raise ValueError(f"Could not read numeric data from {path}") from exc

    data = np.asarray(data, dtype=float)
    data = data[np.isfinite(data)]
    return data


# ---------------------------------------------------------------------------
# Plot generation
# ---------------------------------------------------------------------------

def generate_overlay_plots(data: Dict[str, np.ndarray], outdir: str) -> List[str]:
    """Delegate plot creation to :func:`overlay_from_data`.

    Keeping this function allows tests to monkeypatch the plotting routine
    while the implementation is provided by the service layer.
    """

    return overlay_from_data(data, outdir)


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def main(file_a: str, file_b: str, outdir: str = "overlay_report") -> str:
    """Generate overlay plots for two files and return the PDF path."""

    data: Dict[str, np.ndarray] = {}
    for f in (file_a, file_b):
        label = os.path.splitext(os.path.basename(f))[0]
        data[label] = load_distance_file(f)

    generate_overlay_plots(data, outdir)
    pdf = PlotService.build_parts_pdf(
        outdir,
        pdf_path=os.path.join(outdir, "report.pdf"),
        include_with=True,
        include_inlier=False,
    )
    if not pdf:
        raise FileNotFoundError("No plot images found to build the report")
    return pdf


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create overlay plot report for two distance files"
    )
    parser.add_argument("file_a", help="First distance file (.txt or .csv)")
    parser.add_argument("file_b", help="Second distance file (.txt or .csv)")
    parser.add_argument(
        "--outdir", default="overlay_report", help="Directory for plots and PDF"
    )
    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    main(args.file_a, args.file_b, args.outdir)
