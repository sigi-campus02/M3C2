"""Generate overlay plots and bundle them into a PDF report.

This command line utility compares the distributions from multiple distance
files by creating a set of overlay plots. The resulting images are fed into
the existing :mod:`report_builder` facilities via
:class:`~m3c2.visualization.services.plot_service.PlotService` to assemble a
consolidated PDF report.

Example
-------
    >>> from m3c2.cli.overlay_report import main
    >>> main(["distances_run1.txt", "distances_run2.txt"])

The function above will create PNG plots in the directory ``overlay_report``
and a combined PDF called ``overlay_report/report.pdf``.
"""

from __future__ import annotations

import argparse
import logging
import os
import json
from pathlib import Path
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
    os.makedirs(outdir, exist_ok=True)
    overlays = PlotService.overlay_from_data(data, outdir)
    return overlays


# ---------------------------------------------------------------------------
# Main workflow
# ---------------------------------------------------------------------------

def main(
    overlay_files: List[str],
    overlay_outdir: str | None = None,
    *,
    outdir: str | None = None,
) -> str:
    """Generate overlay plots for ``overlay_files`` and return the PDF path.

    Parameters
    ----------
    overlay_files:
        List of distance files that will be compared.
    overlay_outdir, outdir:
        Target directory for generated plots. ``outdir`` is supported for
        backwards compatibility; if both parameters are supplied a
        :class:`TypeError` is raised.
    """

    if overlay_outdir and outdir:
        raise TypeError("Specify only one of overlay_outdir or outdir")
    overlay_outdir = overlay_outdir or outdir or "overlay_report"

    if len(overlay_files) < 2:
        raise ValueError("At least two distance files are required")

    data: Dict[str, np.ndarray] = {}
    for f in overlay_files:
        label = os.path.splitext(os.path.basename(f))[0]
        data[label] = load_distance_file(f)

    generate_overlay_plots(data, overlay_outdir)
    pdf = PlotService.build_parts_pdf(
        overlay_outdir,
        pdf_path=os.path.join(overlay_outdir, "report.pdf"),
        include_with=True,
        include_inlier=False,
    )
    if not pdf:
        raise FileNotFoundError("No plot images found to build the report")
    return pdf


def build_arg_parser(config_path: str | Path | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create overlay plot report for multiple distance files"
    )
    parser.add_argument(
        "--overlay_files",
        type=str,
        nargs="+",
        help="List of distance files to process.",
    )
    parser.add_argument(
        "--overlay_outdir",
        type=str,
        help="Directory for output plots and reports.",
    )

    # Load defaults from config file if available
    cfg_path = (
        Path(config_path)
        if config_path is not None
        else Path(__file__).resolve().parents[2] / "config.json"
    )
    if cfg_path.exists():
        try:
            with cfg_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle).get("arguments_plot_specific_files", {})
                logger.debug(f"Loaded config from {cfg_path}")
                logger.debug(f"Config data: {data}")
        except (OSError, json.JSONDecodeError):
            data = {}
            logger.debug(f"Could not load config from {cfg_path}")
        files_default = data.get("overlay_files")
        outdir_default = data.get("overlay_outdir")
        if files_default:
            parser.set_defaults(overlay_files=files_default)
        if outdir_default:
            parser.set_defaults(overlay_outdir=outdir_default)

    return parser

def build_arg_parser_onefolder(config_path: str | Path | None = None) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create overlay plot report for multiple distance files"
    )
    parser.add_argument(
        "--folder",
        type=str,
        help="Folder containing distance files to process.",
    )
    parser.add_argument(
        "--overlay_outdir",
        type=str,
        help="Directory for output plots and reports.",
    )
    parser.add_argument(
        "--options",
        type=str,
        nargs="+",
        help="List of plot types to generate.",
    )

    # Load defaults from config file if available
    cfg_path = (
        Path(config_path)
        if config_path is not None
        else Path(__file__).resolve().parents[2] / "config.json"
    )
    if cfg_path.exists():
        try:
            with cfg_path.open("r", encoding="utf-8") as handle:
                data = json.load(handle).get("arguments_plot_files_in_folder", {})
                logger.debug(f"Loaded config from {cfg_path}")
                logger.debug(f"Config data: {data}")
        except (OSError, json.JSONDecodeError):
            data = {}
            logger.debug(f"Could not load config from {cfg_path}")
        folder_default = data.get("folder")
        outdir_default = data.get("overlay_outdir")
        options_default = data.get("options")
        if options_default:
            parser.set_defaults(options=options_default)
        if folder_default:
            parser.set_defaults(folder=folder_default)
        if outdir_default:
            parser.set_defaults(overlay_outdir=outdir_default)

    return parser


if __name__ == "__main__":
    parser = build_arg_parser()
    args = parser.parse_args()
    main(args.folder, args.overlay_outdir, args.options)
