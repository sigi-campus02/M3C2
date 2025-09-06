"""Utilities to persist matplotlib figures as a PDF document.

The :func:`write` function accepts a sequence of matplotlib figures and stores
them in a PDF file.  Basic metadata such as title, creation date and the
command used to invoke the program are embedded in the resulting document.  The
path to the created PDF is returned to the caller.
"""

from __future__ import annotations


from datetime import datetime
import sys
from pathlib import Path
from typing import Iterable

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure


def write(figures: Iterable[Figure], out_path: Path, title: str) -> Path:
    """Write ``figures`` to ``out_path`` and return that path.

    Parameters
    ----------
    figures:
        Iterable of matplotlib :class:`~matplotlib.figure.Figure` instances.  At
        least one figure must be supplied.
    out_path:
        Target location where the PDF should be written.
    title:
        Title embedded into the PDF metadata.

    Returns
    -------
    pathlib.Path
        Location of the generated PDF document.
    """

    figures = list(figures)
    if not figures:
        raise ValueError("No figures supplied")

    # Derive metadata
    creation_date = datetime.now().strftime("%Y-%m-%d")
    cmd = " ".join(sys.argv)

    metadata = {"Title": title, "CreationDate": creation_date, "Creator": cmd}

    with PdfPages(out_path, metadata=metadata) as pdf:
        for fig in figures:
            pdf.savefig(fig)

    return out_path