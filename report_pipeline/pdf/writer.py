from __future__ import annotations

"""Utilities to persist matplotlib figures as a PDF document.

The :func:`write` function accepts a sequence of matplotlib figures and stores
them in a temporary PDF file.  Basic metadata such as title, creation date and
the command used to invoke the program are embedded in the resulting document.
The path to the created PDF is returned to the caller.
"""

from datetime import datetime
import os
import sys
import tempfile
from pathlib import Path
from typing import Iterable

from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure


def write(figures: Iterable[Figure], out: Path | None = None) -> Path:
    """Write ``figures`` to a PDF file and return its path.

    Parameters
    ----------
    figures:
        Iterable of matplotlib :class:`~matplotlib.figure.Figure` instances.  At
        least one figure must be supplied.  The first figure is inspected for a
        ``suptitle`` which is used as the document title.
    out:
        Optional path to the resulting PDF file.  When omitted a temporary file
        is created and its location returned.

    Returns
    -------
    pathlib.Path
        Location of the generated PDF document.  When *out* is provided the
        PDF is written to that path, otherwise a temporary file is created.
    """

    figures = list(figures)
    if not figures:
        raise ValueError("No figures supplied")

    # Derive metadata
    title = None
    first = figures[0]
    if first._suptitle is not None:  # type: ignore[attr-defined]
        title = first._suptitle.get_text()  # type: ignore[union-attr]
    title = title or "Report"
    creation_date = datetime.now().strftime("%Y-%m-%d")
    cmd = " ".join(sys.argv)

    metadata = {"Title": title, "CreationDate": creation_date, "Creator": cmd}

    if out is None:
        fd, tmp = tempfile.mkstemp(suffix=".pdf")
        os.close(fd)
        pdf_path = Path(tmp)
    else:
        pdf_path = Path(out)
        pdf_path.parent.mkdir(parents=True, exist_ok=True)

    with PdfPages(pdf_path, metadata=metadata) as pdf:
        for fig in figures:
            pdf.savefig(fig)

    return pdf_path
