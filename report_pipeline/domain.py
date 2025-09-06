"""Domain models used by the report pipeline.

This module defines lightweight dataclasses representing distance files and
plotting jobs.  A small helper ``parse_label_group`` is provided to derive a
label and optional group from a file path.  When no group information can be
extracted the function simply uses the file stem as label and returns
``None`` for the group.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Optional, Tuple

# Regular expression matching ``<label>__<group>`` or ``<label>--<group>`` where
# the group part is optional.  Using a non-greedy match for the label ensures
# that only the first occurrence of the separator is considered.
_LABEL_GROUP_RE = re.compile(r"^(?P<label>.+?)(?:[-_]{2}(?P<group>.+))?$")


@dataclass(frozen=True)
class DistanceFile:
    """A distance file on disk along with its label and optional group."""

    path: Path
    label: str
    group: Optional[str] = None


@dataclass(frozen=True)
class PlotJob:
    """Description of a single overlay plot.

    The job merely collects :class:`DistanceFile` instances which are later
    loaded by the plotting layer.  Builders are intentionally lightweight and
    therefore never read the referenced files.
    """

    items: list[DistanceFile]
    page_title: Optional[str] = None


def parse_label_group(path: Path) -> Tuple[str, Optional[str]]:
    """Return the label and optional group encoded in *path*.

    The helper extracts the label and group from the stem of *path*.  The
    stem is expected to follow the pattern ``<label>__<group>`` (double
    underscore) or ``<label>--<group>`` (double hyphen).  If neither pattern
    matches, the function falls back to using the stem itself as the label and
    returns ``None`` as the group.

    Parameters
    ----------
    path:
        Path to the file whose stem should be parsed.

    Returns
    -------
    tuple[str, Optional[str]]
        The resolved label and group.  The group value is ``None`` when no
        group information could be detected in the filename.
    """

    m = _LABEL_GROUP_RE.match(path.stem)
    if m:
        return m.group("label"), m.group("group")
    return path.stem, None
