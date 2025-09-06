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


def normalize_group(group: Optional[str]) -> Optional[str]:
    """Return *group* without a trailing numeric suffix.

    The report pipeline often uses naming schemes such as ``g1`` or
    ``experiment2`` where the digits merely indicate repetitions.  For the
    purpose of grouping we want to treat ``g1`` and ``g2`` as the same group
    ``g``.  This helper therefore strips a trailing sequence of digits from the
    provided *group* and returns the simplified name.  ``None`` is returned
    unchanged.
    """

    if group is None:
        return None
    return re.sub(r"\d+$", "", group)


@dataclass(frozen=True)
class DistanceFile:
    """A distance file on disk along with its label and optional group."""

    path: Path
    label: str
    group: Optional[str] = None


@dataclass(frozen=True)
class PlotJob:
    """Data required to produce plots for a list of distance files."""

    items: list[DistanceFile]
    page_title: Optional[str] = None
    plot_type: str = "histogram"


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
        label = m.group("label")
        group = normalize_group(m.group("group"))
        return label, group
    return path.stem, None
