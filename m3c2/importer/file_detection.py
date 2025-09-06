"""Helper utilities for detecting point cloud file formats."""

from __future__ import annotations


from pathlib import Path
import logging

logger = logging.getLogger(__name__)


def detect(base: Path) -> tuple[str | None, Path | None]:
    """Detect available point cloud files for a given base path.

    Parameters
    ----------
    base:
        Path without an extension that serves as the candidate stem for
        supported point cloud file formats.

    Returns
    -------
    tuple[str | None, Path | None]
        A pair ``(kind, path)`` where ``kind`` identifies the detected
        format (e.g. ``"xyz"`` or ``"laslike"``) and ``path`` points to the
        discovered file. ``(None, None)`` is returned if no supported file
        exists.
    """

    logger.debug("Detecting file type for base %s", base)

    mapping = {
        "xyz": base.with_suffix(".xyz"),
        "las": base.with_suffix(".las"),
        "laz": base.with_suffix(".laz"),
        "ply": base.with_suffix(".ply"),
        "obj": base.with_suffix(".obj"),
        "gpc": base.with_suffix(".gpc"),
    }

    if mapping["xyz"].exists():
        logger.debug("Detected XYZ file at %s", mapping["xyz"])
        return "xyz", mapping["xyz"]
    if mapping["las"].exists() or mapping["laz"].exists():
        path = mapping["las"] if mapping["las"].exists() else mapping["laz"]
        logger.debug("Detected LAS/LAZ file at %s", path)
        return "laslike", path
    if mapping["ply"].exists():
        logger.debug("Detected PLY file at %s", mapping["ply"])
        return "ply", mapping["ply"]
    if mapping["obj"].exists():
        logger.debug("Detected OBJ file at %s", mapping["obj"])
        return "obj", mapping["obj"]
    if mapping["gpc"].exists():
        logger.debug("Detected GPC file at %s", mapping["gpc"])
        return "gpc", mapping["gpc"]

    logger.debug("No supported file detected for %s", base)
    return None, None


__all__ = ["detect"]