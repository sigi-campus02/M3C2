"""Small helpers for resolving data paths used by statistics modules."""

from __future__ import annotations

import os


def _resolve(fid: str, filename: str) -> str:
    """Resolve the path for a statistics file.

    The function first checks whether ``filename`` exists inside the
    directory identified by ``fid``.  If the file is found there, that
    path is returned.  Otherwise, the function falls back to the
    repository's ``data`` directory and constructs the path as
    ``data/fid/filename``.
    """

    p1 = os.path.join(fid, filename)
    if os.path.exists(p1):
        return p1
    return os.path.join("data", fid, filename)


__all__ = ["_resolve"]

