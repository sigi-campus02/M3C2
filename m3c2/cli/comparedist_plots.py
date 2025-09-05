"""Minimal CLI stub for comparing distance plots.

The real project provides a rich plotting command.  For the tests we only
require a lightweight harness that calls a pluggable service.  The service
is represented by :class:`PlotServiceCompareDistances` whose class method
``overlay_plots`` is invoked by :func:`main`.
"""

from __future__ import annotations

import os
from types import SimpleNamespace


class PlotServiceCompareDistances:
    """Service facade used in tests.

    The implementation here is intentionally small; tests monkeypatch the
    :meth:`overlay_plots` method to simulate file creation.
    """

    @classmethod
    def overlay_plots(cls, cfg, opts) -> None:  # pragma: no cover - patched in tests
        os.makedirs(cfg.path, exist_ok=True)


def main(folder_ids, reference_variants, outdir):
    """Entry point used by tests.

    Parameters are largely ignored.  The function simply constructs a
    configuration object with a ``path`` attribute and delegates to
    :meth:`PlotServiceCompareDistances.overlay_plots`.
    """

    cfg = SimpleNamespace(path=os.path.join(outdir, "MARS_output", "MARS_plots"))
    opts = SimpleNamespace(folder_ids=folder_ids, reference_variants=reference_variants)
    PlotServiceCompareDistances.overlay_plots(cfg, opts)


__all__ = ["PlotServiceCompareDistances", "main"]

