"""Stub CLI for creating comparison plots.

The real project provides an extensive plotting service which is not required
for the unit tests in this kata.  This lightweight module only contains the
minimal hooks exercised by the tests: a ``PlotServiceCompareDistances`` class
with an ``overlay_plots`` classmethod and a ``main`` function that forwards the
provided arguments to this method.
"""

from __future__ import annotations

import os
from types import SimpleNamespace
from typing import Iterable


class PlotServiceCompareDistances:
    """Placeholder plotting service used in tests."""

    @classmethod
    def overlay_plots(cls, cfg, opts):  # pragma: no cover - patched in tests
        pass


def main(
    *,
    folder_ids: Iterable[str],
    ref_variants: Iterable[str],
    outdir: str,
) -> None:
    """Invoke :meth:`PlotServiceCompareDistances.overlay_plots`.

    Parameters mirror those of the original CLI but are intentionally
    simplified.  A small configuration object exposing a ``path`` attribute is
    created and passed together with the options to the plotting service.
    """

    cfg = SimpleNamespace(path=os.path.join(outdir, "MARS_output", "MARS_plots"))
    opts = SimpleNamespace(folder_ids=list(folder_ids), ref_variants=list(ref_variants))
    PlotServiceCompareDistances.overlay_plots(cfg, opts)


__all__ = ["PlotServiceCompareDistances", "main"]

