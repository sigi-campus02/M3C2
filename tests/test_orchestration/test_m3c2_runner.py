"""Tests for orchestration-level utilities for the M3C2 runner."""

from orchestration.m3c2_runner import M3C2Runner
from m3c2.core.m3c2_runner import M3C2Runner as CoreM3C2Runner


def test_reexport_m3c2_runner():
    """Verify that the orchestration runner re-exports the core implementation.

    This test ensures that :class:`orchestration.m3c2_runner.M3C2Runner` is the
    same class as :class:`m3c2.core.m3c2_runner.M3C2Runner`.

    Returns
    -------
    None
        This test does not return a value.
    """

    assert M3C2Runner is CoreM3C2Runner
