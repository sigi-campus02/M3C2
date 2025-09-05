"""Tests for the :mod:`m3c2_runner` module.

This module focuses on validating that the :class:`M3C2Runner` correctly
interacts with the underlying ``py4dgeo`` library.
"""

import sys
import types
import importlib
import numpy as np
from m3c2.m3c2_core.m3c2_runner import M3C2Runner

def test_m3c2_runner(monkeypatch):
    """Test the M3C2 runner with a dummy ``py4dgeo`` implementation.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to replace the ``py4dgeo`` module with a dummy
        implementation.

    """
    called = {}

    class DummyM3C2:
        def __init__(self, **kwargs):
            called['params'] = (
                kwargs['epochs'],
                kwargs['corepoints'],
                kwargs['cyl_radius'],
                kwargs['normal_radii'],
            )

        def run(self):
            return np.array([1.0]), np.array([0.1])

    dummy_py4dgeo = types.SimpleNamespace(M3C2=DummyM3C2)
    monkeypatch.setitem(sys.modules, 'py4dgeo', dummy_py4dgeo)

    comparison = object()
    reference = object()
    corepoints = np.array([[0.0, 0.0, 0.0]])

    distances, uncertainties = M3C2Runner.run(
        comparison, reference, corepoints, normal=0.5, projection=1.0
    )

    assert np.allclose(distances, [1.0])
    assert np.allclose(uncertainties, [0.1])
    assert called['params'] == ((comparison, reference), corepoints, 1.0, [0.5])
