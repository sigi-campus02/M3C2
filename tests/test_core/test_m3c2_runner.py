import sys
import types
import importlib
import numpy as np


def test_m3c2_runner(monkeypatch):
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

    runner_module = importlib.reload(
        importlib.import_module('orchestration.m3c2_runner')
    )

    mov = object()
    ref = object()
    corepoints = np.array([[0.0, 0.0, 0.0]])

    distances, uncertainties = runner_module.M3C2Runner.run(
        mov, ref, corepoints, normal=0.5, projection=1.0
    )

    assert np.allclose(distances, [1.0])
    assert np.allclose(uncertainties, [0.1])
    assert called['params'] == ((mov, ref), corepoints, 1.0, [0.5])
