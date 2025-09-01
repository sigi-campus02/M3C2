from orchestration.m3c2_runner import M3C2Runner
from m3c2.core.m3c2_runner import M3C2Runner as CoreM3C2Runner


def test_reexport_m3c2_runner():
    assert M3C2Runner is CoreM3C2Runner
