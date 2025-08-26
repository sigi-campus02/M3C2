from __future__ import annotations
import numpy as np
import py4dgeo
from typing import Tuple


class M3C2Runner:
    def run(
        self,
        mov: py4dgeo.Epoch,
        ref: py4dgeo.Epoch,
        corepoints: np.ndarray,
        normal: float,
        project: float,
    ) -> Tuple[np.ndarray, np.ndarray]:
        m3c2 = py4dgeo.M3C2(
            epochs=(mov, ref),
            corepoints=corepoints,
            cyl_radius=project,     
            normal_radii=[normal], 
        )
        distances, uncertainties = m3c2.run()
        return distances, uncertainties
