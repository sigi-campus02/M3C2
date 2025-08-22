import sys
import tempfile
from pathlib import Path
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))
from datasource import DataSource


def test_obj_detection_and_conversion():
    with tempfile.TemporaryDirectory() as tmp:
        obj_file = Path(tmp) / "mov.obj"
        obj_file.write_text("v 0 0 0\n v 1 2 3\n", encoding="utf-8")
        ds = DataSource(tmp)
        kind, path = ds._detect(ds.mov_base)
        assert kind == "obj"
        xyz_path = ds._ensure_xyz(ds.mov_base, (kind, path))
        arr = np.loadtxt(xyz_path)
        assert arr.shape == (2, 3)
        assert np.allclose(arr, np.array([[0, 0, 0], [1, 2, 3]]))


def test_gpc_detection_and_conversion():
    with tempfile.TemporaryDirectory() as tmp:
        gpc_file = Path(tmp) / "ref.gpc"
        gpc_file.write_text("0 0 0\n1 2 3\n", encoding="utf-8")
        ds = DataSource(tmp)
        kind, path = ds._detect(ds.ref_base)
        assert kind == "gpc"
        xyz_path = ds._ensure_xyz(ds.ref_base, (kind, path))
        arr = np.loadtxt(xyz_path)
        assert arr.shape == (2, 3)
        assert np.allclose(arr, np.array([[0, 0, 0], [1, 2, 3]]))
