import numpy as np
from plyfile import PlyData

from m3c2.exporter.ply_exporter import export_xyz_distance


def test_export_xyz_distance(tmp_path):
    points = np.array([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=np.float32)
    distances = np.array([np.nan, 1.5], dtype=np.float32)
    outliers = np.array([0, 1], dtype=np.uint8)

    outply = tmp_path / "cloud.ply"

    export_xyz_distance(points, distances, outliers, str(outply))

    ply = PlyData.read(outply)
    vertex = ply["vertex"]

    assert vertex.count == 2
    assert np.isnan(vertex["distance"][0])
    assert np.isclose(vertex["distance"][1], 1.5)
    np.testing.assert_array_equal(vertex["outlier"], outliers)

    np.testing.assert_array_equal(vertex["x"], points[:, 0])
    np.testing.assert_array_equal(vertex["y"], points[:, 1])
    np.testing.assert_array_equal(vertex["z"], points[:, 2])

