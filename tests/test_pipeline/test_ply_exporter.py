import numpy as np
import pytest
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


def test_export_xyz_distance_missing_dependency(monkeypatch, tmp_path):
    points = np.zeros((2, 3), dtype=np.float32)
    distances = np.zeros(2, dtype=np.float32)
    outliers = np.zeros(2, dtype=np.uint8)
    outply = tmp_path / "cloud.ply"

    monkeypatch.setattr("m3c2.exporter.ply_exporter.PlyData", None)
    monkeypatch.setattr("m3c2.exporter.ply_exporter.PlyElement", None)

    with pytest.raises(RuntimeError):
        export_xyz_distance(points, distances, outliers, str(outply))

    assert not outply.exists()


def test_export_xyz_distance_invalid_points_shape(tmp_path):
    points = np.zeros((2, 2), dtype=np.float32)
    distances = np.zeros(2, dtype=np.float32)
    outliers = np.zeros(2, dtype=np.uint8)
    outply = tmp_path / "cloud.ply"

    with pytest.raises(ValueError):
        export_xyz_distance(points, distances, outliers, str(outply))

    assert not outply.exists()


def test_export_xyz_distance_invalid_distances_length(tmp_path):
    points = np.zeros((2, 3), dtype=np.float32)
    distances = np.zeros(3, dtype=np.float32)
    outliers = np.zeros(2, dtype=np.uint8)
    outply = tmp_path / "cloud.ply"

    with pytest.raises(ValueError):
        export_xyz_distance(points, distances, outliers, str(outply))

    assert not outply.exists()


def test_export_xyz_distance_invalid_outliers_length(tmp_path):
    points = np.zeros((2, 3), dtype=np.float32)
    distances = np.zeros(2, dtype=np.float32)
    outliers = np.zeros(3, dtype=np.uint8)
    outply = tmp_path / "cloud.ply"

    with pytest.raises(ValueError):
        export_xyz_distance(points, distances, outliers, str(outply))

    assert not outply.exists()

