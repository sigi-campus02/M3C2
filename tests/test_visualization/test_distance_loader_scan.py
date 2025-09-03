"""Unit tests for scanning M3C2 distance files by index."""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2]))

import numpy as np
from m3c2.visualization.loaders.distance_loader import scan_distance_files_by_index


def _write_file(path, content: str) -> None:
    """Write text content to a file.

    Parameters
    ----------
    path : pathlib.Path
        Destination path of the file.
    content : str
        Text content to write.
    """
    path.write_text(content)


def test_scan_distance_files_by_index(tmp_path):
    """Verify parsing of distance and inlier files by index.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.
    """
    # valid WITH file for index 1
    _write_file(tmp_path / 'python_a-1-b-1_m3c2_distances.txt', '1.0\n2.0\n3.0\n')
    # valid INLIER file for index 1
    _write_file(
        tmp_path / 'CC_a-1-b-1_m3c2_distances_coordinates_inlier_method.txt',
        'x y z distance\n0 0 0 1.0\n1 1 1 2.0\n',
    )
    # mismatched index should be ignored
    _write_file(tmp_path / 'python_a-1-b-2_m3c2_distances.txt', '0.0\n')
    # invalid INLIER file (too few columns) should be skipped gracefully
    _write_file(
        tmp_path / 'python_a-1-b-1_m3c2_distances_coordinates_inlier_bad.txt',
        'x y z\n0 0 0\n',
    )

    per_index, case_colors = scan_distance_files_by_index(str(tmp_path))

    assert 1 in per_index
    data = per_index[1]
    assert 'a-1 vs b-1' in data['WITH']
    assert 'a-1 vs b-1' in data['INLIER']
    np.testing.assert_array_equal(data['WITH']['a-1 vs b-1'], np.array([1.0, 2.0, 3.0]))
    np.testing.assert_array_equal(data['INLIER']['a-1 vs b-1'], np.array([1.0, 2.0]))
    assert data['CASE_WITH']['a-1 vs b-1'] == 'CASE1'
    assert 'CASE1' in case_colors
    assert 2 not in per_index
