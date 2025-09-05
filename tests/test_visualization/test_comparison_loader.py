"""Tests for the comparison loader utilities."""

import numpy as np
from m3c2.visualization.loaders.comparison_loader import _load_and_mask


def test_load_and_mask_removes_nan(tmp_path):
    """Ensure NaN entries are removed by :func:`_load_and_mask`.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.
    """

    fid_dir = tmp_path / "fid1"
    fid_dir.mkdir()
    (fid_dir / "python_reference_m3c2_distances.txt").write_text("0.0\n1.0\nnan\n2.0\n")
    (fid_dir / "python_reference_ai_m3c2_distances.txt").write_text("0.1\n1.1\n2.1\n3.1\n")
    result = _load_and_mask(str(fid_dir), ["reference", "reference_ai"])
    assert result is not None
    a, b = result
    assert len(a) == len(b) == 3
    assert not np.isnan(a).any()
    assert not np.isnan(b).any()
