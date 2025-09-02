"""Tests for overlay plotting visualization utilities."""

import numpy as np
import matplotlib
matplotlib.use("Agg")

from m3c2.visualization.overlay_plotter import (
    get_common_range,
    plot_overlay_histogram,
    plot_overlay_gauss,
    plot_overlay_weibull,
    plot_overlay_boxplot,
    plot_overlay_qq,
)


def test_get_common_range_empty():
    """Verify handling of empty data in common range computation.

    Returns
    -------
    None
        The function asserts expected behavior and returns ``None``.
    """

    data_min, data_max, x = get_common_range({})
    assert data_min == 0.0
    assert data_max == 1.0
    assert len(x) == 500


def test_plot_functions_create_files(tmp_path):
    """Ensure overlay plotting functions generate the correct files.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.

    Returns
    -------
    None
        This function does not return a value.
    """

    data = {
        "A": np.array([0.0, 1.0, 2.0]),
        "B": np.array([1.0, 2.0, 3.0]),
    }
    colors = {"A": "#ff0000", "B": "#00ff00"}
    data_min, data_max, x = get_common_range(data)
    gauss_params = {k: (float(np.mean(v)), float(np.std(v))) for k, v in data.items()}

    plot_overlay_histogram("f", "name", data, 10, data_min, data_max, colors, str(tmp_path))
    assert (tmp_path / "f_name_OverlayHistogramm.png").is_file()

    plot_overlay_gauss("f", "name", data, gauss_params, x, colors, str(tmp_path))
    assert (tmp_path / "f_name_OverlayGaussFits.png").is_file()

    plot_overlay_weibull("f", "name", data, x, colors, str(tmp_path))
    assert (tmp_path / "f_name_OverlayWeibullFits.png").is_file()

    plot_overlay_boxplot("f", "name", data, colors, str(tmp_path))
    assert (tmp_path / "f_name_Boxplot.png").is_file()

    plot_overlay_qq("f", "name", data, colors, str(tmp_path))
    assert (tmp_path / "f_name_QQPlot.png").is_file()


def test_plot_overlay_boxplot_empty(tmp_path):
    """Test plotting behavior when data input is empty.

    Parameters
    ----------
    tmp_path : pathlib.Path
        Temporary directory provided by pytest.

    Returns
    -------
    None
        The function verifies file creation without returning a value.
    """

    # Should not raise and produce no file when data is empty
    plot_overlay_boxplot("f", "name", {}, {}, str(tmp_path))
    assert not (tmp_path / "f_name_Boxplot.png").exists()

    # Histogram should still create an empty file
    data_min, data_max, _ = get_common_range({})
    plot_overlay_histogram("f", "name", {}, 10, data_min, data_max, {}, str(tmp_path))
    assert (tmp_path / "f_name_OverlayHistogramm.png").is_file()
