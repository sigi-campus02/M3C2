"""Test the default values and immutability of ``PipelineConfig``.

This module verifies that a ``PipelineConfig`` instance exposes expected
default values and behaves as an immutable data structure.
"""

import sys
from pathlib import Path

import pytest
from dataclasses import FrozenInstanceError

# Ensure package root is on sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from m3c2.config.pipeline_config import PipelineConfig


def test_defaults_and_immutability():
    """Verify default values and immutability of ``PipelineConfig``.

    Ensures the configuration object contains the expected default values
    and that its fields remain frozen after instantiation.

    Parameters
    ----------
    None

    Returns
    -------
    None
    """
    config = PipelineConfig(
        data_dir="data",
        folder_id="folder",
        filename_mov="mov.las",
        filename_ref="ref.las",
        filename_singlecloud="single.las",
        mov_as_corepoints=True,
        use_subsampled_corepoints=0,
        only_stats=False,
        stats_singleordistance="single",
        sample_size=10,
        project="demo",
    )

    # Default attributes
    assert config.normal_override is None
    assert config.proj_override is None
    assert config.use_existing_params is False
    assert config.outlier_multiplicator == 3.0
    assert config.outlier_detection_method == "rmse"
    assert config.process_python_CC == "python"
    assert config.output_format == "excel"
    assert config.log_level == "INFO"

    # Fields are frozen
    with pytest.raises(FrozenInstanceError):
        config.project = "other"
