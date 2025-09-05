"""Tests for the pipeline configuration module.

These tests validate default parameter values and the immutability of
``PipelineConfig`` instances to ensure consistent behavior across runs.
"""

import sys
from pathlib import Path

import pytest
from dataclasses import FrozenInstanceError

# Ensure package root is on sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from m3c2.config.pipeline_config import PipelineConfig


def test_defaults_and_immutability():
    config = PipelineConfig(
        data_dir="data",
        folder_id="folder",
        filename_comparison="comparison.las",
        filename_reference="reference.las",
        filename_singlecloud="single.las",
        comparison_as_corepoints=True,
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
