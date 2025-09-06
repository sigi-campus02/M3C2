"""Tests for filename and plot configuration modules.

These tests cover default values, validation behaviour and override handling
for :mod:`m3c2.config.filenames_config` and :mod:`m3c2.config.plot_config`.
"""

import os
import sys
from dataclasses import FrozenInstanceError
from pathlib import Path
from types import SimpleNamespace

import pytest

# Ensure package root on sys.path
sys.path.append(str(Path(__file__).resolve().parents[2]))

from m3c2.config.filenames_config import FileNameParams, FileNames
from m3c2.config.plot_config import (
    PlotConfig,
    PlotOptions,
    PlotOptionsComparedistances,
)
from m3c2.config.pipeline_config import PipelineConfig


def _make_pipeline_cfg(**updates):
    """Helper to construct a minimal :class:`PipelineConfig`."""

    data = dict(
        data_dir="data",
        folder_id="fid",
        filename_comparison="comp.las",
        filename_reference="ref.las",
        filename_singlecloud="single.las",
        use_subsampled_corepoints=0,
        only_stats=False,
        stats_singleordistance="single",
        sample_size=10,
        project="proj",
    )
    data.update(updates)
    return PipelineConfig(**data)


# ---------------------------------------------------------------------------
# Plot option structures
# ---------------------------------------------------------------------------

def test_plot_options_defaults_and_immutability():
    opts = PlotOptions()
    assert opts.plot_hist is True
    assert opts.plot_gauss is True
    assert opts.plot_weibull is True
    assert opts.plot_box is True
    assert opts.plot_qq is True
    assert opts.plot_grouped_bar is True
    assert opts.plot_violin is True

    with pytest.raises(FrozenInstanceError):
        opts.plot_hist = False


def test_plot_options_comparedistances_defaults_and_immutability():
    opts = PlotOptionsComparedistances()
    assert opts.plot_blandaltman is True
    assert opts.plot_passingbablok is True
    assert opts.plot_linearregression is True

    with pytest.raises(FrozenInstanceError):
        opts.plot_blandaltman = False


# ---------------------------------------------------------------------------
# PlotConfig
# ---------------------------------------------------------------------------

def test_plot_config_defaults_and_color_generation():
    cfg = PlotConfig(
        folder_ids=["1"],
        filenames=["dist.txt"],
        project="proj",
        outdir="out",
        versions=["v1", "v2"],
    )

    assert cfg.bins == 256
    assert cfg.colors == {}
    assert cfg.path == os.path.join("out", "proj_output", "proj_plots")

    labels = cfg.labels()
    assert labels == ["v1_dist.txt", "v2_dist.txt"]

    colors = cfg.ensure_colors()
    assert set(colors.keys()) == set(labels)


def test_plot_config_color_override_and_bins():
    custom = {"v1_dist.txt": "#ffffff"}
    cfg = PlotConfig(
        folder_ids=["1"],
        filenames=["dist.txt"],
        project="proj",
        outdir="out",
        versions=["v1"],
        colors=custom,
        bins=128,
    )

    assert cfg.bins == 128
    assert cfg.ensure_colors() == custom


def test_plot_config_missing_required_field():
    with pytest.raises(TypeError):
        PlotConfig(
            filenames=["dist.txt"],
            project="proj",
            outdir="out",
            versions=["v1"],
        )


def test_plot_config_labels_missing_versions():
    cfg = PlotConfig(
        folder_ids=["1"],
        filenames=["dist.txt"],
        project="proj",
        outdir="out",
    )  # versions defaults to None
    with pytest.raises(TypeError):
        cfg.labels()


# ---------------------------------------------------------------------------
# File name handling
# ---------------------------------------------------------------------------

def test_filename_params_from_config_single_and_distance():
    cfg_single = _make_pipeline_cfg(stats_singleordistance="single")
    params_single = FileNameParams.from_config(cfg_single)
    assert params_single.tag == "single.las"
    assert params_single.prefix == "proj"
    assert params_single.fid == "fid"
    assert params_single.method == "rmse"

    with pytest.raises(FrozenInstanceError):
        params_single.prefix = "other"

    cfg_dist = _make_pipeline_cfg(stats_singleordistance="distance")
    params_dist = FileNameParams.from_config(cfg_dist)
    assert params_dist.tag == "comp.las-ref.las"


def test_filename_params_missing_attribute():
    bad_cfg = SimpleNamespace(
        stats_singleordistance="single",
        filename_comparison="comp",
        filename_reference="ref",
        project="proj",
        folder_id="fid",
        outlier_detection_method="rmse",
    )  # missing filename_singlecloud
    with pytest.raises(AttributeError):
        FileNameParams.from_config(bad_cfg)


def test_file_names_stats_distances_extension_validation():
    assert FileNames.stats_distances("proj", "json").endswith("json")
    with pytest.raises(ValueError):
        FileNames.stats_distances("proj", "csv")
