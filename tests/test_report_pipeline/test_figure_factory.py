from pathlib import Path
from unittest.mock import MagicMock

from report_pipeline.domain import DistanceFile
from report_pipeline.plotting import figure_factory
import numpy as np

def test_make_overlay_splits_by_max_per_page(monkeypatch):
    items = [DistanceFile(path=Path(f"{i}.txt"), label=str(i)) for i in range(7)]
    monkeypatch.setattr(figure_factory, "load_distance_series", lambda p: [1, 2, 3])

    axes: list[MagicMock] = []

    def fake_subplots():
        fig = MagicMock()
        ax = MagicMock()
        axes.append(ax)
        return fig, ax

    monkeypatch.setattr(figure_factory.plt, "subplots", fake_subplots)

    figs = figure_factory.make_overlay(items, max_per_page=4)
    assert len(figs) == 2
    assert len(axes[0].hist.call_args_list) == 4
    assert len(axes[1].hist.call_args_list) == 3


def test_make_bland_altman(monkeypatch):
    items = [
        DistanceFile(path=Path("a.txt"), label="A"),
        DistanceFile(path=Path("b.txt"), label="B"),
    ]
    data = {
        "a.txt": np.array([1.0, 2.0, 3.0]),
        "b.txt": np.array([1.1, 2.1, 3.1]),
    }
    monkeypatch.setattr(
        figure_factory,
        "load_distance_series",
        lambda p: data[p.name],
    )
    figs = figure_factory.make_bland_altman(items)
    assert len(figs) == 1


def test_make_linear_regression(monkeypatch):
    items = [
        DistanceFile(path=Path("a.txt"), label="A"),
        DistanceFile(path=Path("b.txt"), label="B"),
    ]
    data = {
        "a.txt": np.array([1.0, 2.0, 3.0, 4.0]),
        "b.txt": np.array([1.2, 2.1, 3.9, 4.2]),
    }
    monkeypatch.setattr(figure_factory, "load_distance_series", lambda p: data[p.name])
    figs = figure_factory.make_linear_regression(items)
    assert len(figs) == 1


def test_make_passing_bablok(monkeypatch):
    items = [
        DistanceFile(path=Path("a.txt"), label="A"),
        DistanceFile(path=Path("b.txt"), label="B"),
    ]
    data = {
        "a.txt": np.array([1.0, 2.0, 3.0, 4.0, 5.0]),
        "b.txt": np.array([1.1, 2.2, 3.1, 4.2, 5.1]),
    }
    monkeypatch.setattr(figure_factory, "load_distance_series", lambda p: data[p.name])
    figs = figure_factory.make_passing_bablok(items)
    assert len(figs) == 1
