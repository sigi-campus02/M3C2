from pathlib import Path
from pathlib import Path
from unittest.mock import MagicMock, call

from report_pipeline.domain import DistanceFile
from report_pipeline.plotting import figure_factory


def test_by_folder_color_strategy_assigns_same_color(monkeypatch):
    items = [
        DistanceFile(path=Path("g1/a.txt"), label="a"),
        DistanceFile(path=Path("g2/b.txt"), label="b"),
        DistanceFile(path=Path("g1/c.txt"), label="c"),
    ]
    monkeypatch.setattr(figure_factory, "load_distance_series", lambda p: [1, 2, 3])
    fig = MagicMock()
    ax = MagicMock()
    monkeypatch.setattr(figure_factory.plt, "subplots", lambda: (fig, ax))

    figure_factory.make_overlay(items, color_mapping="by_folder")
    colors = [call.kwargs.get("color") for call in ax.hist.call_args_list]
    assert colors[0] == colors[2]
    assert colors[0] != colors[1]


def test_by_label_strategy_deterministic_across_calls(monkeypatch):
    items1 = [
        DistanceFile(path=Path("a.txt"), label="a"),
        DistanceFile(path=Path("b.txt"), label="b"),
    ]
    items2 = [DistanceFile(path=Path("b.txt"), label="b")]
    monkeypatch.setattr(figure_factory, "load_distance_series", lambda p: [1, 2, 3])

    axes: list[MagicMock] = []

    def fake_subplots():
        fig = MagicMock()
        ax = MagicMock()
        axes.append(ax)
        return fig, ax

    monkeypatch.setattr(figure_factory.plt, "subplots", fake_subplots)

    figure_factory.make_overlay(items1, color_mapping="by_label")
    figure_factory.make_overlay(items2, color_mapping="by_label")

    first = axes[0].hist.call_args_list[1].kwargs["color"]
    second = axes[1].hist.call_args_list[0].kwargs["color"]
    assert first == second
